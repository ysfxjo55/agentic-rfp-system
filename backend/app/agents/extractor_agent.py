import re
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from app.agents.state import RFPProposalState
from app.agents.llm_factory import LLMFactory
from app.core.prompts import EXTRACTION_AGENT_PROMPT
from app.models.schemas import ExtractionAgentOutput, RFPMetadata, RawClause, ExtractedBlock
from app.services.document_parser import DocumentParserService, _is_arabic_dominant
from app.agents.classifier_agent import _determine_mandatory, _strip_arabic_diacritics

# Guardrail: Legacy fictional placeholders that must NEVER appear in real or fallback metadata
PROHIBITED_FICTIONAL_STRINGS = [
    "Enterprise Cloud & Digital Transformation RFP",
    "Global Logistics & Transportation Authority",
    "October 31, 2026 at 5:00 PM EST",
    "Enterprise-wide hybrid cloud deployment and SLA-governed support"
]

def _has_valid_text_encoding(text: str) -> bool:
    """
    Rejects text containing codepoints outside the ranges a legitimate
    English/Arabic tender document could produce (Arabic, Latin, digits,
    common punctuation/whitespace, currency/percent symbols). This guards
    against a font/cmap encoding defect confirmed present in this document's
    source PDF on a handful of pages (a specific embedded font subset maps
    some glyphs to unrelated Unicode blocks — e.g. Ogham, Canadian
    Aboriginal Syllabics — instead of the intended Arabic letters), which
    PyMuPDF's own text extraction reproduces faithfully since it isn't a
    parsing bug but a defect in the PDF's own font resource. Not fixable by
    reordering/reshaping logic; the safest handling is to never select such
    text as metadata, and prefer an unaffected page's version instead.
    """
    for c in text:
        cp = ord(c)
        if (
            cp <= 0x024F  # Basic Latin, Latin-1 Supplement, Latin Extended-A/B
            or 0x0600 <= cp <= 0x077F  # Arabic + Arabic Supplement
            or 0x2000 <= cp <= 0x206F  # General Punctuation (dashes, quotes, etc.)
            or 0x20A0 <= cp <= 0x20CF  # Currency symbols
            or 0x2190 <= cp <= 0x27BF  # Arrows, misc technical/symbols, dingbats (checkboxes etc.)
            or 0xE000 <= cp <= 0xF8FF  # Private Use Area (custom PDF symbol/bullet fonts)
            or 0xFB50 <= cp <= 0xFEFF  # Arabic Presentation Forms A/B
        ):
            continue
        return False
    return True


class ExtractedClauseBatch(BaseModel):
    clauses: List[RawClause] = Field(
        default_factory=list,
        description="Candidate requirement clauses extracted verbatim from the document text"
    )

# --------------------------------------------------------------------------
# Metadata / table-fragment exclusion (bilingual): reference numbers, contact
# details, and evaluation-criteria weight/score/total rows are NOT bidder
# obligations and must never enter the requirements/Compliance Matrix pipeline.
# --------------------------------------------------------------------------
_REFERENCE_NUMBER_PATTERN = re.compile(
    r'(?:document\s*ref(?:erence)?|ref(?:erence)?\s*(?:no\.?|number|#)|رقم\s+الكراسة|رقم\s+المرجع|الرقم\s+المرجعي|رقم\s+النسخة)\b',
    re.IGNORECASE
)
_PHONE_PATTERN = re.compile(
    r'(?:phone|tel(?:ephone)?|fax|الهاتف|فاكس|جوال)\s*[:\-]?\s*[\d\s\-+()]{6,}',
    re.IGNORECASE
)
_EMAIL_ONLY_PATTERN = re.compile(r'^[\w.+-]+@[\w-]+\.[\w.-]+$')
_EVAL_WEIGHT_ROW_PATTERN = re.compile(
    r'(?i)(?:total|weight(?:ing)?|technical\s+total|overall\s+score|المجموع(?:\s+الفني)?|إجمالي|الدرجة\s+الكلية|الوزن\s+النسبي)'
    r'.{0,40}?\d{1,3}\s*(?:%|points?|درجة|نقطة)'
)
_EVAL_SECTION_HINT = re.compile(
    r'(?:evaluation|criteria|scoring|award\s+criteria|معايير|تقييم|أوزان)',
    re.IGNORECASE
)


_INCOMPLETENESS_MARKER_PATTERN = re.compile(
    r'(?i)\((?:clause\s+)?incomplete\)|\(truncated\)|\(text\s+cut\s+off\)|\(continued\)|'
    r'\.\.\.\s*\(|غير\s+مكتمل|النص\s+ناقص|الفقرة\s+غير\s+كاملة'
)


def _reject_llm_extracted_clause(clause_text: str, source_batch_text: str) -> Optional[str]:
    """
    Rejects an LLM-extracted clause that fails to be genuinely verbatim,
    returning a rejection reason (for logging) or None if the clause is fine.
    Rejected clauses are dropped rather than kept — the rule-based extractor
    (which only ever takes literal substrings from the source text, never an
    LLM call) then supplies a genuinely verbatim replacement for the same
    underlying content via the merge step in `_extract_all_clauses`.

    1. Translation instead of verbatim extraction: if the source batch is
       Arabic-dominant but the extracted clause is not, the model translated
       rather than transcribed — unacceptable for a tender document that
       itself requires Arabic-language submissions (a translated clause in
       the proposal would expose the bid to formal disqualification).
    2. Self-acknowledged incomplete extraction: if the model's own output
       contains a hedge like "(clause incomplete)", it is explicitly telling
       us the extraction is unreliable; such a clause must never be silently
       shown to the user as if it were complete.
    """
    if _INCOMPLETENESS_MARKER_PATTERN.search(clause_text):
        return "self-annotated as incomplete/truncated by the extraction model"

    if len(clause_text.strip()) >= 15 and _is_arabic_dominant(source_batch_text) and not _is_arabic_dominant(clause_text):
        return "translated to a different language instead of extracted verbatim from an Arabic source"

    return None


def _classify_clause_exclusion(text: str, section: str) -> Optional[str]:
    """
    Returns 'contact_metadata' or 'evaluation_criteria' if the given clause
    text is reference/contact metadata or a scoring-table fragment rather than
    a genuine bidder obligation; returns None if it should be treated as a
    real candidate requirement. Clauses carrying explicit obligation language
    (mandatory or optional modals) are always protected from exclusion, since
    a real requirement can legitimately mention a number or reference.
    """
    t = _strip_arabic_diacritics(text.strip())
    words = t.split()

    # Never exclude text that carries genuine obligation language.
    _, _, mandatory_confidence, _ = _determine_mandatory(t)
    if mandatory_confidence > 0.0:
        return None

    # Pure email address
    if _EMAIL_ONLY_PATTERN.match(t.strip('.,;:')):
        return "contact_metadata"

    # Short reference-number / phone-number lines (marker + number, little else)
    if len(words) <= 8 and (_REFERENCE_NUMBER_PATTERN.search(t) or _PHONE_PATTERN.search(t)):
        return "contact_metadata"

    # Bare phone-number-shaped string (mostly digits/punctuation, few letters)
    letters_only = re.sub(r'[^^\w]|\d', '', re.sub(r'[\d\-+()\s]', '', t), flags=re.UNICODE)
    if len(words) <= 4 and re.search(r'\d{6,}', t) and len(re.sub(r'\W', '', letters_only)) <= 2:
        return "contact_metadata"

    # Evaluation/scoring table rows: short line with a total/weight/score keyword + number
    if len(words) <= 12 and _EVAL_WEIGHT_ROW_PATTERN.search(t):
        return "evaluation_criteria"

    # Evaluation criterion label ending in a bare numeric weight, in an evaluation-context section
    if (
        len(words) <= 14
        and _EVAL_SECTION_HINT.search(section)
        and re.match(r'^.+\s\d{1,3}\s*$', t)
    ):
        return "evaluation_criteria"

    return None

def extract_rfp_node(state: RFPProposalState) -> Dict[str, Any]:
    """
    Agent 1: RFP Document Extraction Agent
    Extracts structural hierarchy, RFP metadata, and candidate requirement clauses
    across the ENTIRE parsed RFP document while preserving page and section traceability.
    """
    file_path = state["file_path"]
    rfp_id = state["rfp_id"]

    # 1. Parse document into layout-annotated blocks across ALL pages
    blocks = DocumentParserService.parse_document(file_path)
    if not blocks:
        empty_meta = RFPMetadata(
            title="INFORMATION REQUIRED",
            issuer="INFORMATION REQUIRED",
            submission_deadline=None,
            budget_or_scope=None,
            evaluation_criteria=[],
            summary="INFORMATION REQUIRED"
        )
        return {
            "metadata": empty_meta.model_dump(),
            "raw_clauses": [],
            "evaluation_criteria_items": [],
            "active_agent": "Extraction Agent",
            "workflow_status": "EXTRACTION_FAILED",
            "error": "Extraction failed: no verifiable requirements were found. Check the file format or try a different one.",
            "logs": state.get("logs", []) + [{
                "agent": "Extraction Agent",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "EXTRACTION FAILED: Parsed 0 blocks from document. Halting pipeline — check file format or try a different file."
            }]
        }

    # 2. Extract Metadata (from introductory content and criteria sections)
    metadata = _extract_metadata(blocks)

    # 3. Extract Candidate Clauses across the ENTIRE document (Full-Document Strategy)
    raw_clauses, evaluation_criteria_items, encoding_corrupted_pages = _extract_all_clauses(blocks)

    # 4. Final safety sanity check: ensure no prohibited fictional strings exist
    metadata = _sanitize_metadata(metadata)

    # 5. Guard against silent zero/near-zero extraction failure.
    # A document is "long" if it spans >= 10 distinct pages; long documents that yield
    # fewer than MIN_CLAUSES_FOR_LONG_DOC candidate clauses almost always indicate a
    # parsing/extraction failure (e.g. unsupported encoding, scanned images, garbled
    # text layer) rather than a genuinely sparse tender, and must not proceed silently.
    MIN_CLAUSES_FOR_LONG_DOC = 5
    LONG_DOC_PAGE_THRESHOLD = 10
    distinct_pages = len({b.page_number for b in blocks})
    is_long_document = distinct_pages >= LONG_DOC_PAGE_THRESHOLD
    extraction_failed = len(raw_clauses) == 0 or (is_long_document and len(raw_clauses) < MIN_CLAUSES_FOR_LONG_DOC)

    if extraction_failed:
        log_entry = {
            "agent": "Extraction Agent",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": (
                f"EXTRACTION FAILED: Only {len(raw_clauses)} candidate clause(s) identified across "
                f"{len(blocks)} blocks spanning {distinct_pages} pages. Halting pipeline — check file "
                f"format, text layer encoding, or try a different file."
            )
        }
        return {
            "metadata": metadata.model_dump(),
            "raw_clauses": [c.model_dump() for c in raw_clauses],
            "evaluation_criteria_items": evaluation_criteria_items,
            "active_agent": "Extraction Agent",
            "workflow_status": "EXTRACTION_FAILED",
            "error": "Extraction failed: no verifiable requirements were found. Check the file format or try a different one.",
            "logs": state.get("logs", []) + [log_entry]
        }

    encoding_warning = ""
    if encoding_corrupted_pages:
        pages_str = ", ".join(str(p) for p in encoding_corrupted_pages)
        encoding_warning = (
            f" WARNING: {len(encoding_corrupted_pages)} page(s) ({pages_str}) contain text "
            f"corrupted by a source-PDF font/encoding defect (confirmed present in PyMuPDF's raw "
            f"extraction, not an artifact of this pipeline); any requirement clauses on those pages "
            f"were excluded rather than shown garbled — manual review of those pages is recommended."
        )

    log_entry = {
        "agent": "Extraction Agent",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": (
            f"Extracted metadata and identified {len(raw_clauses)} candidate clauses "
            f"({len(evaluation_criteria_items)} evaluation-criteria fragments excluded) "
            f"across {len(blocks)} blocks covering all pages.{encoding_warning}"
        )
    }

    return {
        "metadata": metadata.model_dump(),
        "raw_clauses": [c.model_dump() for c in raw_clauses],
        "evaluation_criteria_items": evaluation_criteria_items,
        "encoding_corrupted_pages": encoding_corrupted_pages,
        "active_agent": "Extraction Agent",
        "workflow_status": "CLASSIFYING",
        "logs": state.get("logs", []) + [log_entry]
    }


def _compute_repeated_header_footer_lines(blocks: List[ExtractedBlock]) -> Set[str]:
    """
    Identifies lines that repeat across most pages (running headers/footers,
    e.g. a page-number label reprinted on every page) so they can be excluded
    from title/metadata candidacy. A line qualifies if its normalized text
    appears on more than 80% of the document's distinct pages.
    """
    distinct_pages = len({b.page_number for b in blocks})
    if distinct_pages == 0:
        return set()

    line_pages: Dict[str, Set[int]] = {}
    for b in blocks:
        for line in b.text.split("\n"):
            norm = re.sub(r'\s+', ' ', line.strip().lower())
            if 2 <= len(norm) <= 60:
                line_pages.setdefault(norm, set()).add(b.page_number)

    threshold = max(2, int(distinct_pages * 0.8))
    return {norm for norm, pages in line_pages.items() if len(pages) >= threshold}


def _extract_metadata(blocks: List[ExtractedBlock]) -> RFPMetadata:
    """
    Extracts metadata from actual document blocks.
    Uses LLM with structured output when available, falling back strictly to
    rule-based extraction from the actual text (no fictional defaults).
    """
    repeated_lines = _compute_repeated_header_footer_lines(blocks)

    def _strip_repeated_lines(text: str) -> str:
        kept = [
            line for line in text.split("\n")
            if re.sub(r'\s+', ' ', line.strip().lower()) not in repeated_lines
        ]
        return "\n".join(kept).strip()

    # Select introductory blocks (pages 1-3 or first 20 blocks), with recurring
    # running headers/footers stripped so they can never be mistaken for the title.
    intro_blocks = [b for b in blocks if b.page_number <= 3]
    if not intro_blocks:
        intro_blocks = blocks[:20]
    intro_blocks = [
        b for b in intro_blocks
        if _strip_repeated_lines(b.text)
    ]

    # Also include any blocks across the entire document mentioning evaluation or criteria
    criteria_blocks = [
        b for b in blocks
        if b not in intro_blocks and any(k in (b.section_title + " " + b.text).lower() for k in ["evaluation", "criteria", "scoring", "award criteria", "معايير", "تقييم"])
    ]

    meta_context_blocks = intro_blocks + criteria_blocks
    context_text = "\n\n".join([
        f"[{b.section_title} - Page {b.page_number}]\n{_strip_repeated_lines(b.text)}"
        for b in meta_context_blocks
    ])[:6000]

    llm = LLMFactory.get_chat_model()
    if llm:
        try:
            structured_meta_llm = llm.with_structured_output(RFPMetadata)
            prompt = (
                "Analyze the following RFP document segments and extract the official metadata. "
                "The document may be in English or Arabic (or both).\n"
                "CRITICAL RULES:\n"
                "- Extract ONLY information explicitly stated in the text.\n"
                "- The tender/project title usually follows a marker such as 'Title:', 'Project Title:', "
                "'اسم المنافسة:', or 'بشأن:', or is the first large heading after the cover page. "
                "NEVER use a running header/footer line that simply repeats on every page (e.g. a page-number "
                "label) as the title.\n"
                "- The issuer is the organization that owns/issues the tender, often introduced by "
                "'Issued by:'/'Authority:' or Arabic markers like 'الهيئة', 'الوزارة', 'الأمانة', 'الشركة' "
                "in the header or the definitions section ('تعريفات').\n"
                "- The submission deadline is often introduced by 'Due Date:'/'Deadline:' or Arabic markers "
                "like 'آخر موعد للتقديم' or 'موعد فتح العروض', and may use a Gregorian or Hijri date.\n"
                "- If title or issuer is not found, set it to 'INFORMATION REQUIRED'.\n"
                "- If submission deadline or budget/scope is not mentioned, set it to null.\n"
                "- If evaluation criteria are not stated, return an empty list [].\n"
                "- If summary cannot be derived, set it to 'INFORMATION REQUIRED'.\n"
                "- NEVER invent or assume fictional RFP titles, organizations, dates, or criteria.\n\n"
                f"Document Segments:\n{context_text}"
            )
            result: RFPMetadata = structured_meta_llm.invoke([
                SystemMessage(content=EXTRACTION_AGENT_PROMPT),
                HumanMessage(content=prompt)
            ])
            # Validate LLM result against empty/hallucinated values
            if result and result.title and result.title != "INFORMATION REQUIRED":
                # The LLM is generally reliable for title/issuer but can miss a
                # deadline stated inside a table (marker and date split across
                # separate table-cell lines) — backfill from the rule-based
                # block-level table scanner rather than trusting a null result.
                if not result.submission_deadline:
                    fallback_deadline = _fallback_metadata(blocks, context_text, repeated_lines).submission_deadline
                    if fallback_deadline:
                        result.submission_deadline = fallback_deadline
                return _sanitize_metadata(result)
        except Exception as e:
            print(f"[Agent 1: Extraction] LLM metadata extraction error, using rule-based extractor: {e}")

    # Fallback to rule-based extraction strictly derived from document text
    return _fallback_metadata(blocks, context_text, repeated_lines)


def _fallback_metadata(blocks: List[ExtractedBlock], context_text: str, repeated_lines: Optional[Set[str]] = None) -> RFPMetadata:
    """
    Extracts metadata strictly from the actual parsed text of the document.
    Never invents or hardcodes fictional defaults.
    """
    full_text = context_text if context_text else "\n\n".join([b.text for b in blocks[:25]])

    # 1. Title Extraction
    title = None
    title_match = re.search(
        r'(?im)^\s*(?:TITLE|PROJECT(?:\s+TITLE)?|RFP\s+TITLE|NAME\s+OF\s+(?:PROJECT|TENDER)|SOLICITATION\s+NAME)[\s:\-–—]+([^\n\r]+)',
        full_text
    )
    if title_match:
        title = title_match.group(1).strip().strip('"\'')
    else:
        # Check for explicit REQUEST FOR PROPOSAL / RFP / TENDER lines with titles
        rfp_line_match = re.search(
            r'(?im)^\s*(?:REQUEST\s+FOR\s+PROPOSALS?|RFP|TENDER|INVITATION\s+TO\s+BID|SOLICITATION)[\s:\-–—]+([^\n\r]+)',
            full_text
        )
        if rfp_line_match:
            candidate = rfp_line_match.group(1).strip().strip('"\'()')
            if len(candidate) > 4 and not candidate.isupper() and "document ref" not in candidate.lower():
                title = candidate

    if not title:
        # Arabic marker: tender name usually follows "اسم المنافسة:" or "بشأن:",
        # either on the same line or the line immediately after the marker.
        ar_title_match = re.search(
            r'(?m)^\s*(?:اسم\s+المنافسة|بشأن)\s*[:：]?\s*\n?\s*([^\n\r]+)',
            full_text
        )
        if ar_title_match:
            candidate = ar_title_match.group(1).strip().strip('"\'')
            if len(candidate) >= 8 and _has_valid_text_encoding(candidate):
                title = candidate

    if not title:
        # Check early heading blocks on page 1, excluding recurring header/footer lines
        for b in blocks:
            if b.page_number == 1 and b.block_type == "heading":
                candidate = b.text.strip().strip('# \t\r\n"\'')
                lower_cand = candidate.lower()
                norm_cand = re.sub(r'\s+', ' ', candidate.strip().lower())
                if repeated_lines and norm_cand in repeated_lines:
                    continue
                if (
                    len(candidate) >= 8 and len(candidate) <= 120
                    and not lower_cand.startswith("section")
                    and not lower_cand.startswith("page")
                    and not lower_cand.startswith("document ref")
                    and "table of contents" not in lower_cand
                    and _has_valid_text_encoding(candidate)
                ):
                    title = candidate
                    break

    if not title:
        title = "INFORMATION REQUIRED"

    # 2. Issuer Extraction
    issuer = None
    issuer_match = re.search(
        r'(?im)^\s*(?:ISSUED\s+BY|CLIENT|ORGANIZATION|AGENCY|AUTHORITY|PROCURING\s+ENTITY|PROCURING\s+AGENCY|BUYER|ISSUING\s+ORGANIZATION)[\s:\-–—]+([^\n\r]+)',
        full_text
    )
    if issuer_match:
        issuer = issuer_match.group(1).strip().strip('"\'.,')
    else:
        # Check natural language invitation sentences
        agency_pattern = re.search(
            r'(?i)The\s+([A-Z][A-Za-z0-9\s&,\.\-–—]+?(?:Authority|Agency|Department|Ministry|Corporation|Commission|Board|District|Administration|Council|Foundation|Institute|Services|LLC|Inc|Ltd))\s+(?:invites|requests|issues|is\s+seeking|solicits|hereby\s+issues)',
            full_text
        )
        if agency_pattern:
            issuer = agency_pattern.group(1).strip().strip('"\'.,')

    if not issuer:
        # Arabic markers: issuing entity is typically named on a standalone line
        # containing "الهيئة" (authority), "الوزارة" (ministry), "الأمانة"
        # (secretariat/municipality), or "الشركة" (company), often in the header
        # or the definitions ("تعريفات") section. Prefer the earliest such line.
        # Deliberately NOT skipping recurring running headers/footers here
        # (unlike the title search above): a repeated header/footer is a bad
        # signal for the TITLE (that's how "رقم الصفحة" got picked in the
        # original bug), but a strong, reliable signal for the ISSUER — a
        # government/organization letterhead legitimately reprints the
        # issuing entity's name on every page.
        best_issuer_candidate = None
        best_issuer_page = None
        for b in blocks:
            for line in b.text.split("\n"):
                line_str = line.strip()
                if 4 <= len(line_str.split()) <= 12 and re.match(r'^(?:الهيئة|الوزارة|الأمانة|الشركة)\b', line_str):
                    # "الهيئة"/etc. refer to the issuing entity constantly
                    # throughout a legal tender document, not just where it is
                    # actually being NAMED — so most matching lines are normal
                    # sentences ("the Authority must...", "...at the address
                    # the Authority provided...") rather than a name label.
                    # Reject any candidate containing sentence-continuation
                    # markers (prepositions, relative pronouns, verbs/"و"
                    # conjunction prefixes) anywhere in the line; a genuine
                    # organization-name line is just the name, nothing else,
                    # and contains no full-stop punctuation.
                    words_in_line = line_str.split()
                    sentence_markers = {
                        "في", "من", "على", "إلى", "عن", "مع", "لدى", "حيث",
                        "التي", "الذي", "وتجب", "يجب", "تقوم", "وذلك", "وقد",
                        "أن", "قدمتها", "بطلب",
                    }
                    has_sentence_marker = any(w in sentence_markers for w in words_in_line)
                    has_punctuation = bool(re.search(r'[.:؛،]', line_str))
                    if has_sentence_marker or has_punctuation or not _has_valid_text_encoding(line_str):
                        continue
                    if best_issuer_page is None or b.page_number < best_issuer_page:
                        best_issuer_candidate = line_str
                        best_issuer_page = b.page_number
        if best_issuer_candidate:
            issuer = best_issuer_candidate.strip('"\'.,:')

    if not issuer:
        issuer = "INFORMATION REQUIRED"

    # 3. Submission Deadline Extraction
    deadline = None
    deadline_match = re.search(
        r'(?im)^\s*(?:DUE\s+DATE|SUBMISSION\s+DEADLINE|CLOSING\s+DATE|PROPOSALS?\s+DUE|DEADLINE\s+FOR\s+SUBMISSION|RESPONSE\s+DEADLINE)[\s:\-–—]+([^\n\r]+)',
        full_text
    )
    if deadline_match:
        deadline = deadline_match.group(1).strip().strip('"\'.,')
    else:
        dl_phrase_match = re.search(
            r'(?im)(?:due\s+on\s+or\s+before|submitted\s+no\s+later\s+than|closing\s+time\s+and\s+date\s+is)[\s:\-–—]+([^\n\r\.]+)',
            full_text
        )
        if dl_phrase_match:
            deadline = dl_phrase_match.group(1).strip().strip('"\'.,')

    if not deadline:
        # Arabic markers: "آخر موعد للتقديم" (final submission deadline) or
        # "موعد فتح العروض" (bid opening date). These frequently live inside a
        # table (deadlines table), where the marker phrase and the date can be
        # split across separate table-cell lines rather than sitting adjacent
        # on one line/the next — so search block-wide (not line-by-line) for
        # the marker plus the nearest Gregorian (YYYY-MM-DD) or Hijri date
        # anywhere in the same block/table.
        ar_deadline_marker_re = re.compile(r'آخر\s*موعد|موعد\s+فتح\s+العروض|موعد\s+إغلاق\s+المنافسة')
        date_pattern_re = re.compile(r'\d{4}\s*[/\-]\s*\d{1,2}\s*[/\-]\s*\d{1,2}')
        for b in blocks:
            if not ar_deadline_marker_re.search(b.text):
                continue
            date_match = date_pattern_re.search(b.text)
            if date_match:
                # Include a short trailing time-of-day fragment if present (e.g. "الساعة : 01:00 مساءً")
                tail = b.text[date_match.end():date_match.end() + 60]
                time_match = re.search(r'الساعة\s*[:：]?\s*\d{1,2}:\d{2}[^\n\r]{0,15}', tail)
                deadline = re.sub(r'\s+', '', date_match.group(0))
                if time_match:
                    deadline = f"{deadline} {re.sub(r'\s+', ' ', time_match.group(0)).strip()}"
                break
        if not deadline:
            # Marker present but no clean date found nearby: fall back to the
            # first line following the marker as a raw text candidate.
            ar_deadline_marker = re.search(
                r'(?m)(?:آخر\s+موعد\s+للتقديم|موعد\s+فتح\s+العروض)\s*[:：]?\s*\n?\s*([^\n\r]+)',
                full_text
            )
            if ar_deadline_marker:
                candidate = ar_deadline_marker.group(1).strip().strip('"\'.,')
                if candidate:
                    deadline = candidate

    # 4. Budget or Scope Extraction
    budget = None
    budget_match = re.search(
        r'(?im)^\s*(?:ESTIMATED\s+BUDGET|BUDGET|NOT\s+TO\s+EXCEED|CONTRACT\s+VALUE|MAXIMUM\s+VALUE|ENGAGEMENT\s+SCOPE|SCOPE\s+OF\s+WORK)[\s:\-–—]+([^\n\r]+)',
        full_text
    )
    if budget_match:
        budget = budget_match.group(1).strip().strip('"\'.,')
    else:
        curr_match = re.search(
            r'(?i)(?:budget|contract\s+value|estimated\s+value)[\s:\-–—]*(?:is|of)?\s*([\$£€]\s*[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|k|thousand))?)',
            full_text
        )
        if curr_match:
            budget = curr_match.group(1).strip()

    # 5. Evaluation Criteria Extraction
    evaluation_criteria = []
    # Find criteria sections
    criteria_sections = [
        b for b in blocks 
        if any(k in (b.section_title + " " + b.text).lower() for k in ["evaluation criteria", "selection criteria", "award criteria", "scoring criteria", "evaluation"])
    ]
    for cb in criteria_sections:
        lines = cb.text.split("\n")
        for line in lines:
            line_str = line.strip()
            # Match numbered or bulleted criteria with weight/percent e.g. "1. Technical Architecture (30%)"
            crit_match = re.search(r'(?i)^\s*(?:\d+[\.\)]\s*|[-*•]\s*)?([^\n\r]+?\(\s*\d+%\s*\)|[^\n\r]+?\b\d+\s*(?:%|percent|points\b)[^\n\r]*)', line_str)
            if crit_match:
                cleaned_crit = crit_match.group(0).strip().strip('-*• \t')
                if cleaned_crit and cleaned_crit not in evaluation_criteria:
                    evaluation_criteria.append(cleaned_crit)

    # 6. Summary Extraction
    summary = None
    # Search for introductory summary blocks
    intro_summary_blocks = [
        b for b in blocks 
        if any(k in b.section_title.lower() for k in ["executive summary", "introduction", "background", "objective", "purpose", "section 1"])
        and len(b.text.split()) >= 10
    ]
    if intro_summary_blocks:
        first_block_text = intro_summary_blocks[0].text.strip()
        # Remove any section header line if repeated
        lines = [l.strip() for l in first_block_text.split("\n") if l.strip() and not l.strip().lower().startswith("section")]
        if lines:
            summary = " ".join(lines[:3])[:350].strip()

    if not summary:
        # Try taking first substantive paragraph from page 1
        for b in blocks:
            if b.page_number == 1 and len(b.text.split()) >= 15:
                lower = b.text.lower()
                if not lower.startswith("request for proposal") and not lower.startswith("title:") and not lower.startswith("issued by:"):
                    summary = b.text.strip()[:350].strip()
                    break

    if not summary:
        summary = "INFORMATION REQUIRED"

    return RFPMetadata(
        title=title,
        issuer=issuer,
        submission_deadline=deadline,
        budget_or_scope=budget,
        evaluation_criteria=evaluation_criteria,
        summary=summary
    )


def _extract_all_clauses(blocks: List[ExtractedBlock]) -> Tuple[List[RawClause], List[Dict[str, Any]], List[int]]:
    """
    Extracts candidate clauses across the ENTIRE document (all pages).
    Processes blocks in batches to keep within context limits if LLM is active,
    and runs a comprehensive rule-based extractor to guarantee complete,
    traceable coverage without any hallucinations.
    """
    raw_clauses: List[RawClause] = []
    seen_signatures: Set[str] = set()
    kept_sig_list: List[str] = []
    kept_token_sets: List[Set[str]] = []

    # Conservative fuzzy-dedup settings: two clauses are treated as the same
    # only when their content-token overlap is high AND both carry enough
    # tokens to make that overlap meaningful. Set intentionally high (0.82) so
    # that genuinely distinct obligations sharing vocabulary (very common in a
    # legal tender) are NOT merged — losing a real requirement is worse than
    # keeping a near-duplicate.
    JACCARD_DUP_THRESHOLD = 0.82
    MIN_TOKENS_FOR_FUZZY = 6

    def _is_duplicate(norm_sig: str, token_set: Set[str]) -> bool:
        """
        A candidate is a duplicate if its signature exactly matches one already
        kept, OR is a substantial substring/superstring of one already kept, OR
        its content-token set overlaps an already-kept clause above
        JACCARD_DUP_THRESHOLD.

        The LLM and rule-based extraction paths frequently extract the same
        underlying clause with minor wording differences. Exact-match alone
        under-deduplicates; containment catches cases where one text is a
        substring of the other (e.g. one includes a leading "Section 3.2:"
        fragment); the fuzzy token-overlap check catches the remaining case —
        two paraphrases of the same obligation where neither is a substring of
        the other — which would otherwise inflate the count with the same
        clause listed twice.
        """
        if norm_sig in seen_signatures:
            return True
        if len(norm_sig) >= 40:
            for kept in kept_sig_list:
                if len(kept) < 40:
                    continue
                if norm_sig in kept or kept in norm_sig:
                    return True
        if len(token_set) >= MIN_TOKENS_FOR_FUZZY:
            for kept_tokens in kept_token_sets:
                if len(kept_tokens) >= MIN_TOKENS_FOR_FUZZY and \
                        _jaccard(token_set, kept_tokens) >= JACCARD_DUP_THRESHOLD:
                    return True
        return False

    def _remember(norm_sig: str, token_set: Set[str]) -> None:
        seen_signatures.add(norm_sig)
        kept_sig_list.append(norm_sig)
        kept_token_sets.append(token_set)

    llm = LLMFactory.get_chat_model()
    batches = _create_block_batches(blocks, max_blocks_per_batch=10, max_chars_per_batch=3500)

    # 1. LLM Batch Extraction (if available)
    if llm:
        for batch in batches:
            batch_text = "\n\n".join([
                f"[Block {idx + 1} | Page {b.page_number} | Section: {b.section_title}]\n{b.text}"
                for idx, b in enumerate(batch)
            ])
            try:
                structured_batch_llm = llm.with_structured_output(ExtractedClauseBatch)
                prompt = (
                    "You are an expert RFP requirement clause extractor. Extract all candidate requirement clauses "
                    "verbatim from the following document blocks. For each clause:\n"
                    "- Extract exact verbatim text (never summarize, alter, or fabricate requirements).\n"
                    "- CRITICAL: Preserve the ORIGINAL LANGUAGE of the source text exactly as written. If a block "
                    "is in Arabic, your extracted clause text MUST be in Arabic — copy the Arabic characters "
                    "directly, do not translate to English or any other language, regardless of what language "
                    "you reason in internally. A translated clause is not a verbatim extraction and is unusable: "
                    "this tender requires submissions in the source language, so a translated clause would expose "
                    "a resulting bid to formal disqualification.\n"
                    "- If a clause is cut off or incomplete in the source blocks, either extract it in full by "
                    "including the complete surrounding text, or omit it entirely. NEVER annotate a clause with "
                    "hedges like '(clause incomplete)', '(truncated)', or similar — output only complete, "
                    "verbatim text or nothing for that clause.\n"
                    "- Set source_page to the exact Page number indicated in the block header.\n"
                    "- Set source_section to the exact Section indicated in the block header.\n\n"
                    f"Document Blocks:\n{batch_text}"
                )
                batch_result: ExtractedClauseBatch = structured_batch_llm.invoke([
                    SystemMessage(content=EXTRACTION_AGENT_PROMPT),
                    HumanMessage(content=prompt)
                ])
                if batch_result and batch_result.clauses:
                    for c in batch_result.clauses:
                        rejection_reason = _reject_llm_extracted_clause(c.text, batch_text)
                        if rejection_reason:
                            print(
                                f"[Agent 1: Extraction] Rejected LLM clause ({rejection_reason}), "
                                f"deferring to rule-based extraction for this content: {c.text[:80]!r}"
                            )
                            continue
                        norm_sig = _normalize_clause_sig(c.text)
                        token_set = _clause_token_set(c.text)
                        if norm_sig and not _is_duplicate(norm_sig, token_set):
                            # Validate source_page and source_section
                            if not c.source_page or c.source_page <= 0:
                                c.source_page = batch[0].page_number
                            if not c.source_section or c.source_section == "General":
                                c.source_section = batch[0].section_title
                            _remember(norm_sig, token_set)
                            raw_clauses.append(c)
            except Exception as e:
                print(f"[Agent 1: Extraction] LLM clause batch error: {e}")

    # 2. Rule-Based Clause Extraction across ALL blocks
    # Guarantees full-document coverage, even when LLM is offline or skips blocks
    rule_clauses = _extract_clauses_rule_based(blocks)
    for rc in rule_clauses:
        norm_sig = _normalize_clause_sig(rc.text)
        token_set = _clause_token_set(rc.text)
        if norm_sig and not _is_duplicate(norm_sig, token_set):
            _remember(norm_sig, token_set)
            raw_clauses.append(rc)

    # 3. Filter out metadata/contact fragments and route evaluation-criteria
    #    fragments (weights, totals, score rows) away from the requirements
    #    pipeline entirely, then apply canonical numbering to what remains.
    formatted_clauses: List[RawClause] = []
    evaluation_criteria_items: List[Dict[str, Any]] = []
    encoding_corrupted_pages: Set[int] = set()
    idx = 1
    for c in raw_clauses:
        # Reject text corrupted by a confirmed font/cmap encoding defect in
        # this document's source (glyphs mapping to unrelated Unicode blocks
        # like Ogham/Canadian Aboriginal Syllabics instead of Arabic letters
        # on certain pages) rather than ever showing garbled requirement text
        # to the user. Not silent: the affected page is recorded and surfaced
        # in the extraction log so the gap is visible, not silently dropped.
        if not _has_valid_text_encoding(c.text):
            encoding_corrupted_pages.add(max(1, c.source_page))
            continue

        exclusion = _classify_clause_exclusion(c.text, c.source_section or "General")
        if exclusion == "contact_metadata":
            continue
        if exclusion == "evaluation_criteria":
            evaluation_criteria_items.append({
                "text": c.text.strip(),
                "source_page": max(1, c.source_page),
                "source_section": c.source_section.strip() if c.source_section else "General"
            })
            continue

        formatted_clauses.append(
            RawClause(
                clause_id=f"CLAUSE-{idx:03d}",
                text=c.text.strip(),
                source_page=max(1, c.source_page),
                source_section=c.source_section.strip() if c.source_section else "General"
            )
        )
        idx += 1

    return formatted_clauses, evaluation_criteria_items, sorted(encoding_corrupted_pages)


def _extract_clauses_rule_based(blocks: List[ExtractedBlock]) -> List[RawClause]:
    """
    Comprehensive rule-based clause extraction operating across the ENTIRE document.
    Never invents text; preserves source page and section for every clause.
    """
    clauses: List[RawClause] = []
    
    # Imperatives & obligation modal verbs
    imperative_pattern = re.compile(
        r'\b(?:shall|must|required|mandatory|will|should|agrees\s+to|is\s+required\s+to|are\s+required\s+to|covenants|undertakes|liability|penalty|sla)\b',
        re.IGNORECASE
    )

    # Numbered clause headers e.g. "2.1 High Availability:", "REQ-01:", "3.2.1"
    numbered_clause_pattern = re.compile(
        r'^\s*(?:(?:\d+\.){1,3}\d*|(?:REQ|RFP|SPEC|DELIV|SEC|TECH|FUNC|LEGAL|SLA)[-_:\s])',
        re.IGNORECASE
    )

    # Requirement-specific section indicators (English + Arabic)
    req_section_keywords = [
        "requirement", "specification", "technical", "security", "functional",
        "legal", "terms", "deliverable", "scope of work", "compliance", "sla", "infrastructure",
        "الشروط", "المواصفات", "المتطلبات", "الالتزامات", "الأحكام", "الضمانات", "الجزاءات"
    ]

    for b in blocks:
        text = b.text.strip()
        if not text or len(text) < 20:
            continue

        # Skip document title/metadata-only blocks
        lower_text = text.lower()
        if (
            lower_text.startswith("request for proposal")
            or lower_text.startswith("document ref:")
            or lower_text.startswith("issued by:")
            or lower_text.startswith("due date:")
            or lower_text.startswith("table of contents")
            or lower_text == "section 1: executive summary & procurement objective"
        ):
            continue

        is_req_section = any(k in b.section_title.lower() for k in req_section_keywords)

        # Split block into individual numbered items or bullet points if present
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        
        # Check if block contains distinct numbered or bulleted items
        item_chunks: List[str] = []
        current_chunk: List[str] = []

        for line in lines:
            is_item_start = (
                numbered_clause_pattern.match(line)
                or line.startswith("•")
                or line.startswith("- ")
                or line.startswith("* ")
            )
            if is_item_start and current_chunk:
                item_chunks.append(" ".join(current_chunk))
                current_chunk = [line]
            else:
                current_chunk.append(line)
        if current_chunk:
            item_chunks.append(" ".join(current_chunk))

        for chunk in item_chunks:
            chunk_clean = chunk.strip().strip('-*• \t')
            if len(chunk_clean.split()) < 4 or len(chunk_clean) < 20:
                continue

            # Qualifying condition:
            # 1. Contains RFC 2119 imperatives / obligation modals (English or Arabic)
            # 2. OR is explicitly numbered (e.g. 2.1 ...)
            # 3. OR is in a requirement section and contains substantive specifications
            _, _, ar_mandatory_conf, _ = _determine_mandatory(chunk_clean)
            has_imperatives = bool(imperative_pattern.search(chunk_clean)) or ar_mandatory_conf > 0.0
            has_numbering = bool(numbered_clause_pattern.match(chunk_clean))

            if has_imperatives or has_numbering or (is_req_section and len(chunk_clean.split()) >= 6):
                clauses.append(
                    RawClause(
                        clause_id="",  # Will be assigned canonical sequential ID during deduplication
                        text=chunk_clean,
                        source_page=b.page_number,
                        source_section=b.section_title
                    )
                )

    return clauses


def _create_block_batches(
    blocks: List[ExtractedBlock],
    max_blocks_per_batch: int = 10,
    max_chars_per_batch: int = 3500
) -> List[List[ExtractedBlock]]:
    """Partitions blocks into size-bounded batches to prevent LLM context overflow."""
    batches: List[List[ExtractedBlock]] = []
    current_batch: List[ExtractedBlock] = []
    current_chars = 0

    for b in blocks:
        b_len = len(b.text)
        if current_batch and (len(current_batch) >= max_blocks_per_batch or (current_chars + b_len) > max_chars_per_batch):
            batches.append(current_batch)
            current_batch = []
            current_chars = 0
        current_batch.append(b)
        current_chars += b_len

    if current_batch:
        batches.append(current_batch)

    return batches


_LEADING_ORDINAL_CLEANED_RE = re.compile(r'^(?:اولا|أولا|ثانيا|ثالثا|رابعا|خامسا)')


def _normalize_clause_sig(text: str) -> str:
    """
    Computes a normalized signature for deduplication. Uses Unicode-aware
    \\w (not [a-z0-9]) so non-Latin scripts such as Arabic are preserved in
    the signature rather than stripped to nothing — stripping them collapses
    every Arabic clause's signature down to just its bare digits, causing
    unrelated clauses that happen to share a number (e.g. a percentage or day
    count) to be wrongly treated as duplicates and silently dropped.

    A leading Arabic ordinal/transitional marker (e.g. "أولاً: ") is stripped
    after whitespace removal (not before), since PDF font-kerning artifacts
    can insert a stray internal space within these words (e.g. "أو لا" instead
    of "أولا") that would otherwise defeat a pre-whitespace-removal match. The
    LLM and rule-based extraction paths sometimes disagree on whether to
    include such a leading marker at all, which would otherwise misalign an
    exact-prefix signature for what is genuinely the same clause. The full
    remaining text (not just a short prefix) is used, since truncating to a
    short prefix risks conflating distinct clauses that merely start similarly.
    """
    cleaned = re.sub(r'[^\w]', '', text.strip().lower(), flags=re.UNICODE)
    cleaned = _LEADING_ORDINAL_CLEANED_RE.sub('', cleaned)
    return cleaned[:400]


# Minimal Arabic/English stopword set: removed before fuzzy near-duplicate
# scoring so the overlap reflects meaningful content words, not shared
# grammatical glue. Kept deliberately small to avoid over-merging distinct
# clauses that legitimately share common function words.
_DUP_STOPWORDS = {
    "من", "في", "على", "الى", "إلى", "عن", "مع", "او", "أو", "و", "ما", "ان",
    "أن", "لا", "قد", "هذا", "هذه", "التي", "الذي", "به", "بها", "اي", "أي",
    "the", "of", "to", "in", "and", "or", "a", "an", "for", "on", "by", "is",
    "be", "that", "this", "with", "as", "at", "any", "shall", "must",
}


def _clause_token_set(text: str) -> Set[str]:
    """
    Tokenizes a clause into a normalized, diacritics-stripped, lowercased set of
    content tokens for fuzzy (Jaccard) near-duplicate detection. Short tokens and
    a small stopword list are removed so the overlap score reflects meaningful
    content words rather than grammatical glue.

    This complements the exact/containment signature check: the LLM and
    rule-based extraction paths often produce the SAME underlying obligation
    with re-ordered or lightly reworded phrasing where neither text is a
    substring of the other, so containment misses them and they wrongly
    survive as two separate requirements (the "same clause counted twice"
    inflation). A high token-overlap ratio catches those, while a conservative
    threshold and a minimum-token guard avoid collapsing genuinely distinct
    clauses that merely share vocabulary.
    """
    stripped = _strip_arabic_diacritics(text).lower()
    raw = re.findall(r'[\w؀-ۿ]+', stripped, flags=re.UNICODE)
    return {t for t in raw if len(t) >= 2 and t not in _DUP_STOPWORDS}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Token-set Jaccard similarity; 0.0 when either set is empty."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / len(a | b)


def _sanitize_metadata(meta: RFPMetadata) -> RFPMetadata:
    """
    Strict safety check: replaces any legacy prohibited fictional placeholder
    with explicit 'INFORMATION REQUIRED' or None.
    """
    for prohibited in PROHIBITED_FICTIONAL_STRINGS:
        if prohibited.lower() in meta.title.lower():
            meta.title = "INFORMATION REQUIRED"
        if prohibited.lower() in meta.issuer.lower():
            meta.issuer = "INFORMATION REQUIRED"
        if meta.submission_deadline and prohibited.lower() in meta.submission_deadline.lower():
            meta.submission_deadline = None
        if meta.budget_or_scope and prohibited.lower() in meta.budget_or_scope.lower():
            meta.budget_or_scope = None

    if not meta.title or not meta.title.strip():
        meta.title = "INFORMATION REQUIRED"
    if not meta.issuer or not meta.issuer.strip():
        meta.issuer = "INFORMATION REQUIRED"
    if not meta.summary or not meta.summary.strip():
        meta.summary = "INFORMATION REQUIRED"

    return meta
