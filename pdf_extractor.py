from typing import List, Dict, Optional, Tuple, Union
from io import BytesIO
import re

try:
    import PyPDF2
except Exception:
    PyPDF2 = None


def _read_pdf_bytes(pdf_bytes: bytes) -> str:
    text = ""
    if PyPDF2 is None:
        return text
    reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    for page in reader.pages:
        try:
            text += page.extract_text() or ""
        except Exception:
            continue
    return text


def extract_text_from_files(files: List[Tuple[str, bytes]]) -> List[Dict[str, Union[str, Dict[str, str]]]]:
    results = []
    for name, content in files:
        raw_text = _read_pdf_bytes(content)
        cleaned = _clean_text(raw_text)
        sections = split_sections(cleaned)
        results.append({"name": name, "text": cleaned, "sections": sections})
    return results


def split_sections(text: str) -> Dict[str, str]:
    title = ""
    abstract = ""
    body = text
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        title = lines[0]
    abstract_match = re.search(r"\babstract\b[\s:\-]*", text, flags=re.IGNORECASE)
    if abstract_match:
        start = abstract_match.end()
        remainder = text[start:]
        end_match = re.search(r"\n[a-z ]{3,15}\n", remainder, flags=re.IGNORECASE)
        end = end_match.start() if end_match else len(remainder)
        abstract = remainder[:end].strip()
        body = (text[:abstract_match.start()] + remainder[end:]).strip()
    return {"title": title, "abstract": abstract, "body": body}


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

