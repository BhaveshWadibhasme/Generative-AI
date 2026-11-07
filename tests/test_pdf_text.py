"""PDF text extraction used by the Streamlit chatbot."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "google gemini LLM" / "llm_chat.py"


def load_app():
    spec = importlib.util.spec_from_file_location("llm_chat_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


APP = load_app()


def build_pdf(text):
    """Return a one-page PDF whose extracted text is ``text``."""
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 100 700 Td ({escaped}) Tj ET\n".encode("ascii")
    page = (
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
    )
    content = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"endstream"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        page,
        content,
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    parts = [b"%PDF-1.4\n"]
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(sum(len(part) for part in parts))
        parts.append(f"{number} 0 obj\n".encode("ascii") + obj + b"\nendobj\n")
    xref_position = sum(len(part) for part in parts)
    xref = [
        b"xref\n",
        f"0 {len(objects) + 1}\n".encode("ascii"),
        b"0000000000 65535 f \n",
    ]
    for offset in offsets[1:]:
        xref.append(f"{offset:010d} 00000 n \n".encode("ascii"))
    trailer = (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_position}\n%%EOF\n"
    ).encode("ascii")
    return b"".join(parts + xref + [trailer])


def test_extract_text_from_no_pdfs():
    assert APP.extract_text_from_pdfs([]) == ""


def test_extract_text_concatenates_pdfs_in_order(tmp_path):
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first.write_bytes(build_pdf("Alpha"))
    second.write_bytes(build_pdf("Beta"))
    extracted = APP.extract_text_from_pdfs([str(first), str(second)])
    assert extracted == "AlphaBeta"
