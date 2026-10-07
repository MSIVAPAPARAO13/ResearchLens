import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = []
    for page in doc:
        full_text.append(page.get_text("text"))
    return "\n".join(full_text)
