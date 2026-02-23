
def load_text_file(filepath):
    with open(filepath,"r",encoding="utf-8")as f:
        return f.read()
    
def chunk_text(text,chunk_size = 300, overlap = 50):
    words = text.split()
    chunks = []
    
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += chunk_size - overlap

    return chunks  

from PyPDF2 import PdfReader

def load_pdf_file(filepath):
    reader = PdfReader(filepath)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    return text