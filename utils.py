from PyPDF2 import PdfReader

def read_pdf(file):
    text = ""
    reader = PdfReader(file)
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

def read_txt(file):
    return file.read().decode('utf-8')
