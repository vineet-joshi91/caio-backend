# parser.py

import pdfplumber
import pandas as pd

def parse_file(uploaded_file):
    filename = uploaded_file.name
    ext = filename.split('.')[-1].lower()

    if ext == 'pdf':
        return parse_pdf(uploaded_file)
    elif ext in ['xlsx', 'xls']:
        return parse_excel(uploaded_file)
    elif ext == 'docx':
        return parse_docx(uploaded_file)
    elif ext == 'txt':
        return uploaded_file.read().decode('utf-8')
    else:
        return "[Unsupported file type]"

def parse_pdf(file_obj):
    try:
        with pdfplumber.open(file_obj) as pdf:
            text = "\n".join([page.extract_text() or '' for page in pdf.pages])
        return text.strip() or "[No text found in PDF]"
    except Exception as e:
        return f"PDF parsing error: {e}"

def parse_excel(file_obj):
    try:
        df = pd.read_excel(file_obj)
        return df.to_string(index=False)
    except Exception as e:
        return f"Excel parsing error: {e}"

def parse_docx(file_obj):
    try:
        from docx import Document
        document = Document(file_obj)
        return "\n".join([para.text for para in document.paragraphs if para.text.strip()])
    except Exception as e:
        return f"DOCX parsing error: {e}"
