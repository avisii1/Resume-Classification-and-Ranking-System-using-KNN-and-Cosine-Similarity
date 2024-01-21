from pdfminer.high_level import extract_text


def extract_text_from_pdf(pdf_path):
  try:
    text = extract_text(pdf_path)
    return text
  except Exception as e:
    return str(e)