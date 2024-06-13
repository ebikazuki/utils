import fitz
def pdf_to_png(input_pdf_path, output_png_path):
    # PDFをPNGに変換
    pdf_document = fitz.open(input_pdf_path)
    page = pdf_document.load_page(0)  # 最初のページを読み込む
    pix = page.get_pixmap(dpi=72) #pdfのデフォルトは72
    pix.save(output_png_path)
    
input_pdf_path = "ebi.pdf"
output_png_path = "ebi.png"
pdf_to_png(input_pdf_path, output_png_path)