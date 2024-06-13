# 別でpopplerをインストールしておく必要がある

from pdf2image import convert_from_path

def pdf_to_png(input_pdf_path, output_png_path):
    # PDFをPNGに変換
    pages = convert_from_path(input_pdf_path, 72)
    for page in pages:
        page.save(output_png_path, 'PNG')


    
input_pdf_path = "ebi.pdf"
output_png_path = "ebi2.png"
pdf_to_png(input_pdf_path, output_png_path)