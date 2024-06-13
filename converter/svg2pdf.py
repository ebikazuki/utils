from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF

# Path to the uploaded SVG file
input_svg_path = "ebi.svg"
output_pdf_path = "ebi.pdf"

# Convert SVG to PDF
drawing = svg2rlg(input_svg_path)
renderPDF.drawToFile(drawing, output_pdf_path)
