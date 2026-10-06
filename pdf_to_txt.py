import sys
import os

try:
    import pdfplumber
except ImportError:
    print("Устанавливаю pdfplumber...")
    os.system(f"{sys.executable} -m pip install pdfplumber -q")
    import pdfplumber

def pdf_to_txt(pdf_path, txt_path=None):
    if txt_path is None:
        txt_path = os.path.splitext(pdf_path)[0] + '.txt'

    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text += f'\n\n--- Страница {i+1} ---\n\n'
                text += page_text

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f'Готово: {txt_path} ({len(text)} символов, {len(pdf.pages)} страниц)')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Использование: python pdf_to_txt.py "путь_к_файлу.pdf"')
        sys.exit(1)
    pdf_to_txt(sys.argv[1])
