from markitdown import MarkItDown

class PDFConvertor:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def convert(self):
        markdown = MarkItDown()
        try:
            result = markdown.convert(self.file_path)
            return result.text_content
        finally:
            if hasattr(markdown, 'close'):
                markdown.close()