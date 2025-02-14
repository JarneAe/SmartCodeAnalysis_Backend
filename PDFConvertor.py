from markitdown import MarkItDown

class PDFConvertor:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def convert(self):
        markdown = MarkItDown()
        try:
            result = markdown.convert(self.file_path)
            return result.text_content  # Ensure this is the correct attribute
        finally:
            # If MarkItDown has a close or cleanup method, call it here
            if hasattr(markdown, 'close'):
                markdown.close()