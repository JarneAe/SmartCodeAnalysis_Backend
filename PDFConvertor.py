import os
from markitdown import MarkItDown

class PDFConvertor:
    def __init__(self, file_path: str, save_dir: str = None):
        self.file_path = file_path
        self.save_dir = save_dir

    def convert(self):
        markdown = MarkItDown()
        try:
            result = markdown.convert(self.file_path)
            markdown_text = result.text_content  # Ensure this is the correct attribute

            if self.save_dir:
                self._save_markdown(markdown_text)

            return markdown_text
        finally:
            # Close or cleanup if needed
            if hasattr(markdown, 'close'):
                markdown.close()

    def _save_markdown(self, content: str):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        file_name = os.path.splitext(os.path.basename(self.file_path))[0] + ".md"
        save_path = os.path.join(self.save_dir, file_name)

        with open(save_path, "w", encoding="utf-8") as md_file:
            md_file.write(content)
