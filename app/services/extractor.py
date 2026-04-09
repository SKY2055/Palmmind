import io
import re
from typing import Optional
from PyPDF2 import PdfReader


class TextExtractor:
    @staticmethod
    def extract_from_pdf(file_content: bytes) -> str:
        """Extract text from PDF file with OCR fallback for image-based PDFs."""
        pdf_file = io.BytesIO(file_content)
        text = ""
        
        try:
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        except Exception as e:
            print(f"[TextExtractor] PyPDF2 extraction error: {e}")
            try:
                text = TextExtractor.extract_from_pdf_ocr(pdf_file)
            except Exception as ocr_error:
                print(f"[TextExtractor] OCR extraction failed: {ocr_error}")
                raise Exception(f"Unable to extract text from PDF. The file may be corrupted or password-protected. Error: {ocr_error}")
        
        if not text.strip() or len(text.strip()) < 50:
            try:
                text = TextExtractor.extract_from_pdf_ocr(pdf_file)
            except Exception as e:
                print(f"[TextExtractor] OCR extraction failed: {e}")
                if not text.strip():
                    raise Exception(f"Unable to extract text from PDF. The file may be corrupted, image-based, or password-protected. Error: {e}")
        
        return text
    
    @staticmethod
    def extract_from_pdf_ocr(pdf_file: io.BytesIO) -> str:
        """Extract text from PDF using OCR for image-based PDFs."""
        try:
            from pdf2image import convert_from_bytes
            import pytesseract
            
            images = convert_from_bytes(pdf_file.read())
            
            text = ""
            for image in images:
                text += pytesseract.image_to_string(image) + "\n"
            
            return text
        except ImportError:
            print("[TextExtractor] OCR dependencies not installed. Install pdf2image and pytesseract for image-based PDF support.")
            return ""
        except Exception as e:
            print(f"[TextExtractor] OCR extraction error: {e}")
            return ""
    
    @staticmethod
    def extract_from_txt(file_content: bytes) -> str:
        """Extract text from TXT file with UTF-8 fallback to latin-1."""
        try:
            return file_content.decode("utf-8")
        except UnicodeDecodeError:
            return file_content.decode("latin-1")
    
    @classmethod
    def extract(cls, file_content: bytes, file_type: str) -> str:
        """Extract text based on file type."""
        if file_type == ".pdf":
            return cls.extract_from_pdf(file_content)
        elif file_type == ".txt":
            return cls.extract_from_txt(file_content)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean extracted text by normalizing whitespace and removing control characters."""
        text = re.sub(r'\s+', ' ', text)
        text = ''.join(char for char in text if char == '\n' or ord(char) >= 32)
        return text.strip()
