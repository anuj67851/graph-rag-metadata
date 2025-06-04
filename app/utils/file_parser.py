import os
from io import BytesIO
from typing import Union, IO

import PyPDF2
import docx
from markdown import markdown
from bs4 import BeautifulSoup # To strip HTML from markdown conversion


class FileParsingError(Exception):
    """Custom exception for errors during file parsing."""
    pass


def parse_txt(file_content: Union[bytes, IO[bytes]]) -> str:
    """
    Parses a plain text file (.txt) and returns its content as a string.

    Args:
        file_content: The content of the file as bytes or a file-like object.

    Returns:
        The extracted text content.

    Raises:
        FileParsingError: If there's an error decoding the file.
    """
    try:
        if isinstance(file_content, bytes):
            text = file_content.decode('utf-8')
        else: # IO[bytes]
            text = file_content.read().decode('utf-8')
        return text
    except UnicodeDecodeError as e:
        # Try other common encodings if UTF-8 fails
        try:
            if isinstance(file_content, bytes):
                text = file_content.decode('latin-1')
            else:
                file_content.seek(0) # Reset stream position
                text = file_content.read().decode('latin-1')
            return text
        except Exception as e_inner:
            raise FileParsingError(f"Error decoding TXT file: {e_inner}")
    except Exception as e:
        raise FileParsingError(f"Error parsing TXT file: {e}")


def parse_pdf(file_content: Union[bytes, IO[bytes]]) -> str:
    """
    Parses a PDF file (.pdf) and extracts text content from all pages.

    Args:
        file_content: The content of the file as bytes or a file-like object.

    Returns:
        The concatenated text content from all pages.

    Raises:
        FileParsingError: If the PDF is encrypted or cannot be read.
    """
    try:
        if isinstance(file_content, bytes):
            pdf_file = BytesIO(file_content)
        else: # IO[bytes]
            pdf_file = file_content

        pdf_reader = PyPDF2.PdfReader(pdf_file)
        if pdf_reader.is_encrypted:
            # Attempt to decrypt with an empty password, common for some "protected" PDFs
            try:
                pdf_reader.decrypt('')
            except Exception:
                raise FileParsingError("Cannot parse encrypted PDF: File is password protected.")

        text_parts = []
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text_parts.append(page.extract_text() or "") # Add empty string if no text
        return "\n".join(text_parts)
    except PyPDF2.errors.PdfReadError as e:
        raise FileParsingError(f"Invalid PDF file or PyPDF2 error: {e}")
    except Exception as e:
        raise FileParsingError(f"Error parsing PDF file: {e}")


def parse_docx(file_content: Union[bytes, IO[bytes]]) -> str:
    """
    Parses a DOCX file (.docx) and extracts text content from all paragraphs.

    Args:
        file_content: The content of the file as bytes or a file-like object.

    Returns:
        The concatenated text content from all paragraphs.

    Raises:
        FileParsingError: If the file is not a valid DOCX or other parsing error occurs.
    """
    try:
        if isinstance(file_content, bytes):
            doc_file = BytesIO(file_content)
        else: # IO[bytes]
            doc_file = file_content

        document = docx.Document(doc_file)
        text_parts = [paragraph.text for paragraph in document.paragraphs]
        return "\n".join(text_parts)
    except Exception as e:
        # python-docx can raise various exceptions, including for non-DOCX XML formats
        raise FileParsingError(f"Error parsing DOCX file: {e}. Ensure it's a valid .docx file.")


def parse_markdown(file_content: Union[bytes, IO[bytes]]) -> str:
    """
    Parses a Markdown file (.md), converts it to HTML, then strips HTML tags to get plain text.

    Args:
        file_content: The content of the file as bytes or a file-like object.

    Returns:
        The extracted plain text content.

    Raises:
        FileParsingError: If there's an error decoding or processing the Markdown.
    """
    try:
        if isinstance(file_content, bytes):
            md_text = file_content.decode('utf-8')
        else: # IO[bytes]
            md_text = file_content.read().decode('utf-8')

        html = markdown(md_text)
        # Use BeautifulSoup to strip HTML tags and get text
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator='\n', strip=True)
        return text
    except UnicodeDecodeError as e:
        try:
            if isinstance(file_content, bytes):
                md_text = file_content.decode('latin-1')
            else:
                file_content.seek(0)
                md_text = file_content.read().decode('latin-1')
            html = markdown(md_text)
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator='\n', strip=True)
            return text
        except Exception as e_inner:
            raise FileParsingError(f"Error decoding Markdown file: {e_inner}")
    except Exception as e:
        raise FileParsingError(f"Error parsing Markdown file: {e}")


def extract_text_from_file(filename: str, file_content: Union[bytes, IO[bytes]]) -> str:
    """
    Detects file type based on extension and calls the appropriate parser.

    Args:
        filename: The name of the file, used to determine its extension.
        file_content: The content of the file as bytes or a file-like object.

    Returns:
        The extracted text content from the file.

    Raises:
        ValueError: If the file type is unsupported.
        FileParsingError: If any error occurs during parsing.
    """
    _, extension = os.path.splitext(filename.lower())

    if extension == ".txt":
        return parse_txt(file_content)
    elif extension == ".pdf":
        return parse_pdf(file_content)
    elif extension == ".docx":
        return parse_docx(file_content)
    elif extension == ".md":
        return parse_markdown(file_content)
    else:
        raise ValueError(f"Unsupported file type: {extension}. Supported types are .txt, .pdf, .docx, .md.")


if __name__ == "__main__":
    # Create dummy files for testing
    # Ensure this script is run from the 'graph_rag_app' root or adjust paths.
    # For simplicity, we'll create content in memory.

    # --- Test TXT ---
    print("--- Testing TXT ---")
    txt_content_bytes = "This is a simple text file.\nIt has two lines.".encode('utf-8')
    try:
        txt_text = extract_text_from_file("sample.txt", txt_content_bytes)
        print("TXT Parsed:\n", txt_text)
    except (ValueError, FileParsingError) as e:
        print(f"Error parsing TXT: {e}")

    # --- Test Markdown ---
    print("\n--- Testing Markdown ---")
    md_content_bytes = "# Title\n\nThis is **bold** and *italic*.\n\n- Item 1\n- Item 2".encode('utf-8')
    try:
        md_text = extract_text_from_file("sample.md", md_content_bytes)
        print("Markdown Parsed (Plain Text):\n", md_text)
    except (ValueError, FileParsingError) as e:
        print(f"Error parsing Markdown: {e}")

    # --- Test DOCX (requires creating a dummy DOCX in memory or on disk) ---
    # This is more complex to create purely in memory without saving.
    # For a quick test, you might manually create a 'dummy.docx' and load it.
    print("\n--- Testing DOCX (manual dummy file needed or skip) ---")
    try:
        # Create a dummy docx in memory
        from docx import Document as DocxDocument
        doc = DocxDocument()
        doc.add_paragraph("Hello, this is a test DOCX paragraph.")
        doc.add_paragraph("Another paragraph for testing.")
        docx_io = BytesIO()
        doc.save(docx_io)
        docx_io.seek(0)
        docx_text = extract_text_from_file("dummy.docx", docx_io)
        print("DOCX Parsed:\n", docx_text)
    except ImportError:
        print("Skipping DOCX test: python-docx not fully available for in-memory creation or import issue.")
    except (ValueError, FileParsingError) as e:
        print(f"Error parsing DOCX: {e}")


    # --- Test PDF (requires creating a dummy PDF in memory or on disk) ---
    # Creating a valid PDF programmatically is non-trivial without external libraries like reportlab.
    # This test will likely be more effective with an actual small PDF file.
    print("\n--- Testing PDF (manual dummy file recommended or skip) ---")
    print("Note: Programmatic PDF creation is complex. Test with a real small PDF if possible.")
    # As a placeholder, let's try with known non-PDF bytes to see error handling
    non_pdf_bytes = b"This is not a PDF."
    try:
        pdf_text = extract_text_from_file("dummy.pdf", non_pdf_bytes)
        print("PDF Parsed:\n", pdf_text) # Should not reach here with non_pdf_bytes
    except FileParsingError as e:
        print(f"Error parsing (expected for non-PDF) PDF: {e}")
    except ValueError as e:
        print(f"Error (ValueError) with PDF: {e}")

    # Test with the sample_document.md we defined earlier
    print("\n--- Testing with sample_document.md ---")
    sample_md_path = "sample_document.md" # Assuming it's in the root
    if os.path.exists(sample_md_path):
        with open(sample_md_path, 'rb') as f:
            try:
                parsed_sample_text = extract_text_from_file(sample_md_path, f)
                print(f"Successfully parsed '{sample_md_path}'. First 300 chars:\n'{parsed_sample_text[:300]}...'")
            except (ValueError, FileParsingError) as e:
                print(f"Error parsing '{sample_md_path}': {e}")
    else:
        print(f"'{sample_md_path}' not found. Skipping test.")