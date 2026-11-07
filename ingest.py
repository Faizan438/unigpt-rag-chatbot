import os
import glob
from io import BytesIO
import shutil

import pymupdf
import chromadb
import pytesseract

from PIL import Image
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_FOLDER = BASE_DIR
DB_PATH = os.path.join(BASE_DIR, "db")

COLLECTION_NAME = "unigpt"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
elif shutil.which("tesseract"):
    pass  
else:
    print("Warning: Tesseract OCR not found. Scanned PDF OCR fallback might fail.")



print("Loading embedding model...")

model = SentenceTransformer(EMBEDDING_MODEL)


def extract_pdf_text(pdf_path):
    filename = os.path.basename(pdf_path)
    pages_data = []

    try:
        with pymupdf.open(pdf_path) as pdf:
            for page_index in range(len(pdf)):
                page = pdf.load_page(page_index)
                page_number = page_index + 1

                text = page.get_text("text", sort=True) or ""

                try:
                    for table in page.find_tables().tables:
                        md = table.to_markdown()
                        if md.strip():
                            text += "\n\n" + md
                except Exception as e:
                    print(f"Table error page {page_number}: {e}")

                if len(text.strip()) < 50 or (page.get_images() and len(text.strip()) < 500):
                    print(f"Running OCR on {filename} page {page_number}...")
                    try:
                        pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                        image = Image.open(BytesIO(pix.tobytes("png")))
                        ocr_text = pytesseract.image_to_string(image, lang="eng") or ""
                        text += "\n\n" + ocr_text
                    except Exception as e:
                        print(f"OCR error page {page_number}: {e}")

                print(f"{filename} page {page_number}: {len(text.strip())} chars")

                if text.strip():
                    pages_data.append({"page": page_number, "text": text.strip()})
    except Exception as e:
        print(f"Could not process {filename}: {e}")

    return pages_data



ROW_PREFIXES = ("BS", "Bs", "BSc", "BBA", "MBBS", "LLB", "Doctor", "Bachelor")

def split_text(text):
    chunks = []
    start = 0
    step = CHUNK_SIZE - CHUNK_OVERLAP

    while start < len(text):
        chunk = text[start:start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def create_chunks(text, filename=""):

    if filename == "UG-Fee-Structure.pdf":
        rows = []
        other_lines = []

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith(ROW_PREFIXES) and any(ch.isdigit() for ch in line):
                rows.append(line)
            else:
                other_lines.append(line)

        chunks = [
            f"Superior University UG fee structure Fall 2026. Program fee row: {row} "
            f"(Columns: Admission Fee one time, Misc fee per semester, "
            f"Number of Semesters, Tuition Fee per semester (1st semester), Total Fee. "
            f"Fees may be revised 5% to 10% annually.)"
            for row in rows
        ]

        if other_lines:
            chunks += split_text("\n".join(other_lines))

        return chunks

    return split_text(text)


print("\nConnecting to ChromaDB...")

db = chromadb.PersistentClient(
    path=DB_PATH
)

try:
    db.delete_collection(
        name=COLLECTION_NAME
    )
    print("Old collection deleted.")

except Exception:
    print("No previous collection found.")

collection = db.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


pdf_files = glob.glob(
    os.path.join(PDF_FOLDER, "*.pdf")
)

print(f"\nPDF files found: {len(pdf_files)}")


all_chunks = []
all_ids = []
all_metadatas = []
facts_path = os.path.join(BASE_DIR, "handbook_facts.txt")
if os.path.exists(facts_path):
    with open(facts_path, encoding="utf-8") as f:
        facts_blocks = [b.strip() for b in f.read().split("\n\n") if b.strip()]
    for i, block in enumerate(facts_blocks):
        all_chunks.append(block)
        all_ids.append(f"handbook_facts_{i}")
        all_metadatas.append({
            "source": "International-Students-Final-Handbook.pdf",
            "page": "11-13 (verified summary)",
            "chunk_id": i,
        })
    print(f"Added {len(facts_blocks)} verified fact chunks.")


for pdf_path in pdf_files:

    filename = os.path.basename(pdf_path)

    pages = extract_pdf_text(pdf_path)

    if not pages:
        print(f"Skipping: {filename}")
        continue

    for page_data in pages:

        page_number = page_data["page"]
        page_text = page_data["text"]

        chunks = create_chunks(page_text, filename)

        print(
            f"Page {page_number}: "
            f"{len(chunks)} chunks created"
        )

        for chunk_index, chunk in enumerate(chunks):

            all_chunks.append(chunk)

            all_ids.append(
                f"{filename}_page_{page_number}_chunk_{chunk_index}"
            )

            all_metadatas.append({
                "source": filename,
                "page": page_number,
                "chunk_id": chunk_index
            })


if all_chunks:

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        all_chunks,
        show_progress_bar=True
    )

    print("\nUploading chunks to ChromaDB...")

    collection.add(
        ids=all_ids,
        documents=all_chunks,
        embeddings=embeddings.tolist(),
        metadatas=all_metadatas
    )

    print("\n================================")
    print("INGESTION COMPLETED SUCCESSFULLY")
    print("================================")

    print(f"Total PDFs found: {len(pdf_files)}")
    print(f"Total chunks uploaded: {len(all_chunks)}")

else:

    print("\nNo readable PDF content found.")



print("\nVerifying database...")

print(
    "Total chunks in database:",
    collection.count()
)

print("Database path:", DB_PATH)