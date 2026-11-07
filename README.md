# 🎓 UniGPT: Superior University Assistant

A multilingual **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about admissions, fees, visa process and exchange programs, using only official university documents. It understands **English and Roman Urdu** and cites the PDF name and page number for every answer.

![UniGPT demo](docs/demo.jpeg)

*Example: asking for the semester fee of BS Artificial Intelligence. The answer is grounded in `UG-Fee-Structure.pdf` and cites the page.*

---

## Features

- **Multilingual retrieval**: Roman Urdu and English questions work against English documents (`paraphrase-multilingual-MiniLM-L12-v2` embeddings).
- **Grounded answers with citations**: every answer lists the source PDF and page.
- **Hallucination control**: the LLM is instructed to answer only from retrieved context, and a distance threshold filters out irrelevant chunks. If nothing relevant is found, the bot says so instead of guessing.
- **Table-aware ingestion**: the fee structure PDF is parsed into per-program rows, so a question like *"BS Computer Science ki total fee kitni hai?"* retrieves the exact row.
- **OCR fallback**: image-heavy or scanned pages are processed with Tesseract.
- **Prompt-injection guard**: retrieved text is treated as reference material only.
- **Graceful error handling**: rate limits (429), wrong model name (404) and empty database are handled with clear messages.

---

## Architecture

```
PDFs ──► Text + table extraction (PyMuPDF) ──► OCR fallback (Tesseract)
                         │
                         ▼
            Document-specific chunking
   (row-wise for fee tables, fixed-size for the rest)
                         │
                         ▼
   Multilingual embeddings (Sentence-Transformers)
                         │
                         ▼
          ChromaDB (persistent, cosine similarity)
                         │
   User question ──► embed ──► top-k retrieval + distance filter
                         │
                         ▼
        Gemini 2.5 Flash (context-only system prompt)
                         │
                         ▼
          Answer + sources (Streamlit chat UI)
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Vector database | ChromaDB (persistent, cosine) |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (Sentence-Transformers) |
| LLM | Google Gemini (`gemini-2.5-flash`, configurable) |
| PDF parsing | PyMuPDF (text, tables), pypdf |
| OCR | Tesseract via `pytesseract` |

---

## Project Structure

```
├── app.py                 # Streamlit chat interface
├── rag.py                 # Retrieval + prompt + Gemini generation
├── ingest.py              # PDF parsing, chunking, embedding, indexing
├── handbook_facts.txt     # Curated, verified facts layer (see Design Notes)
├── eval/
│   └── questions.json     # Evaluation question set
├── requirements.txt
└── .env.example
```

---

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/unigpt.git
cd unigpt
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

### 2. Install Tesseract (for OCR fallback)

- Windows: install from the [UB Mannheim builds](https://github.com/UB-Mannheim/tesseract/wiki). The default path `C:\Program Files\Tesseract-OCR\tesseract.exe` is detected automatically.
- Linux: `sudo apt install tesseract-ocr`
- macOS: `brew install tesseract`

### 3. Configure environment variables

Create a `.env` file (see `.env.example`):

```
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

Get a free key from [Google AI Studio](https://aistudio.google.com/).

### 4. Add your documents

Place the source PDFs in the project root. This repo expects:

- `International-Students-Final-Handbook.pdf`
- `UG-Fee-Structure.pdf`

### 5. Build the index and run

```bash
python ingest.py
streamlit run app.py
```

---

## Example Questions

| Question | Expected source |
|---|---|
| Fall intake kab hota hai? | Handbook, p. 11 |
| Semester exchange ke liye minimum CGPA kitna hai? | Handbook, p. 17 |
| What documents do I need for the visa process? | Handbook, p. 11 |
| BS Computer Science ki total fee kitni hai? | Fee Structure |

---

## Evaluation

A set of questions with expected answers is stored in `eval/questions.json`.

| Metric | Result |
|---|---|
| Questions | _TBD_ |
| Correct answers | _TBD_ |

> Fill this table after running your evaluation.

---

## Design Notes and Lessons Learned

**1. Different documents need different chunking.**
The fee structure is a table, so each program row becomes its own chunk, enriched with column names so the embedding model understands the numbers. The handbook is a design-heavy brochure; fixed 500-character chunks split facts across boundaries.

**2. Curated facts layer.**
Standard parsers extracted several handbook pages (cards, icons, multi-column layouts) in a broken reading order. To guarantee correct answers for critical information (eligibility, intake dates, visa steps), `handbook_facts.txt` holds a manually verified summary that is indexed alongside the raw PDF chunks. This is a hybrid approach: raw documents for coverage, curated facts for precision.

**3. Gemini 2.5 thinking tokens.**
2.5 models count "thinking" tokens inside `max_output_tokens`, which can truncate answers. Thinking is disabled and the limit raised for this use case.

---

## Limitations and Future Work

- Add hybrid search (BM25 + embeddings) and a reranker.
- Page-level chunking for design-heavy PDFs to remove the need for the manual facts file.
- Conversation memory for follow-up questions.
- Automated evaluation (retrieval hit-rate and answer accuracy) with `run_eval.py`.
- Deployment on Streamlit Community Cloud or Hugging Face Spaces.

---

## Disclaimer

This is an independent portfolio project and is not officially affiliated with Superior University. Fees and policies may change; always confirm with the university.

## Author

**<Your Name>**: [GitHub](https://github.com/<your-username>) · [LinkedIn](https://linkedin.com/in/<your-profile>)
