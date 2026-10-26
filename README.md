# DocuMind - Document Question Answering

## Overview
DocuMind is a Python application that lets users upload PDF or TXT documents and search for relevant passages using natural-language questions.

## Features
- Upload PDF and TXT documents
- Extract text from documents
- Split text into searchable passages
- Retrieve relevant passages using TF-IDF and cosine similarity
- Display source page numbers and text similarity scores
- Preview extracted document text
- Simple interactive Streamlit interface

## Technologies
- Python
- Streamlit
- PyMuPDF
- scikit-learn
- TF-IDF
- Cosine similarity

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
## Run the Application

python -m streamlit run app.py

## How It Works
1. Extract text from the uploaded document.
2. Divide text into overlapping passages.
3. Convert passages and the question into TF-IDF vectors.
4. Rank passages using cosine similarity.
5. Display the most relevant passages and their page numbers.

## Limitations
- The application retrieves passages rather than generating complete answers.
- Similarity scores are not confidence scores.
- Scanned PDFs may require OCR.
- Search quality depends on the extracted text and question wording.

## Privacy
Do not upload personal documents or sensitive information to a public repository. The application is intended for educational use.

## Project
Created for the CodeNova AI/ML internship — Task 4.
