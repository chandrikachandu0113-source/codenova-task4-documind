import re

import pymupdf
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DocuMind - Document Q&A",
    page_icon="📚",
    layout="wide",
)

st.title("📚 DocuMind — Document Question Answering")

st.write(
    "Upload your cover letter or another PDF/TXT document. "
    "Ask questions and find relevant passages from the document."
)

st.info(
    "This free version uses local text retrieval. It finds relevant "
    "passages rather than generating answers with a paid AI API."
)


# ============================================================
# 2. TEXT EXTRACTION
# ============================================================

def extract_pdf_text(file_bytes):
    """Extract selectable text from a PDF, page by page."""

    pages = []

    with pymupdf.open(stream=file_bytes, filetype="pdf") as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "page": page_number,
                    "text": text,
                })

    return pages


def extract_txt_text(file_bytes):
    """Extract text from a TXT file."""

    text = file_bytes.decode(
        "utf-8-sig",
        errors="replace",
    ).strip()

    if not text:
        return []

    return [{
        "page": 1,
        "text": text,
    }]


# ============================================================
# 3. CLEAN AND SPLIT DOCUMENT TEXT
# ============================================================

def clean_text(text):
    """Normalize whitespace without removing meaningful words."""

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_chunks(pages, words_per_chunk=140, overlap=30):
    """Create searchable passages while preserving page numbers."""

    chunks = []

    for page_item in pages:
        words = clean_text(page_item["text"]).split()

        start = 0

        while start < len(words):
            end = min(start + words_per_chunk, len(words))
            passage = " ".join(words[start:end]).strip()

            if passage:
                chunks.append({
                    "page": page_item["page"],
                    "text": passage,
                })

            if end >= len(words):
                break

            start = end - overlap

    return chunks


# ============================================================
# 4. QUESTION KEYWORD EXPANSION
# ============================================================

def expand_question(question):
    """
    Add related words for common questions.
    This improves retrieval but does not invent document facts.
    """

    question_lower = question.lower()

    groups = {
        "skills": [
            "skill", "skills", "strengths", "abilities",
            "proficient", "proficiency", "knowledge",
        ],
        "education": [
            "education", "degree", "college", "university",
            "qualification", "academic", "study",
        ],
        "experience": [
            "experience", "work", "employment", "internship",
            "responsibilities", "achievements",
        ],
        "position": [
            "position", "job", "role", "applying",
            "application", "opportunity", "vacancy",
        ],
        "motivation": [
            "interested", "interest", "motivation",
            "career", "objective", "goal", "purpose",
        ],
        "projects": [
            "project", "projects", "developed", "created",
            "implemented", "built",
        ],
        "contact": [
            "contact", "email", "phone", "address",
            "name", "applicant",
        ],
    }

    additions = []

    for group_words in groups.values():
        if any(word in question_lower for word in group_words):
            additions.extend(group_words)

    return question + " " + " ".join(additions)


# ============================================================
# 5. RETRIEVE RELEVANT PASSAGES
# ============================================================

def find_relevant_passages(question, chunks, top_k=3):
    """
    Rank document passages using word and character TF-IDF.
    No paid API or external AI service is required.
    """

    if not question.strip() or not chunks:
        return []

    texts = [chunk["text"] for chunk in chunks]
    expanded_question = expand_question(question)

    score_arrays = []

    configurations = [
        {
            "analyzer": "word",
            "ngram_range": (1, 2),
            "stop_words": "english",
        },
        {
            "analyzer": "char_wb",
            "ngram_range": (3, 5),
            "stop_words": None,
        },
    ]

    for config in configurations:
        try:
            vectorizer = TfidfVectorizer(
                analyzer=config["analyzer"],
                ngram_range=config["ngram_range"],
                stop_words=config["stop_words"],
                sublinear_tf=True,
            )

            document_matrix = vectorizer.fit_transform(texts)
            question_vector = vectorizer.transform(
                [expanded_question]
            )

            scores = cosine_similarity(
                question_vector,
                document_matrix,
            ).flatten()

            score_arrays.append(scores)

        except ValueError:
            continue

    if not score_arrays:
        return []

    # Combine both retrieval methods.
    combined_scores = sum(score_arrays) / len(score_arrays)

    ranked_indices = combined_scores.argsort()[::-1]

    results = []

    for index in ranked_indices:
        score = float(combined_scores[index])

        if score <= 0:
            continue

        results.append({
            "page": chunks[index]["page"],
            "text": chunks[index]["text"],
            "score": score,
        })

        if len(results) >= top_k:
            break

    return results


# ============================================================
# 6. SIDEBAR: UPLOAD DOCUMENT
# ============================================================

with st.sidebar:
    st.header("📄 Upload your document")

    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=["pdf", "txt"],
        help="Upload your own cover letter or other text document.",
    )

    st.divider()

    st.caption("Supported formats: PDF and TXT")
    st.caption(
        "Scanned PDFs may require OCR, which is not included."
    )


# ============================================================
# 7. HANDLE UPLOAD
# ============================================================

if uploaded_file is None:
    st.subheader("How to use DocuMind")

    st.markdown(
        """
        1. Upload your cover letter or another document.
        2. Confirm the filename shown by the application.
        3. Preview the extracted text.
        4. Enter a question about the document.
        5. Review the retrieved passages and source pages.
        """
    )

    st.stop()


file_bytes = uploaded_file.getvalue()

st.success(
    f"Currently loaded document: {uploaded_file.name}"
)

try:
    if uploaded_file.name.lower().endswith(".pdf"):
        pages = extract_pdf_text(file_bytes)
    else:
        pages = extract_txt_text(file_bytes)

except Exception as error:
    st.error(f"Unable to read the document: {error}")
    st.stop()


# ============================================================
# 8. CHECK EXTRACTED TEXT
# ============================================================

if not pages:
    st.error(
        "No readable text was extracted. If your PDF contains scanned "
        "images, OCR is needed before its text can be searched."
    )
    st.stop()


chunks = split_into_chunks(pages)

if not chunks:
    st.warning("No searchable passages were found.")
    st.stop()


# ============================================================
# 9. DOCUMENT STATISTICS
# ============================================================

total_words = sum(
    len(page["text"].split())
    for page in pages
)

col1, col2, col3 = st.columns(3)

col1.metric("Pages with text", len(pages))
col2.metric("Searchable passages", len(chunks))
col3.metric("Document words", total_words)


# ============================================================
# 10. PREVIEW THE UPLOADED DOCUMENT
# ============================================================

with st.expander("👀 Preview extracted text", expanded=True):

    st.write(f"**File:** {uploaded_file.name}")

    for page_item in pages[:3]:
        st.markdown(f"**Page {page_item['page']}**")
        st.text(page_item["text"][:2500])

    if len(pages) > 3:
        st.caption("Preview limited to the first three pages.")


# ============================================================
# 11. ASK A QUESTION
# ============================================================

st.divider()

st.subheader("💬 Ask a question about your document")

st.write(
    "Ask about information that may appear in your document, "
    "such as its purpose, qualifications, experience, or education."
)

with st.form("question_form"):

    question = st.text_input(
        "Your question",
        placeholder="Example: What is the main purpose of this document?",
    )

    submitted = st.form_submit_button(
        "🔍 Find Answer",
        use_container_width=True,
    )


# ============================================================
# 12. DISPLAY RETRIEVED EVIDENCE
# ============================================================

if submitted:

    if not question.strip():
        st.warning("Please enter a question.")

    else:
        results = find_relevant_passages(
            question,
            chunks,
            top_k=3,
        )

        st.subheader("📌 Answer Evidence")

        if not results:
            st.warning(
                "No matching passage was found. Try using words that "
                "appear in the extracted-text preview."
            )

        else:
            st.write(
                f"Retrieved {len(results)} relevant passage(s) "
                f"from **{uploaded_file.name}**."
            )

            for number, result in enumerate(results, start=1):

                with st.container(border=True):

                    st.markdown(f"**Passage {number}**")

                    st.caption(
                        f"Source page: {result['page']}"
                    )

                    st.write(result["text"])

                    st.caption(
                        f"Text similarity score: {result['score']:.3f}"
                    )

            st.info(
                "These passages are evidence retrieved from the document. "
                "They are not automatically generated answers. Similarity "
                "scores indicate text matching, not factual confidence."
            )


# ============================================================
# 13. FOOTER
# ============================================================

st.divider()

st.caption(
    "DocuMind | CodeNova AI/ML Internship — Task 4 | "
    "Educational document retrieval project"
)
