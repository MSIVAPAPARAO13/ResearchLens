# Section-Aware Research Paper Summarization

This project explores how NLP techniques can be used to summarize academic research papers in a structured and transparent way. Instead of producing a single generic summary, the system analyzes a research paper PDF, reconstructs its section structure, and generates a separate summary for each section (e.g., Abstract, Introduction, Methodology, Results, Conclusion).

The focus of the project is not only on generating fluent summaries, but on handling **real academic PDFs**, which are often noisy, inconsistently formatted, and difficult to process automatically. The system therefore prioritizes **structure, factual consistency, and reproducibility**, which are important in research and industry workflows.

---

## What the System Does

- Extracts text from academic PDF files  
- Detects and reconstructs section boundaries from noisy layouts  
- Cleans common PDF artifacts such as navigation text, table remnants, and variable codes  
- Applies extractive sentence filtering to keep section-relevant content  
- Generates section-specific summaries using a transformer-based model  
- Provides an interactive Streamlit interface to explore summaries  

---

## System Pipeline

PDF
→ Text extraction
→ Section detection
→ Noise cleaning
→ Sentence filtering
→ Hierarchical summarization
→ Section-wise output


Each section is summarized independently with configurable length constraints, helping the summaries reflect the role each section plays in academic writing (e.g., results vs. methodology).

---

## Project Structure

- app.py # Streamlit application
- pdf_utils.py # PDF text extraction
- section_parser.py # Section detection logic
- chunking.py # Token-aware text chunking
- hierarchical.py # Hierarchical summarization pipeline
- summarizer.py # Transformer-based summarization
- README.md


---

## Tech Stack

- **Python**
- **Hugging Face Transformers** (BART)
- **spaCy** (sentence segmentation)
- **PyMuPDF** (PDF parsing)
- **Streamlit** (interactive UI)

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```
2. Install dependencies:
 ```bash  
   pip install -r requirements.txt
```
3. Download the spaCy English model:
```bash
   python -m spacy download en_core_web_sm
```
Running the Application
```bash
   python -m streamlit run app.py
```

Upload a research paper PDF and explore the generated summaries by section.

---

Why Not Just Use ChatGPT?

General-purpose tools like ChatGPT are handy for quick exploration, but they operate as black boxes. This project was built to demonstrate how a controllable and explainable NLP pipeline can be designed specifically for academic documents.

Unlike prompt-based summarization, this system:
- Explicitly reconstructs document structure
- Applies consistent, section-aware summarization rules
- Exposes intermediate processing steps
- Can be automated and evaluated at scale
These properties are important when working with research papers in academic, enterprise, or privacy-sensitive settings.

---

Evaluation Notes

The quality of the summaries was evaluated qualitatively by checking:
- Faithfulness to the original section content
- Separation between methodology, results, and discussion
- Robustness to noisy PDF formatting
The project intentionally prioritizes structural correctness and factual preservation over stylistic fluency.

---

Limitations

- Academic PDFs often contain OCR and layout noise
- Tables and figures are not parsed semantically
- The summarization model is not fine-tuned specifically on scientific corpora

Despite these limitations, the system produces stable and reliable section-wise summaries across different papers.

---

Future Work
- Fine-tuning on scientific datasets (e.g., arXiv or PubMed)
- Table- and figure-aware summarization
- Multi-paper literature review summarization
- Support for long-context models such as LED or Long-T5
