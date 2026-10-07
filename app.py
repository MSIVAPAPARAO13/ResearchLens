import streamlit as st
from pdf_utils import extract_text_from_pdf
from section_parser import split_into_sections
from hierarchical import summarize_section  # fixed name

st.set_page_config(layout="wide")
st.title("📑 Section-wise Research Paper Summarizer")

uploaded_pdf = st.file_uploader(
    "Upload a research paper (PDF)",
    type=["pdf"]
)

if uploaded_pdf:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_pdf.read())

    raw_text = extract_text_from_pdf("temp.pdf")
    sections = split_into_sections(raw_text)

    for section, text in sections.items():
        with st.expander(section.upper()):
            summary = summarize_section(section, text)  # pass section name
            st.write(summary)

