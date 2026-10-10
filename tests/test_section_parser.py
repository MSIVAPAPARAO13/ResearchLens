"""Unit tests for academic section parsing and detection."""

from researchlens.preprocessing.section_parser import split_into_sections


def test_split_into_sections_canonical_headers():
    raw_text = """
    Title: Attention Mechanisms in Deep Learning
    Abstract
    In this work, we present an attention mechanism that processes sequential data efficiently.
    Our results demonstrate substantial gains over conventional methods across benchmarks.

    Introduction
    Natural language processing has undergone significant transformations in recent years.
    Transformer architectures have emerged as the dominant paradigm.

    Methodology
    We train our architecture on the standard benchmark using Adam optimizer with warmup.
    The learning rate was initialized to 5e-5.

    Results
    The model achieved an accuracy of 92.5% on the test split, outperforming baseline models.

    Conclusion
    We proposed an improved model architecture and validated its empirical efficacy.
    """

    sections = split_into_sections(raw_text, min_length=20)
    assert "abstract" in sections
    assert "introduction" in sections
    assert "methodology" in sections
    assert "results" in sections
    assert "conclusion" in sections

    assert "attention mechanism" in sections["abstract"].lower()
    assert "transformer architectures" in sections["introduction"].lower()
    assert "accuracy of 92.5%" in sections["results"].lower()


def test_section_filtering_threshold():
    raw_text = """
    Abstract
    Short
    Introduction
    This is a long introductory text that clearly surpasses the minimal length threshold.
    """
    sections = split_into_sections(raw_text, min_length=50)
    assert "abstract" not in sections  # 'Short' is below 50 chars
    assert "introduction" in sections
