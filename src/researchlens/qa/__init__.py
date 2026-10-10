"""Question Answering and prompt engineering package."""

from researchlens.qa.prompts import QA_USER_TEMPLATE, SCIENTIFIC_QA_SYSTEM_PROMPT
from researchlens.qa.question_answering import (
    NO_EVIDENCE_ANSWER,
    BaseQuestionAnswerer,
    BaseQuestionAnswering,
    ExtractiveQASystem,
    LocalQuestionAnswerer,
    QAResponse,
    SourceCitation,
)

__all__ = [
    "SCIENTIFIC_QA_SYSTEM_PROMPT",
    "QA_USER_TEMPLATE",
    "QAResponse",
    "SourceCitation",
    "BaseQuestionAnswerer",
    "BaseQuestionAnswering",
    "LocalQuestionAnswerer",
    "ExtractiveQASystem",
    "NO_EVIDENCE_ANSWER",
]
