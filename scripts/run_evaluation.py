import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from researchlens.evaluation.evaluation_runner import EvaluationRunner
from researchlens.ingestion.document import ChunkData
from researchlens.qa.question_answering import LocalQuestionAnswerer
from researchlens.retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    FAISSVectorStore,
    HybridRetriever,
    ScientificRetrievalPipeline,
    get_embedding_model,
)
from researchlens.utils.logging import get_logger

logger = get_logger("run_evaluation")


def run_summarization_eval(runner: EvaluationRunner):
    """Run baseline ROUGE summarization evaluation."""
    logger.info("Running summarization benchmark...")
    sample_predictions = [
        "This paper proposes a convolutional neural network architecture for automated biomedical image segmentation.",
        "We introduce a section-aware summarization pipeline using BART conditional generation and hierarchical synthesis.",
    ]
    sample_references = [
        "A convolutional neural network model is proposed for biomedical image segmentation in academic papers.",
        "We present an academic research paper summarization system using section boundaries and transformer models.",
    ]

    results = runner.evaluate_summaries(sample_predictions, sample_references)

    print("\n" + "=" * 60)
    print("SUMMARIZATION EVALUATION BENCHMARK (ROUGE)")
    print("=" * 60)
    for metric, values in results["rouge_metrics"].items():
        print(
            f"{metric.upper():<8}: Precision={values['precision']:.4f}, "
            f"Recall={values['recall']:.4f}, F1={values['fmeasure']:.4f}"
        )
    print("=" * 60)


def build_evaluation_corpus() -> Tuple[List[ChunkData], List[Tuple[str, str, List[str]]]]:
    """Construct realistic scientific chunks and benchmark query categories."""
    chunks = [
        ChunkData(
            chunk_id="chunk_intro_01",
            text="Deep learning models for scientific document intelligence face significant challenges with domain-specific terminology and multi-modal layouts. In this paper, we address the problem of automated scientific information extraction and semantic document retrieval from unstructured academic literature.",
            token_count=35,
            section_name="introduction",
            page_number=1,
            document_id="doc_scipaper",
            filename="scipaper.pdf",
            chunk_index=0,
        ),
        ChunkData(
            chunk_id="chunk_model_02",
            text="We propose SciTransformer, a multi-scale self-attention transformer architecture with adaptive layer normalization and rotary position embeddings designed specifically for long-context scientific papers.",
            token_count=26,
            section_name="methodology",
            page_number=3,
            document_id="doc_scipaper",
            filename="scipaper.pdf",
            chunk_index=1,
        ),
        ChunkData(
            chunk_id="chunk_train_03",
            text="The model was trained using AdamW optimizer with a learning rate of 2e-5, cosine annealing schedule, and batch size 64 across 8 NVIDIA A100 GPUs for 50 epochs with weight decay 0.01.",
            token_count=34,
            section_name="methodology",
            page_number=4,
            document_id="doc_scipaper",
            filename="scipaper.pdf",
            chunk_index=2,
        ),
        ChunkData(
            chunk_id="chunk_dataset_04",
            text="All experimental models were trained and benchmarked on the ImageNet-1K and PubMed-200k-RCT datasets, evaluated against the SciCite and S2ORC scientific corpora.",
            token_count=24,
            section_name="experiments",
            page_number=5,
            document_id="doc_scipaper",
            filename="scipaper.pdf",
            chunk_index=3,
        ),
        ChunkData(
            chunk_id="chunk_metrics_05",
            text="The evaluation metrics reported include Precision, Recall, Macro-F1 score, Mean Reciprocal Rank (MRR), and Root Mean Squared Error (RMSE) for regression benchmarks.",
            token_count=24,
            section_name="experiments",
            page_number=5,
            document_id="doc_scipaper",
            filename="scipaper.pdf",
            chunk_index=4,
        ),
        ChunkData(
            chunk_id="chunk_results_06",
            text="On the regression benchmark, our model achieved a test RMSE of 0.284 and Mean Absolute Error of 0.192, outperforming baseline models by 14.6%.",
            token_count=23,
            section_name="results",
            page_number=7,
            document_id="doc_scipaper",
            filename="scipaper.pdf",
            chunk_index=5,
        ),
        ChunkData(
            chunk_id="chunk_limits_07",
            text="The authors identified key limitations: quadratic memory complexity for sequences exceeding 4,096 tokens, computational cost of cross-attention reranking, and reduced zero-shot accuracy on non-English scientific papers.",
            token_count=29,
            section_name="discussion",
            page_number=8,
            document_id="doc_scipaper",
            filename="scipaper.pdf",
            chunk_index=6,
        ),
        ChunkData(
            chunk_id="chunk_distractor_08",
            text="Previous works by Vaswani et al. and Devlin et al. established foundational self-attention and masked language modeling paradigms across general natural language processing benchmarks.",
            token_count=24,
            section_name="related_work",
            page_number=2,
            document_id="doc_scipaper",
            filename="scipaper.pdf",
            chunk_index=7,
        ),
        ChunkData(
            chunk_id="chunk_distractor_09",
            text="In conclusion, scientific language modeling benefits from hybrid indexing strategies that bridge lexical keyword matching with dense semantic embeddings.",
            token_count=20,
            section_name="conclusion",
            page_number=9,
            document_id="doc_scipaper",
            filename="scipaper.pdf",
            chunk_index=8,
        ),
    ]

    # Benchmark query suite covering 7 positive query types:
    # (Category, Query, Ground truth chunk IDs)
    eval_queries = [
        ("A. Semantic", "What problem does the paper address?", ["chunk_intro_01"]),
        ("B. Exact entity", "What dataset was used?", ["chunk_dataset_04"]),
        ("C. Model", "What architecture was proposed?", ["chunk_model_02"]),
        ("D. Metric", "What evaluation metrics were reported?", ["chunk_metrics_05"]),
        ("E. Numerical", "What was the RMSE?", ["chunk_results_06"]),
        ("F. Method", "How was the model trained?", ["chunk_train_03"]),
        ("G. Limitation", "What limitations did the authors identify?", ["chunk_limits_07"]),
    ]

    return chunks, eval_queries


def run_retrieval_comparison_eval(runner: EvaluationRunner):
    """Execute complete retrieval comparison across Dense, BM25, Hybrid, and Cross-Encoder."""
    logger.info("Initializing retrieval evaluation corpus and indexes...")
    chunks, eval_queries = build_evaluation_corpus()

    # 1. Build Dense Index
    emb_model = get_embedding_model()
    embeddings = emb_model.embed_documents(chunks)
    vector_store = FAISSVectorStore()
    vector_store.add_chunks(chunks, embeddings)
    dense_retriever = DenseRetriever(
        embedding_model=emb_model,
        vector_store=vector_store,
        similarity_threshold=0.0,
    )

    # 2. Build BM25 Index
    bm25_retriever = BM25Retriever()
    bm25_retriever.index_chunks(chunks)

    # 3. Build Reranker
    reranker = CrossEncoderReranker()

    # 4. Pipeline
    pipeline = ScientificRetrievalPipeline(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
        dense_weight=0.6,
        bm25_weight=0.4,
    )

    # Strategies to evaluate
    strategies = ["dense", "bm25", "hybrid", "hybrid_reranked"]
    strategy_runs: Dict[str, List[List[str]]] = {s: [] for s in strategies}
    relevant_sets: List[List[str]] = [expected for _, _, expected in eval_queries]

    for cat, query, expected_ids in eval_queries:
        for strat in strategies:
            results = pipeline.retrieve(query, strategy=strat, top_k=5)
            retrieved_ids = [r.doc_id for r in results]
            strategy_runs[strat].append(retrieved_ids)

    # 5. Weight Ablation Study on Hybrid Retriever
    weight_configs = [
        ("dense=1.0 / bm25=0.0", 1.0, 0.0),
        ("dense=0.75 / bm25=0.25", 0.75, 0.25),
        ("dense=0.60 / bm25=0.40", 0.60, 0.40),
        ("dense=0.50 / bm25=0.50", 0.50, 0.50),
        ("dense=0.25 / bm25=0.75", 0.25, 0.75),
        ("dense=0.0 / bm25=1.0", 0.0, 1.0),
    ]

    ablation_metrics: Dict[str, Dict[str, float]] = {}
    for name, dw, bw in weight_configs:
        abl_hybrid = HybridRetriever(
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever,
            dense_weight=dw,
            bm25_weight=bw,
        )
        abl_runs = []
        for _, query, _ in eval_queries:
            res = abl_hybrid.retrieve(query, top_k=5)
            abl_runs.append([r.doc_id for r in res])

        abl_eval = runner.evaluate_retrieval(abl_runs, relevant_sets, k_values=[1, 3, 5], save_filename="temp_abl.json")
        ablation_metrics[name] = abl_eval["metrics"]

    # 6. Generate comparison reports
    report = runner.compare_retrieval_strategies(
        strategy_runs=strategy_runs,
        relevant_sets=relevant_sets,
        k_values=[1, 3, 5],
        ablation_runs=ablation_metrics,
        save_json="retrieval_comparison.json",
        save_md="retrieval_comparison.md",
    )

    print("\n" + "=" * 90)
    print("PHASE 3 RETRIEVAL EVALUATION BENCHMARK: STRATEGY COMPARISON")
    print("=" * 90)
    print(f"{'Strategy':<18} | {'P@1':<7} | {'R@1':<7} | {'P@3':<7} | {'R@3':<7} | {'P@5':<7} | {'R@5':<7} | {'MRR':<7} | {'NDCG@3':<7}")
    print("-" * 90)
    for strat, m in report["strategies"].items():
        print(
            f"{strat:<18} | {m['precision@1']:<7.4f} | {m['recall@1']:<7.4f} | "
            f"{m['precision@3']:<7.4f} | {m['recall@3']:<7.4f} | {m['precision@5']:<7.4f} | {m['recall@5']:<7.4f} | "
            f"{m['mrr']:<7.4f} | {m['ndcg@3']:<7.4f}"
        )
    print("=" * 90)

    print("\n" + "=" * 70)
    print("DENSE vs BM25 WEIGHT ABLATION STUDY")
    print("=" * 70)
    print(f"{'Configuration':<26} | {'Precision@3':<12} | {'Recall@3':<10} | {'MRR':<8} | {'NDCG@3':<8}")
    print("-" * 70)
    for config_name, abl_m in ablation_metrics.items():
        print(
            f"{config_name:<26} | {abl_m['precision@3']:<12.4f} | {abl_m['recall@3']:<10.4f} | "
            f"{abl_m['mrr']:<8.4f} | {abl_m['ndcg@3']:<8.4f}"
        )
    print("=" * 70)

    # 7. Test Negative / Unsupported Query
    print("\n" + "=" * 70)
    print("QUERY TYPE H: NEGATIVE / UNSUPPORTED QUERY REFUSAL TEST")
    print("=" * 70)
    negative_query = "What was the GDP of India in 2020?"
    unsupported_candidates = pipeline.retrieve(negative_query, strategy="hybrid_reranked", top_k=5)
    qa_engine = LocalQuestionAnswerer(confidence_threshold=0.25)
    qa_response = qa_engine.answer_question(negative_query, unsupported_candidates)

    print(f"Negative Query: '{negative_query}'")
    print(f"System Answer : '{qa_response.answer}'")
    print(f"Confidence    : {qa_response.confidence:.4f}")
    assert "could not find enough evidence" in qa_response.answer.lower(), "QA must refuse unsupported query!"
    print("Refusal check : PASSED (System successfully refused unsupported query)")
    print("=" * 70 + "\n")


def run_extraction_eval(runner: EvaluationRunner):
    """Run Gold Standard Information Extraction benchmark evaluating Models, Datasets, Metrics, Results, and Limitations."""
    logger.info("Running Phase 4 Scientific Information Extraction benchmark...")
    from researchlens.extraction import extract_research_profile
    from researchlens.ingestion.document import ChunkData, Document, DocumentMetadata, Page

    # Test Paper 1: Computer Vision Paper
    doc1_p1 = Page(page_number=1, text="Deep Learning for Biomedical Image Segmentation\nAuthors: Alice Smith, Bob Jones\nMIT")
    doc1_c1 = ChunkData("d1_c1", "We propose ResNet-50 and U-Net CNN architectures trained on ImageNet-1K and CIFAR-100 datasets.", 18, "methodology", page_number=2)
    doc1_c2 = ChunkData("d1_c2", "Models achieved Accuracy of 94.2% and F1-Score of 0.91.", 12, "results", page_number=5)
    doc1_c3 = ChunkData("d1_c3", "A primary limitation of our approach is the quadratic memory footprint and computational cost.", 15, "discussion", page_number=7)
    doc1 = Document("vision_paper.pdf", doc1_p1.text + "\n" + doc1_c1.text + "\n" + doc1_c2.text + "\n" + doc1_c3.text, [doc1_p1], chunks=[doc1_c1, doc1_c2, doc1_c3], metadata=DocumentMetadata(title="Deep Learning for Biomedical Image Segmentation"))

    # Test Paper 2: Scientific NLP Transformer Paper
    doc2_p1 = Page(page_number=1, text="SciTransformer: Multi-Scale Transformer for Document Intelligence\nAuthors: Carol White\nStanford University")
    doc2_c1 = ChunkData("d2_c1", "We introduce SciTransformer, fine-tuning BERT and RoBERTa on SQuAD and PubMed-200k-RCT corpora.", 18, "methodology", page_number=3)
    doc2_c2 = ChunkData("d2_c2", "Model achieved test RMSE of 0.284 and MAE of 0.192, evaluated with ROUGE-L and BLEU-4 metrics.", 16, "results", page_number=6)
    doc2_c3 = ChunkData("d2_c3", "One limitation is reduced zero-shot accuracy on non-English scientific papers.", 12, "discussion", page_number=8)
    doc2 = Document("nlp_paper.pdf", doc2_p1.text + "\n" + doc2_c1.text + "\n" + doc2_c2.text + "\n" + doc2_c3.text, [doc2_p1], chunks=[doc2_c1, doc2_c2, doc2_c3], metadata=DocumentMetadata(title="SciTransformer: Multi-Scale Transformer for Document Intelligence"))

    # Test Paper 3: Predictive Maintenance ML Paper
    doc3_p1 = Page(page_number=1, text="Machine Learning for Industrial Predictive Maintenance\nAuthors: David Brown\nCMU")
    doc3_c1 = ChunkData("d3_c1", "We benchmark Random Forest, SVM, and XGBoost models on the CMAPSS dataset.", 14, "methodology", page_number=2)
    doc3_c2 = ChunkData("d3_c2", "Experiments yielded MAE of 26.04 and MSE of 12.3, reported via ROC-AUC and MAE metrics.", 15, "results", page_number=4)
    doc3_c3 = ChunkData("d3_c3", "Drawback includes sensitivity to high feature dimensionality.", 8, "discussion", page_number=6)
    doc3 = Document("pm_paper.pdf", doc3_p1.text + "\n" + doc3_c1.text + "\n" + doc3_c2.text + "\n" + doc3_c3.text, [doc3_p1], chunks=[doc3_c1, doc3_c2, doc3_c3], metadata=DocumentMetadata(title="Machine Learning for Industrial Predictive Maintenance"))

    test_docs = [doc1, doc2, doc3]

    # Gold Standard References
    gold_references = {
        "models": [
            ["ResNet-50", "U-Net", "CNN"],
            ["SciTransformer", "BERT", "RoBERTa"],
            ["Random Forest", "SVM", "XGBoost"],
        ],
        "datasets": [
            ["ImageNet-1K", "CIFAR-100"],
            ["SQuAD", "PubMed-200k-RCT"],
            ["CMAPSS"],
        ],
        "metrics": [
            ["Accuracy", "F1-Score"],
            ["RMSE", "MAE", "ROUGE-L", "BLEU-4"],
            ["MAE", "MSE", "ROC-AUC"],
        ],
        "results": [
            ["94.2%", "0.91"],
            ["0.284", "0.192"],
            ["26.04", "12.3"],
        ],
        "limitations": [
            ["quadratic memory footprint"],
            ["zero-shot accuracy on non-English scientific papers"],
            ["high feature dimensionality"],
        ],
    }

    # Generate Predictions via ExtractionPipeline
    category_predictions = {cat: [] for cat in gold_references}

    for doc in test_docs:
        prof = extract_research_profile(doc)
        category_predictions["models"].append([m.value for m in prof.models if m.value != "Not detected"])
        category_predictions["datasets"].append([d.value for d in prof.datasets if d.value != "Not detected"])
        category_predictions["metrics"].append([m.value for m in prof.metrics if m.value != "Not detected"])
        category_predictions["results"].append([r.value for r in prof.results])
        category_predictions["limitations"].append(
            [lim.value for lim in prof.limitations if lim.value != "Not detected"]
        )

    ext_report = runner.evaluate_extraction(
        category_predictions=category_predictions,
        category_references=gold_references,
        num_papers=len(test_docs),
        save_json="extraction_results.json",
        save_md="extraction_report.md",
    )

    print("\n" + "=" * 70)
    print("PHASE 4 SCIENTIFIC INFORMATION EXTRACTION BENCHMARK")
    print("=" * 70)
    print(f"{'Category':<18} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 70)
    for cat, m in ext_report["categories"].items():
        print(f"{cat.title():<18} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1_score']:<10.4f}")
    print("=" * 70 + "\n")


def run_research_gap_eval(runner: EvaluationRunner):
    """Run Phase 5 Research Gap Intelligence evaluation and ablation benchmark."""
    from researchlens.extraction import extract_research_profile
    from researchlens.ingestion.document import ChunkData, Document, DocumentMetadata, Page

    logger.info("Running Phase 5 Research Gap Intelligence benchmark...")

    doc1_p1 = Page(1, "Deep Convolutional Neural Networks for Image Recognition\nJohn Doe\nAbstract: We present ResNet for image recognition.")
    doc1_c1 = ChunkData("d1_c1", "A primary limitation of our approach is the quadratic memory footprint during long-sequence training.", 16, "discussion", page_number=5)
    doc1 = Document("vision_paper.pdf", doc1_p1.text + "\n" + doc1_c1.text, [doc1_p1], chunks=[doc1_c1], metadata=DocumentMetadata(title="Deep Convolutional Neural Networks for Image Recognition"))

    doc2_p1 = Page(1, "Transformer-Based Document Intelligence Architecture\nJane Smith\nAbstract: We propose SciTransformer for multi-scale attention.")
    doc2_c1 = ChunkData("d2_c1", "One limitation is zero-shot accuracy on non-English scientific papers.", 10, "limitations", page_number=7)
    doc2_c2 = ChunkData("d2_c2", "In the future, we plan to extend evaluation to multilingual scientific corpora.", 12, "future work", page_number=8)
    doc2 = Document("nlp_paper.pdf", doc2_p1.text + "\n" + doc2_c1.text + "\n" + doc2_c2.text, [doc2_p1], chunks=[doc2_c1, doc2_c2], metadata=DocumentMetadata(title="Transformer-Based Document Intelligence Architecture"))

    doc3_p1 = Page(1, "Machine Learning for Industrial Predictive Maintenance\nAlice Johnson\nAbstract: We benchmark models on CMAPSS.")
    doc3_c1 = ChunkData("d3_c1", "Drawback includes sensitivity to high feature dimensionality.", 8, "discussion", page_number=6)
    doc3 = Document("pm_paper.pdf", doc3_p1.text + "\n" + doc3_c1.text, [doc3_p1], chunks=[doc3_c1], metadata=DocumentMetadata(title="Machine Learning for Industrial Predictive Maintenance"))

    test_docs = [doc1, doc2, doc3]
    profiles = [extract_research_profile(d) for d in test_docs]

    reference_gaps = [
        [{"statement": "quadratic memory footprint", "category": "EFFICIENCY"}],
        [{"statement": "zero-shot accuracy on non-English", "category": "GENERALIZATION"}, {"statement": "multilingual scientific corpora", "category": "GENERALIZATION"}],
        [{"statement": "sensitivity to high feature dimensionality", "category": "METHODOLOGY"}],
    ]

    gap_report = runner.evaluate_research_gaps(
        predicted_profiles=profiles,
        reference_gaps=reference_gaps,
        num_papers=len(test_docs),
        save_json="gap_results.json",
        save_md="gap_report.md",
    )

    print("\n" + "=" * 75)
    print("PHASE 5 AUTOMATED RESEARCH-GAP INTELLIGENCE BENCHMARK & ABLATION")
    print("=" * 75)
    print(f"{'Ablation Mode':<30} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 75)
    for mode, m in gap_report["ablation_study"].items():
        print(f"{mode:<30} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1_score']:<10.4f}")
    print("=" * 75 + "\n")


def run_literature_eval(runner: EvaluationRunner):
    """Run Phase 6 Multi-Paper Literature Intelligence evaluation and ablation benchmark."""
    from researchlens.analysis import LiteratureCollection, LiteratureDocument
    from researchlens.extraction import extract_research_profile
    from researchlens.ingestion.document import ChunkData, Document, DocumentMetadata, Page

    logger.info("Running Phase 6 Multi-Paper Literature Intelligence benchmark...")

    doc1_p1 = Page(1, "Deep Convolutional Neural Networks for Image Recognition\nJohn Doe\nAbstract: We present ResNet for image recognition.")
    doc1_c1 = ChunkData("d1_c1", "A primary limitation of our approach is the quadratic memory footprint during long-sequence training.", 16, "discussion", page_number=5)
    doc1 = Document("vision_paper.pdf", doc1_p1.text + "\n" + doc1_c1.text, [doc1_p1], chunks=[doc1_c1], metadata=DocumentMetadata(title="Deep Convolutional Neural Networks for Image Recognition"))

    doc2_p1 = Page(1, "Transformer-Based Document Intelligence Architecture\nJane Smith\nAbstract: We propose SciTransformer for multi-scale attention.")
    doc2_c1 = ChunkData("d2_c1", "One limitation is zero-shot accuracy on non-English scientific papers.", 10, "limitations", page_number=7)
    doc2_c2 = ChunkData("d2_c2", "In the future, we plan to extend evaluation to multilingual scientific corpora.", 12, "future work", page_number=8)
    doc2 = Document("nlp_paper.pdf", doc2_p1.text + "\n" + doc2_c1.text + "\n" + doc2_c2.text, [doc2_p1], chunks=[doc2_c1, doc2_c2], metadata=DocumentMetadata(title="Transformer-Based Document Intelligence Architecture"))

    coll = LiteratureCollection(collection_id="eval_coll", collection_name="Benchmark Collection")
    coll.add_document(LiteratureDocument("d1", "vision_paper.pdf", "Vision Paper", profile=extract_research_profile(doc1)))
    coll.add_document(LiteratureDocument("d2", "nlp_paper.pdf", "NLP Paper", profile=extract_research_profile(doc2)))

    reference_themes = [
        ["quadratic memory footprint", "zero-shot accuracy on non-English"],
    ]
    reference_conflicts = [[]]

    lit_report = runner.evaluate_literature_intelligence(
        collections=[coll],
        reference_themes=reference_themes,
        reference_conflicts=reference_conflicts,
        save_json="literature_results.json",
        save_md="literature_report.md",
    )

    print("\n" + "=" * 75)
    print("PHASE 6 MULTI-PAPER LITERATURE INTELLIGENCE BENCHMARK & ABLATION")
    print("=" * 75)
    print(f"{'Clustering Ablation Mode':<30} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 75)
    for mode, m in lit_report["ablation_study"].items():
        print(f"{mode:<30} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1_score']:<10.4f}")
    print("=" * 75 + "\n")


def main():
    runner = EvaluationRunner()
    run_summarization_eval(runner)
    run_retrieval_comparison_eval(runner)
    run_extraction_eval(runner)
    run_research_gap_eval(runner)
    run_literature_eval(runner)


if __name__ == "__main__":
    main()

