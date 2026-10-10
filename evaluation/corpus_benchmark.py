"""Controlled 10-Paper Gold-Standard Scientific Evaluation Corpus for Phase 8."""

from dataclasses import dataclass
from typing import List

from researchlens.ingestion.document import ChunkData, Document, DocumentMetadata, Page
from researchlens.preprocessing.section_parser import split_into_sections


@dataclass
class GoldQAPair:
    """Benchmark Question-Answer-Citation ground truth pair."""

    question: str
    category: str  # factoid, method, dataset, metric, result, limitation, unsupported
    expected_answer_keywords: List[str]
    expected_chunk_ids: List[str]
    expected_page: int
    expected_section: str
    is_supported: bool = True


@dataclass
class BenchmarkPaper:
    """Standardized representation of an evaluated research paper with annotations."""

    paper_id: str
    title: str
    domain: str
    year: int
    authors: List[str]
    dataset: str
    task: str
    full_text: str
    pages: List[Page]
    chunks: List[ChunkData]
    gold_models: List[str]
    gold_datasets: List[str]
    gold_metrics: List[str]
    gold_results: List[str]
    gold_limitations: List[str]
    gold_future_work: List[str]
    qa_pairs: List[GoldQAPair]

    def to_document(self) -> Document:
        """Convert BenchmarkPaper to Document object."""
        doc = Document(
            filename=f"{self.paper_id}.pdf",
            full_text=self.full_text,
            pages=self.pages,
            chunks=self.chunks,
            metadata=DocumentMetadata(
                title=self.title,
                author=", ".join(self.authors),
            ),
        )
        doc.sections = split_into_sections(self.full_text)
        return doc


def build_10_paper_corpus() -> List[BenchmarkPaper]:
    """Construct a rigorous 10-paper benchmark corpus spanning diverse scientific fields."""
    papers: List[BenchmarkPaper] = []

    # 1. NLP Paper
    p1_text = """Title: SciTransformer: Multi-Scale Transformer for Document Intelligence
Authors: Carol White, David Lee
Abstract: We present SciTransformer, a multi-scale self-attention transformer architecture with adaptive layer normalization and rotary position embeddings designed specifically for long-context scientific papers.
Introduction: Scientific document intelligence presents steep challenges due to technical lexicons and complex hierarchical section layouts.
Methodology: We propose SciTransformer, fine-tuning BERT and RoBERTa backbones with cross-attention on SQuAD and PubMed-200k-RCT corpora.
Experiments: Models are trained using AdamW optimizer with learning rate 2e-5 across 8 NVIDIA A100 GPUs.
Results: On the regression benchmark, our model achieved a test RMSE of 0.284 and MAE of 0.192, evaluated with ROUGE-L and BLEU-4 metrics.
Limitations: One limitation is reduced zero-shot accuracy on non-English scientific papers and quadratic memory footprint.
Future Work: In the future, we plan to extend evaluation to multilingual scientific corpora and sparse attention.
Conclusion: SciTransformer demonstrates superior representation fidelity on academic text parsing."""
    p1_chunks = [
        ChunkData("p1_c1", p1_text[:350], 45, "abstract", page_number=1, document_id="p1", filename="p1.pdf"),
        ChunkData("p1_c2", "We propose SciTransformer, fine-tuning BERT and RoBERTa backbones with cross-attention on SQuAD and PubMed-200k-RCT corpora.", 22, "methodology", page_number=3, document_id="p1", filename="p1.pdf"),
        ChunkData("p1_c3", "On the regression benchmark, our model achieved a test RMSE of 0.284 and MAE of 0.192, evaluated with ROUGE-L and BLEU-4 metrics.", 24, "results", page_number=5, document_id="p1", filename="p1.pdf"),
        ChunkData("p1_c4", "One limitation is reduced zero-shot accuracy on non-English scientific papers and quadratic memory footprint.", 18, "limitations", page_number=7, document_id="p1", filename="p1.pdf"),
    ]
    p1_qa = [
        GoldQAPair("What model was proposed?", "method", ["SciTransformer", "BERT", "RoBERTa"], ["p1_c2"], 3, "methodology"),
        GoldQAPair("What was the test RMSE achieved?", "result", ["0.284"], ["p1_c3"], 5, "results"),
        GoldQAPair("What are the limitations of the model?", "limitation", ["zero-shot", "non-English", "quadratic memory"], ["p1_c4"], 7, "limitations"),
        GoldQAPair("What is the inflation rate in France?", "unsupported", ["not find enough evidence"], [], 0, "none", is_supported=False),
    ]
    papers.append(BenchmarkPaper(
        "p1_nlp", "SciTransformer: Multi-Scale Transformer for Document Intelligence",
        "NLP", 2024, ["Carol White", "David Lee"], "PubMed-200k-RCT", "Document Intelligence",
        p1_text, [Page(1, p1_text)], p1_chunks,
        ["SciTransformer", "BERT", "RoBERTa"], ["SQuAD", "PubMed-200k-RCT"], ["RMSE", "MAE", "ROUGE-L", "BLEU-4"],
        ["0.284", "0.192"], ["reduced zero-shot accuracy on non-English", "quadratic memory footprint"],
        ["multilingual scientific corpora", "sparse attention"], p1_qa
    ))

    # 2. Computer Vision Paper
    p2_text = """Title: Deep Residual U-Net for Biomedical Image Segmentation
Authors: Alice Smith, Bob Jones
Abstract: We present ResNet-50 and U-Net convolutional architectures for automated segmentation of microscopic cellular tissues.
Introduction: Biomedical microscopy imaging requires pixel-level boundary demarcation under varying contrast.
Methodology: The architecture pairs a ResNet-50 backbone with U-Net decoder skip-connections trained on ImageNet-1K and CIFAR-100 datasets.
Experiments: Pre-processed stained histology slices were trained over 100 epochs using binary cross-entropy loss.
Results: Models achieved Accuracy of 94.2% and F1-Score of 0.91 on held-out validation sets.
Limitations: A primary limitation of our approach is the quadratic memory footprint and computational cost during high-resolution inference.
Future Work: Future investigations will explore mixed-precision quantization for edge deployment.
Conclusion: Deep Residual U-Net sets a competitive benchmark in cellular image segmentation."""
    p2_chunks = [
        ChunkData("p2_c1", p2_text[:300], 40, "abstract", page_number=1, document_id="p2", filename="p2.pdf"),
        ChunkData("p2_c2", "The architecture pairs a ResNet-50 backbone with U-Net decoder skip-connections trained on ImageNet-1K and CIFAR-100 datasets.", 21, "methodology", page_number=2, document_id="p2", filename="p2.pdf"),
        ChunkData("p2_c3", "Models achieved Accuracy of 94.2% and F1-Score of 0.91 on held-out validation sets.", 16, "results", page_number=4, document_id="p2", filename="p2.pdf"),
        ChunkData("p2_c4", "A primary limitation of our approach is the quadratic memory footprint and computational cost during high-resolution inference.", 20, "limitations", page_number=6, document_id="p2", filename="p2.pdf"),
    ]
    p2_qa = [
        GoldQAPair("Which datasets were used for training?", "dataset", ["ImageNet-1K", "CIFAR-100"], ["p2_c2"], 2, "methodology"),
        GoldQAPair("What was the accuracy achieved?", "result", ["94.2%"], ["p2_c3"], 4, "results"),
        GoldQAPair("What is the primary computational bottleneck?", "limitation", ["quadratic memory", "computational cost"], ["p2_c4"], 6, "limitations"),
        GoldQAPair("Who directed the movie Inception?", "unsupported", ["not find enough evidence"], [], 0, "none", is_supported=False),
    ]
    papers.append(BenchmarkPaper(
        "p2_cv", "Deep Residual U-Net for Biomedical Image Segmentation",
        "Computer Vision", 2023, ["Alice Smith", "Bob Jones"], "ImageNet-1K", "Cellular Segmentation",
        p2_text, [Page(1, p2_text)], p2_chunks,
        ["ResNet-50", "U-Net", "CNN"], ["ImageNet-1K", "CIFAR-100"], ["Accuracy", "F1-Score"],
        ["94.2%", "0.91"], ["quadratic memory footprint", "computational cost"],
        ["mixed-precision quantization"], p2_qa
    ))

    # 3. Machine Learning / Optimization Paper
    p3_text = """Title: Adaptive Gradient Optimization with Decoupled Weight Decay
Authors: Elena Rostova, Marcus Vance
Abstract: We analyze convergence behaviors of decoupled weight decay algorithms in non-convex stochastic gradient optimization.
Introduction: Standard Adam optimizer ties L2 regularization to gradients, leading to suboptimal generalization.
Methodology: We evaluate AdamW and SGD baseline optimizers on Transformer and ResNet architectures.
Experiments: Benchmarked across CIFAR-10 and ImageNet classification tasks using cosine annealing schedules.
Results: Our configuration attained top-1 Accuracy of 88.6% and reduced loss by 18.4%.
Limitations: Sensitivity to initial learning rate hyperparameter choices remains an empirical challenge.
Future Work: We aim to formulate automated scheduling heuristics without manual tuning.
Conclusion: Decoupled weight decay reliably stabilizes deep network convergence."""
    p3_chunks = [
        ChunkData("p3_c1", p3_text[:300], 38, "abstract", page_number=1, document_id="p3", filename="p3.pdf"),
        ChunkData("p3_c2", "We evaluate AdamW and SGD baseline optimizers on Transformer and ResNet architectures.", 15, "methodology", page_number=2, document_id="p3", filename="p3.pdf"),
        ChunkData("p3_c3", "Our configuration attained top-1 Accuracy of 88.6% and reduced loss by 18.4%.", 15, "results", page_number=4, document_id="p3", filename="p3.pdf"),
        ChunkData("p3_c4", "Sensitivity to initial learning rate hyperparameter choices remains an empirical challenge.", 14, "limitations", page_number=5, document_id="p3", filename="p3.pdf"),
    ]
    p3_qa = [
        GoldQAPair("What optimizers were benchmarked?", "method", ["AdamW", "SGD"], ["p3_c2"], 2, "methodology"),
        GoldQAPair("What top-1 accuracy was reported?", "result", ["88.6%"], ["p3_c3"], 4, "results"),
        GoldQAPair("What is the main drawback noted?", "limitation", ["Sensitivity to initial learning rate"], ["p3_c4"], 5, "limitations"),
        GoldQAPair("What is the population of Tokyo?", "unsupported", ["not find enough evidence"], [], 0, "none", is_supported=False),
    ]
    papers.append(BenchmarkPaper(
        "p3_opt", "Adaptive Gradient Optimization with Decoupled Weight Decay",
        "Machine Learning", 2023, ["Elena Rostova", "Marcus Vance"], "CIFAR-10", "Stochastic Optimization",
        p3_text, [Page(1, p3_text)], p3_chunks,
        ["AdamW", "SGD", "Transformer", "ResNet"], ["CIFAR-10", "ImageNet"], ["Accuracy"],
        ["88.6%"], ["Sensitivity to initial learning rate"],
        ["automated scheduling heuristics"], p3_qa
    ))

    # 4. Time Series Paper
    p4_text = """Title: Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting
Authors: Bryan Lim, Sercan Arik
Abstract: Multi-horizon forecasting combines static covariates and exogenous inputs to forecast dynamic financial and supply-chain signals.
Introduction: Classical autoregressive models struggle to capture heterogeneous temporal relationships.
Methodology: We propose Temporal Fusion Transformer (TFT) utilizing LSTM encoders and sparse self-attention mechanisms on the Electricity and Traffic datasets.
Experiments: Evaluated across quantile loss metrics including Quantile Loss and MAE over 5-year historical benchmarks.
Results: TFT achieved Quantile-Loss of 0.052 and MAE of 14.2, outperforming DeepAR baselines by 12.3%.
Limitations: High computational latency during multi-head attention processing on sequence horizons exceeding 1,000 steps.
Future Work: Explore linear state-space models as drop-in attention replacements.
Conclusion: TFT offers high predictive accuracy alongside interpretable attention maps."""
    p4_chunks = [
        ChunkData("p4_c1", p4_text[:300], 40, "abstract", page_number=1, document_id="p4", filename="p4.pdf"),
        ChunkData("p4_c2", "We propose Temporal Fusion Transformer (TFT) utilizing LSTM encoders and sparse self-attention mechanisms on the Electricity and Traffic datasets.", 23, "methodology", page_number=3, document_id="p4", filename="p4.pdf"),
        ChunkData("p4_c3", "TFT achieved Quantile-Loss of 0.052 and MAE of 14.2, outperforming DeepAR baselines by 12.3%.", 17, "results", page_number=5, document_id="p4", filename="p4.pdf"),
        ChunkData("p4_c4", "High computational latency during multi-head attention processing on sequence horizons exceeding 1,000 steps.", 16, "limitations", page_number=7, document_id="p4", filename="p4.pdf"),
    ]
    p4_qa = [
        GoldQAPair("What model architecture was proposed?", "method", ["Temporal Fusion Transformer", "TFT", "LSTM"], ["p4_c2"], 3, "methodology"),
        GoldQAPair("What MAE score did the model achieve?", "result", ["14.2"], ["p4_c3"], 5, "results"),
        GoldQAPair("What limitation occurs with long sequences?", "limitation", ["High computational latency", "1,000 steps"], ["p4_c4"], 7, "limitations"),
        GoldQAPair("What is the recipe for chocolate cake?", "unsupported", ["not find enough evidence"], [], 0, "none", is_supported=False),
    ]
    papers.append(BenchmarkPaper(
        "p4_ts", "Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting",
        "Time Series", 2022, ["Bryan Lim", "Sercan Arik"], "Electricity", "Forecasting",
        p4_text, [Page(1, p4_text)], p4_chunks,
        ["Temporal Fusion Transformer", "TFT", "LSTM"], ["Electricity", "Traffic"], ["Quantile-Loss", "MAE"],
        ["0.052", "14.2"], ["High computational latency during multi-head attention"],
        ["linear state-space models"], p4_qa
    ))

    # 5. Healthcare / Clinical NLP Paper
    p5_text = """Title: Clinical Predictive Modeling Using Graph Neural Networks on Electronic Health Records
Authors: Sarah Jenkins, Neil Patel
Abstract: Graph representation learning applied to clinical diagnosis codes improves readmission risk stratifications.
Introduction: Tabular patient medical records lack contextual relational representations connecting comorbidities.
Methodology: We propose ClinicalGNN using Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT) trained on the MIMIC-III EHR dataset.
Experiments: Models were trained on 40,000 inpatient admissions with binary patient mortality classification targets.
Results: ClinicalGNN achieved ROC-AUC of 0.884 and Precision of 0.821, outperforming Logistic Regression baselines.
Limitations: Evaluation was restricted to a single hospital system benchmark dataset, limiting demographic generalization.
Future Work: Validate cross-institutional transfer learning across heterogeneous healthcare networks.
Conclusion: Graph neural representations significantly elevate clinical predictive precision."""
    p5_chunks = [
        ChunkData("p5_c1", p5_text[:300], 36, "abstract", page_number=1, document_id="p5", filename="p5.pdf"),
        ChunkData("p5_c2", "We propose ClinicalGNN using Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT) trained on the MIMIC-III EHR dataset.", 22, "methodology", page_number=2, document_id="p5", filename="p5.pdf"),
        ChunkData("p5_c3", "ClinicalGNN achieved ROC-AUC of 0.884 and Precision of 0.821, outperforming Logistic Regression baselines.", 16, "results", page_number=4, document_id="p5", filename="p5.pdf"),
        ChunkData("p5_c4", "Evaluation was restricted to a single hospital system benchmark dataset, limiting demographic generalization.", 16, "limitations", page_number=6, document_id="p5", filename="p5.pdf"),
    ]
    p5_qa = [
        GoldQAPair("What graph architectures were implemented?", "method", ["ClinicalGNN", "GCN", "GAT"], ["p5_c2"], 2, "methodology"),
        GoldQAPair("What ROC-AUC was obtained?", "result", ["0.884"], ["p5_c3"], 4, "results"),
        GoldQAPair("What was the demographic generalization constraint?", "limitation", ["single hospital system", "benchmark dataset"], ["p5_c4"], 6, "limitations"),
        GoldQAPair("How tall is Mount Everest?", "unsupported", ["not find enough evidence"], [], 0, "none", is_supported=False),
    ]
    papers.append(BenchmarkPaper(
        "p5_health", "Clinical Predictive Modeling Using Graph Neural Networks on Electronic Health Records",
        "Healthcare", 2023, ["Sarah Jenkins", "Neil Patel"], "MIMIC-III", "Readmission Risk",
        p5_text, [Page(1, p5_text)], p5_chunks,
        ["ClinicalGNN", "GCN", "GAT"], ["MIMIC-III"], ["ROC-AUC", "Precision"],
        ["0.884", "0.821"], ["Evaluation was restricted to a single hospital system benchmark dataset"],
        ["cross-institutional transfer learning"], p5_qa
    ))

    # 6. Industrial Predictive Maintenance Paper
    p6_text = """Title: Deep Convolutional Neural Networks for Industrial Tool Remaining Useful Life Prediction
Authors: David Brown, Frank Miller
Abstract: Prognostics of turbofan engine degradation enables preventative maintenance scheduling before catastrophic failures occur.
Introduction: Run-to-failure sensor data exhibits non-linear noise and multi-sensor correlations.
Methodology: We benchmark Random Forest, SVM, and 1D-CNN architectures on the CMAPSS dataset.
Experiments: Evaluated under four operating regimes across FD001 and FD002 sensor sequences.
Results: Experiments yielded MAE of 26.04 and MSE of 12.3 on the test harness.
Limitations: Drawback includes sensitivity to high feature dimensionality and sensor noise calibration.
Future Work: Incorporate physics-informed degradation priors into neural loss functions.
Conclusion: Deep convolutional feature extraction reliably predicts machinery failure horizons."""
    p6_chunks = [
        ChunkData("p6_c1", p6_text[:300], 35, "abstract", page_number=1, document_id="p6", filename="p6.pdf"),
        ChunkData("p6_c2", "We benchmark Random Forest, SVM, and 1D-CNN architectures on the CMAPSS dataset.", 15, "methodology", page_number=2, document_id="p6", filename="p6.pdf"),
        ChunkData("p6_c3", "Experiments yielded MAE of 26.04 and MSE of 12.3 on the test harness.", 14, "results", page_number=4, document_id="p6", filename="p6.pdf"),
        ChunkData("p6_c4", "Drawback includes sensitivity to high feature dimensionality and sensor noise calibration.", 14, "limitations", page_number=6, document_id="p6", filename="p6.pdf"),
    ]
    p6_qa = [
        GoldQAPair("What models were compared for prognostic estimation?", "method", ["Random Forest", "SVM", "1D-CNN"], ["p6_c2"], 2, "methodology"),
        GoldQAPair("What was the test MAE on CMAPSS?", "result", ["26.04"], ["p6_c3"], 4, "results"),
        GoldQAPair("What causes vulnerability in tool life prediction?", "limitation", ["high feature dimensionality", "sensor noise"], ["p6_c4"], 6, "limitations"),
        GoldQAPair("Who won the 1994 FIFA World Cup?", "unsupported", ["not find enough evidence"], [], 0, "none", is_supported=False),
    ]
    papers.append(BenchmarkPaper(
        "p6_eng", "Deep Convolutional Neural Networks for Industrial Tool Remaining Useful Life Prediction",
        "Engineering", 2022, ["David Brown", "Frank Miller"], "CMAPSS", "Remaining Useful Life",
        p6_text, [Page(1, p6_text)], p6_chunks,
        ["Random Forest", "SVM", "1D-CNN"], ["CMAPSS"], ["MAE", "MSE"],
        ["26.04", "12.3"], ["sensitivity to high feature dimensionality"],
        ["physics-informed degradation priors"], p6_qa
    ))

    # 7. Speech / Audio Paper
    p7_text = """Title: Conformer: Convolution-Augmented Transformer for End-to-End Speech Recognition
Authors: Anmol Gulati, James Qin
Abstract: Conformer combines depthwise separable convolutions with self-attention for acoustic feature modeling.
Introduction: Transformers excel at global context whereas convolutional networks capture local acoustic patterns.
Methodology: The proposed Conformer model integrates feed-forward modules, multi-head attention, and convolutional modules evaluated on LibriSpeech.
Experiments: Models were trained on 960 hours of speech using SpecAugment data augmentations.
Results: Conformer achieved Word Error Rate (WER) of 1.9% on test-clean and 4.4% on test-other benchmarks.
Limitations: Streaming inference is constrained by bidirectional attention lookahead requirements and latency.
Future Work: Implement causal chunked attention mechanisms for zero-latency mobile transcription.
Conclusion: Conformer demonstrates state-of-the-art automatic speech recognition efficiency."""
    p7_chunks = [
        ChunkData("p7_c1", p7_text[:300], 38, "abstract", page_number=1, document_id="p7", filename="p7.pdf"),
        ChunkData("p7_c2", "The proposed Conformer model integrates feed-forward modules, multi-head attention, and convolutional modules evaluated on LibriSpeech.", 21, "methodology", page_number=3, document_id="p7", filename="p7.pdf"),
        ChunkData("p7_c3", "Conformer achieved Word Error Rate (WER) of 1.9% on test-clean and 4.4% on test-other benchmarks.", 18, "results", page_number=5, document_id="p7", filename="p7.pdf"),
        ChunkData("p7_c4", "Streaming inference is constrained by bidirectional attention lookahead requirements and latency.", 14, "limitations", page_number=7, document_id="p7", filename="p7.pdf"),
    ]
    p7_qa = [
        GoldQAPair("What model was tested on LibriSpeech?", "method", ["Conformer", "Transformer"], ["p7_c2"], 3, "methodology"),
        GoldQAPair("What Word Error Rate was achieved on test-clean?", "result", ["1.9%"], ["p7_c3"], 5, "results"),
        GoldQAPair("Why is streaming transcription constrained?", "limitation", ["bidirectional attention", "latency"], ["p7_c4"], 7, "limitations"),
        GoldQAPair("What is the speed of sound in seawater?", "unsupported", ["not find enough evidence"], [], 0, "none", is_supported=False),
    ]
    papers.append(BenchmarkPaper(
        "p7_speech", "Conformer: Convolution-Augmented Transformer for End-to-End Speech Recognition",
        "Speech", 2022, ["Anmol Gulati", "James Qin"], "LibriSpeech", "Speech Recognition",
        p7_text, [Page(1, p7_text)], p7_chunks,
        ["Conformer", "Transformer"], ["LibriSpeech"], ["WER"],
        ["1.9%", "4.4%"], ["Streaming inference is constrained by bidirectional attention"],
        ["causal chunked attention"], p7_qa
    ))

    # 8. Reinforcement Learning Paper
    p8_text = """Title: Offline Reinforcement Learning via Conservative Q-Learning for Robotics Control
Authors: Aviral Kumar, Sergey Levine
Abstract: Offline reinforcement learning learns decision policies from static demonstration buffers without environment exploration.
Introduction: Out-of-distribution state actions cause distributional shift and catastrophic Q-value overestimation.
Methodology: Conservative Q-Learning (CQL) penalizes state-action values under policy distributions benchmarked against D4RL environments.
Experiments: Evaluated across robotic locomotion and manipulation benchmark suites over 1 million gradient steps.
Results: CQL achieved Normalized Score of 78.4 on hopper-medium, outperforming standard SAC baselines by 24.1%.
Limitations: Over-conservative value estimates can hinder discovery of non-suboptimal trajectories.
Future Work: Develop adaptive conservatism scaling tuned dynamically during policy rollouts.
Conclusion: CQL establishes robust offline policy stability on complex robotics benchmarks."""
    p8_chunks = [
        ChunkData("p8_c1", p8_text[:300], 37, "abstract", page_number=1, document_id="p8", filename="p8.pdf"),
        ChunkData("p8_c2", "Conservative Q-Learning (CQL) penalizes state-action values under policy distributions benchmarked against D4RL environments.", 18, "methodology", page_number=2, document_id="p8", filename="p8.pdf"),
        ChunkData("p8_c3", "CQL achieved Normalized Score of 78.4 on hopper-medium, outperforming standard SAC baselines by 24.1%.", 17, "results", page_number=4, document_id="p8", filename="p8.pdf"),
        ChunkData("p8_c4", "Over-conservative value estimates can hinder discovery of non-suboptimal trajectories.", 13, "limitations", page_number=6, document_id="p8", filename="p8.pdf"),
    ]
    p8_qa = [
        GoldQAPair("What algorithm was introduced for offline policy optimization?", "method", ["Conservative Q-Learning", "CQL"], ["p8_c2"], 2, "methodology"),
        GoldQAPair("What normalized score was recorded on hopper-medium?", "result", ["78.4"], ["p8_c3"], 4, "results"),
        GoldQAPair("What is the downside of conservative value bounds?", "limitation", ["Over-conservative", "trajectories"], ["p8_c4"], 6, "limitations"),
        GoldQAPair("What is the capital of Australia?", "unsupported", ["not find enough evidence"], [], 0, "none", is_supported=False),
    ]
    papers.append(BenchmarkPaper(
        "p8_rl", "Offline Reinforcement Learning via Conservative Q-Learning for Robotics Control",
        "Reinforcement Learning", 2023, ["Aviral Kumar", "Sergey Levine"], "D4RL", "Robotics Policy Control",
        p8_text, [Page(1, p8_text)], p8_chunks,
        ["Conservative Q-Learning", "CQL", "SAC"], ["D4RL"], ["Normalized Score"],
        ["78.4"], ["Over-conservative value estimates"],
        ["adaptive conservatism scaling"], p8_qa
    ))

    # 9. Graph Representation Learning Paper
    p9_text = """Title: Inductive Representation Learning on Large-Scale Heterogeneous Knowledge Graphs
Authors: William Hamilton, Jure Leskovec
Abstract: Node classification across multi-relational knowledge graphs requires inductive node neighborhood aggregations.
Introduction: Transductive embedding techniques fail when novel entities are added during graph expansion.
Methodology: GraphSAGE and Relational GCN architectures aggregate localized node embeddings on the OGB-MAG and WordNet datasets.
Experiments: Benchmarked across academic citation networks evaluating link prediction and node typing accuracy.
Results: Achieved Micro-F1 of 0.842 and MRR of 0.765, outperforming DeepWalk baselines.
Limitations: Quadratic memory footprint during multi-hop neighborhood sampling on dense graph hubs.
Future Work: Investigate sub-graph sketching and randomized pruning to bound neighborhood expansion.
Conclusion: Inductive graph convolutions scale successfully to web-scale relational graphs."""
    p9_chunks = [
        ChunkData("p9_c1", p9_text[:300], 36, "abstract", page_number=1, document_id="p9", filename="p9.pdf"),
        ChunkData("p9_c2", "GraphSAGE and Relational GCN architectures aggregate localized node embeddings on the OGB-MAG and WordNet datasets.", 18, "methodology", page_number=3, document_id="p9", filename="p9.pdf"),
        ChunkData("p9_c3", "Achieved Micro-F1 of 0.842 and MRR of 0.765, outperforming DeepWalk baselines.", 15, "results", page_number=5, document_id="p9", filename="p9.pdf"),
        ChunkData("p9_c4", "Quadratic memory footprint during multi-hop neighborhood sampling on dense graph hubs.", 14, "limitations", page_number=7, document_id="p9", filename="p9.pdf"),
    ]
    p9_qa = [
        GoldQAPair("Which graph neural frameworks were analyzed?", "method", ["GraphSAGE", "Relational GCN"], ["p9_c2"], 3, "methodology"),
        GoldQAPair("What Micro-F1 was attained on OGB-MAG?", "result", ["0.842"], ["p9_c3"], 5, "results"),
        GoldQAPair("What constraint arises on dense graph hubs?", "limitation", ["Quadratic memory footprint", "neighborhood sampling"], ["p9_c4"], 7, "limitations"),
        GoldQAPair("Who painted the Mona Lisa?", "unsupported", ["not find enough evidence"], [], 0, "none", is_supported=False),
    ]
    papers.append(BenchmarkPaper(
        "p9_graph", "Inductive Representation Learning on Large-Scale Heterogeneous Knowledge Graphs",
        "Graph ML", 2022, ["William Hamilton", "Jure Leskovec"], "OGB-MAG", "Inductive Node Classification",
        p9_text, [Page(1, p9_text)], p9_chunks,
        ["GraphSAGE", "Relational GCN", "DeepWalk"], ["OGB-MAG", "WordNet"], ["Micro-F1", "MRR"],
        ["0.842", "0.765"], ["Quadratic memory footprint during multi-hop neighborhood sampling"],
        ["sub-graph sketching", "randomized pruning"], p9_qa
    ))

    # 10. Scientific AI / Physics-Informed ML Paper
    p10_text = """Title: Physics-Informed Neural Networks for High-Dimensional Differential Equations
Authors: George Karniadakis, Paris Perdikaris
Abstract: Physics-Informed Neural Networks (PINNs) integrate Navier-Stokes physical conservation laws into neural network loss gradients.
Introduction: Solving high-dimensional non-linear partial differential equations requires extensive meshing in classical finite element solvers.
Methodology: Multilayer perceptrons trained with automatic differentiation penalties enforcing physics residuals on Burgers and Navier-Stokes benchmarks.
Experiments: Evaluated across Reynolds number fluid dynamics simulations across spatiotemporal domains.
Results: PINN models attained L2 relative error of 1.4e-3 and Mean Squared Error of 2.1e-5.
Limitations: Training suffers from spectral bias where high-frequency flow harmonics fail to converge smoothly.
Future Work: Explore multi-scale Fourier feature embeddings to overcome spectral bias.
Conclusion: Deep physics-informed architectures bridge observational data with foundational scientific laws."""
    p10_chunks = [
        ChunkData("p10_c1", p10_text[:300], 36, "abstract", page_number=1, document_id="p10", filename="p10.pdf"),
        ChunkData("p10_c2", "Multilayer perceptrons trained with automatic differentiation penalties enforcing physics residuals on Burgers and Navier-Stokes benchmarks.", 18, "methodology", page_number=2, document_id="p10", filename="p10.pdf"),
        ChunkData("p10_c3", "PINN models attained L2 relative error of 1.4e-3 and Mean Squared Error of 2.1e-5.", 17, "results", page_number=4, document_id="p10", filename="p10.pdf"),
        ChunkData("p10_c4", "Training suffers from spectral bias where high-frequency flow harmonics fail to converge smoothly.", 15, "limitations", page_number=6, document_id="p10", filename="p10.pdf"),
    ]
    p10_qa = [
        GoldQAPair("What equations were integrated into the neural loss?", "method", ["Navier-Stokes", "Burgers", "differential equations"], ["p10_c2"], 2, "methodology"),
        GoldQAPair("What was the reported L2 relative error?", "result", ["1.4e-3"], ["p10_c3"], 4, "results"),
        GoldQAPair("What mathematical bias hinders high-frequency convergence?", "limitation", ["spectral bias", "harmonics"], ["p10_c4"], 6, "limitations"),
        GoldQAPair("What is the freezing point of ethanol?", "unsupported", ["not find enough evidence"], [], 0, "none", is_supported=False),
    ]
    papers.append(BenchmarkPaper(
        "p10_physics", "Physics-Informed Neural Networks for High-Dimensional Differential Equations",
        "Scientific AI", 2023, ["George Karniadakis", "Paris Perdikaris"], "Navier-Stokes", "Differential Equations",
        p10_text, [Page(1, p10_text)], p10_chunks,
        ["PINN", "Multilayer perceptron"], ["Burgers", "Navier-Stokes"], ["L2 relative error", "MSE"],
        ["1.4e-3", "2.1e-5"], ["Training suffers from spectral bias"],
        ["multi-scale Fourier feature embeddings"], p10_qa
    ))

    return papers
