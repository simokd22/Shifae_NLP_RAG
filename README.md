# Shifar_NLP_RAG
from pathlib import Path

readme = r"""# SHIFAA Arabic Mental Health RAG

A research project that explores **Retrieval-Augmented Generation (RAG)** for Arabic mental-health question answering. The system uses the SHIFAA Arabic Mental Health Consultations dataset to retrieve relevant expert consultation responses and provide a context-grounded answer in Modern Standard Arabic.

> **Important:** This is an academic prototype, not a medical device or a substitute for a licensed mental-health professional. Generated answers may be incomplete or incorrect. Do not use the system for diagnosis, treatment decisions, or emergencies.

## Project goals

- Prepare and normalize Arabic mental-health consultation text.
- Build a semantic knowledge index over expert responses.
- Retrieve relevant passages for a user's question.
- Rerank candidate passages to improve relevance.
- Generate an Arabic response grounded in retrieved context, with an extractive fallback where available.
- Evaluate retrieval and generation separately and compare retrieval-based answering with conventional diagnosis-classification baselines.

## Dataset

The project uses the **SHIFAA Arabic Mental Health Consultations** dataset, reported to contain 35,648 consultations. Its main fields include patient questions, expert answers, and hierarchical diagnoses.

Dataset source:
- Kaggle: <https://www.kaggle.com/datasets/ahmedseleem/shifaa-arabic-mental-health-consultations>

The dataset is not included in this repository/notebook. Download it from the source and follow its licence and usage conditions. Avoid publishing personal or sensitive information.

## How the system works

1. **Load and prepare data** — combine the question title and question, remove unused columns, and filter missing or very short answers.
2. **Normalize Arabic text** — use PyArabic to normalize characters and remove tashkeel, tatweel, and ligatures; clean URLs, emails, phone numbers, whitespace, and punctuation.
3. **Split the data** — create a stratified train/test split based on diagnosis categories.
4. **Chunk expert responses** — use a sliding window with a target size of 600 characters and 100-character overlap; very short chunks are filtered out.
5. **Create embeddings** — encode chunks with a multilingual Sentence-Transformers model. The notebook considers `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, `intfloat/multilingual-e5-small`, and `Alibaba-NLP/gte-multilingual-base`.
6. **Build a FAISS index** — use vector similarity search to find candidate passages.
7. **Retrieve and rerank** — retrieve semantically similar chunks and rerank them with a Cross-Encoder, including relevance signals associated with medical context.
8. **Generate an answer** — pass selected context and the user's question to an Arabic-capable language model. The prompt requests a formal Arabic response grounded in the retrieved material; an extractive mode is available as a fallback.
9. **Evaluate** — measure retrieval quality and generated-answer similarity, overlap, and safety-related checks.

## Main technologies

- Python
- pandas and NumPy
- PyArabic
- NLTK
- Sentence-Transformers
- FAISS
- Hugging Face Transformers
- PyTorch
- scikit-learn
- ROUGE / BERTScore-related evaluation tools
- Matplotlib and Seaborn for analysis and visualization
- Kaggle notebooks / Google Colab for data and GPU execution

## Running the notebook

The supplied project is a Jupyter/Colab notebook and contains environment-specific Kaggle and Google Colab imports. The easiest starting point is to open it in **Google Colab** or a compatible notebook environment with GPU access.

### 1. Open the notebook

Upload `RAG_SHIFAE_LV` to Google Colab and, if needed, rename it to `RAG_SHIFAE_LV.ipynb`.

### 2. Install dependencies

The notebook begins by installing packages. A typical setup includes:

```bash
pip install arabic-reshaper pyarabic gensim python-bidi wordcloud
pip install sentence-transformers faiss-cpu bert-score rouge-score
pip install bitsandbytes accelerate evaluate
```

It also imports packages such as `torch`, `transformers`, `kagglehub`, `nltk`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `seaborn`. Install any missing package in your environment.

### 3. Configure dataset access

The notebook downloads the dataset using `kagglehub` and also contains a Kaggle-specific input path:

```python
path = "/kaggle/input/shifaa-arabic-mental-health-consultations"
```

If running in Google Colab, ensure Kaggle access is configured or update the path to the directory where the CSV files were downloaded. The notebook also imports Google Drive utilities, so remove or adapt those cells if you are not using Colab.

### 4. Run cells in order

Run the notebook from top to bottom. The workflow covers data exploration, preprocessing, stratified splitting, chunking, embeddings, FAISS indexing, retrieval/reranking, answer generation, and evaluation.

**Hardware note:** Loading a generative model and embedding a large collection can require substantial RAM, GPU memory, and disk space. A free GPU runtime may not be sufficient for every model. Model downloads also require an internet connection.

## Evaluation

The project evaluates two parts of the pipeline:

### Retrieval

Metrics used include:

- Recall@k
- Precision@k
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG@k)
- Average and maximum cosine similarity

The accompanying research paper reports retrieval results including **Recall@5 = 0.881**, **NDCG@5 = 0.894**, and **MRR = 0.889** on its reported evaluation set. Results may vary depending on the notebook version, split, index, thresholds, and sample selection.

### Generation

Metrics/checks include:

- Semantic similarity between generated and reference answers
- Arabic BERTScore
- Context overlap / faithfulness-related measures
- Response length ratio
- Arabic-language coverage and safety-related checks

The paper reports a mean semantic similarity of approximately **0.955** and a BERTScore F1 of **0.723** on a small generation sample. The reported safety score is **77.8%**. These measurements are preliminary and should not be interpreted as clinical validation or proof that hallucinations are absent. The generation sample is small, and professional human evaluation is still needed.

### Baseline comparison

The project also investigates conventional diagnosis classification using TF-IDF, Word2Vec, FastText, and classifiers such as Logistic Regression, SVM, Naive Bayes, and Random Forest, plus an AraBERT-based hierarchical classification attempt. These approaches predict labels, whereas the RAG pipeline retrieves consultation context and generates a response; their metrics are therefore not directly interchangeable.

## Limitations and future work

- Generation evaluation is based on a small sample and needs broader testing.
- Arabic dialects are not comprehensively handled; the target is primarily Modern Standard Arabic.
- Clinical evaluation by qualified Arabic-speaking mental-health professionals has not yet been established.
- GPU and memory constraints limit the number and size of models that can be tested.
- Similarity and overlap metrics alone cannot establish medical correctness, safety, or faithfulness.
- Future work could add dialect detection, evaluate stronger Arabic-capable models, improve safety and escalation behavior, and conduct systematic expert evaluation before considering any real-world deployment.

## Research materials

The project is accompanied by a report/paper titled **“Evidence-Based Arabic Mental Health Question Answering with Retrieval-Augmented Generation on SHIFAA.”** Consult the report for the detailed methodology, experiments, and references.

## Authors

- **Bouba Ahmed**
- **Lkhalidi Mohamed**

Academic context: Master 2 — Intelligent Systems for Education, École Normale Supérieure de Meknès, Moulay Ismail University of Meknès, Morocco.

## References

1. Lewis et al., “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,” NeurIPS, 2020.
2. Gao et al., “Retrieval-Augmented Generation for Large Language Models: A Survey,” arXiv:2312.10997, 2023.
3. Antoun, Baly, and Hajj, “AraBERT: Transformer-based Model for Arabic Language Understanding,” OSACT, 2020.
4. Selem, “SHIFAA Arabic Mental Health Consultations Dataset,” Kaggle.

---

*This README describes the supplied notebook and accompanying paper. Verify the exact runtime configuration, metrics, and dataset-processing counts by rerunning the notebook before reproducing or publishing the results.*
"""

path = Path("/mnt/data/README_SHIFAA_RAG.md")
path.write_text(readme, encoding="utf-8")
print(f"Created: {path}\nSize: {path.stat().st_size} bytes")
