# 📚 Descriptive Answer Evaluation System

An AI-powered descriptive answer evaluation system that automatically grades free-text responses using a **rubric-based multi-metric NLP scoring framework**. Unlike traditional Automatic Short Answer Grading (ASAG) systems that rely on a fixed model answer or a single semantic similarity score, this system evaluates answers across multiple dimensions to provide **objective, explainable, and human-aligned assessment**.

---

## 🚀 Key Features

- 📋 Rubric-based descriptive answer evaluation
- 🧠 Multi-metric scoring instead of single similarity scoring
- 🤖 Semantic understanding using Sentence-BERT
- 🔍 Logical consistency verification using DeBERTa NLI
- 📝 Reasoning quality analysis using spaCy dependency parsing
- 📊 Rubric-wise coverage analysis
- 💬 Automated feedback generation
- 🌐 Interactive Streamlit interface

---

# Motivation

Traditional descriptive answer evaluation systems suffer from several limitations:

- Depend on manually written model answers
- Cannot distinguish explanation from keyword dumping
- Fail to detect contradictions
- Ignore logical reasoning and answer coherence

Our system addresses these limitations by introducing a **rubric-driven multi-metric evaluation framework** that scores answers across six independent quality dimensions, making grading more objective and interpretable.

---

# System Architecture

```
Question
      │
Rubric Points
      │
Student Answer
      │
────────────── NLP Pipeline ──────────────

Sentence-BERT
     │
     ├── Relevance
     ├── Coverage
     └── Coherence

DeBERTa NLI
     │
     ├── Consistency
     └── Internal Consistency

spaCy
     │
Reasoning Quality

──────────── Score Fusion ─────────────

Final Marks
Rubric Coverage
Feedback Generation
```

---

## 🛠️ Tech Stack

### Programming Language

- Python

### Framework

- Streamlit

### NLP & Deep Learning

- Sentence-BERT (all-MiniLM-L6-v2)
- DeBERTa-v3 NLI
- spaCy
- Hugging Face Transformers
- PyTorch

### Machine Learning

- Scikit-learn

### Data Processing

- Pandas
- NumPy

### Visualization

- Plotly
- Matplotlib

---

# Multi-Metric Evaluation Framework

Instead of assigning marks using a single similarity score, the system evaluates answers using **six complementary metrics**.

| Metric | Purpose | Technique |
|---------|----------|-----------|
| Relevance | Measures whether the answer is on-topic | Sentence-BERT |
| Coverage | Measures rubric point coverage | Sentence-BERT + Sentence Matching |
| Consistency | Detects factual agreement with rubric | DeBERTa NLI |
| Reasoning Quality | Measures explanatory depth | spaCy + Discourse Analysis |
| Coherence | Measures logical sentence flow | Sentence-BERT |
| Internal Consistency | Detects self-contradictions | DeBERTa NLI |

---

# Scoring Pipeline

1. Encode question, rubric points, and answer using Sentence-BERT.
2. Match every rubric point with the most relevant answer sentence.
3. Verify logical correctness using DeBERTa Natural Language Inference.
4. Analyze reasoning quality using discourse markers and dependency parsing.
5. Compute coherence between consecutive sentences.
6. Detect contradictions within the student's answer.
7. Fuse all six metrics using tuned weights.
8. Generate:
   - Final Score
   - Metric-wise Score Breakdown
   - Rubric Coverage Report
   - Automated Feedback

---

## 📈 Results

The proposed framework was evaluated against human annotations on a custom dataset containing **50 descriptive answers** across **five Computer Science topics**.

| Metric | Score |
|---------|--------|
| Pearson Correlation | **0.847** |
| Spearman Correlation | **0.813** |
| Quadratic Weighted Kappa | **0.764** |

Compared with traditional approaches:

| Method | Pearson | Spearman | QWK |
|---------|----------|-----------|------|
| TF-IDF | 0.758 | 0.740 | 0.424 |
| SBERT Only | 0.534 | 0.516 | 0.157 |
| **Our Framework** | **0.847** | **0.813** | **0.764** |

---

## 📂 Project Structure

```
Descriptive-Answer-Evaluation/

├── app.py
├── requirements.txt
├── README.md
├── NLP_Project_Colab.ipynb
├── models/
├── utils/
└── data/
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/VarunreddyChintha/Descriptive-Answer-Evaluation.git

cd Descriptive-Answer-Evaluation
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# Future Improvements

- Automatic rubric generation using LLMs
- Multi-language evaluation
- OCR-based handwritten answer evaluation
- PDF answer sheet assessment
- Explainable AI visualizations
- Teacher dashboard
- Student performance analytics

---

## 👨‍💻 Author

**Varun Reddy**

M.Tech, Computer Science and Engineering

International Institute of Information Technology Bangalore (IIIT Bangalore)

GitHub: https://github.com/VarunreddyChintha

---

⭐ If you found this project useful, consider giving it a Star.
