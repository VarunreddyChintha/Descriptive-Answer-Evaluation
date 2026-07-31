# 📚 Descriptive Answer Evaluation System

An AI-powered web application that automatically evaluates descriptive answers using Natural Language Processing (NLP) and Transformer-based language models. The system compares student responses with predefined rubrics and provides accurate scoring along with meaningful feedback.

---

## 🚀 Features

- 📝 Evaluate descriptive answers automatically
- 📋 Rubric-based answer assessment
- 🤖 Semantic similarity using Transformer models
- 📊 Automated score prediction
- 💬 AI-generated feedback for student answers
- 🌐 Interactive Streamlit web interface
- ⚡ Fast and user-friendly evaluation process

---

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Frontend:** Streamlit
- **Machine Learning:** Scikit-learn
- **Deep Learning:** PyTorch
- **NLP Libraries:**
  - Transformers (Hugging Face)
  - Sentence Transformers
  - NLTK
  - spaCy
- **Data Processing:** Pandas
  - NumPy
- **Visualization:** Plotly, Matplotlib

---

## 📂 Project Structure

```
Descriptive-Answer-Evaluation/
│
├── app_v2.py                 # Streamlit application
├── requirements.txt          # Project dependencies
├── README.md
├── NLP_Project_Colab.ipynb   # Development notebook

```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/VarunreddyChintha/Descriptive-Answer-Evaluation.git
cd Descriptive-Answer-Evaluation
```

### Create a virtual environment (Recommended)

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run app_v2.py
```

The application will open in your browser at:

```
http://localhost:8501
```

---

## 📖 How It Works

1. Enter the question.
2. Provide the evaluation rubric.
3. Enter the student's descriptive answer.
4. The system analyzes the answer using NLP techniques and transformer embeddings.
5. The application:
   - Computes semantic similarity
   - Evaluates rubric coverage
   - Predicts the score
   - Generates constructive feedback

---

## 📦 Requirements

Major dependencies include:

- Streamlit
- PyTorch
- Transformers
- Sentence Transformers
- Scikit-learn
- Pandas
- NumPy
- NLTK
- spaCy
- Plotly

Install them using:

```bash
pip install -r requirements.txt
```

---

## 🎯 Future Improvements

- Support multiple subjects
- OCR support for handwritten answer sheets
- PDF answer evaluation
- Multi-language answer evaluation
- Explainable AI for score justification
- Dashboard for teachers and students
- Authentication and user management

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes.
4. Push to your branch.
5. Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Varun Reddy**

M.Tech, Computer Science and Engineering  
International Institute of Information Technology, Bangalore (IIIT Bangalore)

GitHub: https://github.com/VarunreddyChintha

---

⭐ If you found this project useful, consider giving it a ⭐ on GitHub.