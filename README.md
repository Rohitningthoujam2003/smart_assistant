 🧠 Smart Assistant for Research Summarization

An AI-powered assistant that summarizes uploaded research documents, answers questions, and quizzes users with logic-based comprehension tasks.

---

## Setup Instructions

1. Clone the Repository
```bash
git clone https://github.com/Rohitningthoujam2003/smart_assistant.git
cd smar_assistant
2. Create and Activate Virtual Environment
python -m venv smartenv
smartenv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
4. Run the App
streamlit run app.py

##Architecture / Reasoning Flow
Upload File: Supports PDF or TXT using PyPDF2 or plain read.

Summarization: Uses facebook/bart-large-cnn via Hugging Face.

Q&A: Uses distilbert-squad for question answering.

Challenge Me:

Generates logic questions via valhalla/t5-base-qg-hl

Compares user answer with reference using similarity score.

Displays document-based justification.
