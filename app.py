import streamlit as st
from utils import read_pdf, read_txt
from summarizer import generate_summary
from qa_engine import ask_question
from challenge_me import generate_questions, evaluate_answer

# 🧠 Page setup
st.set_page_config(page_title="Smart Assistant", layout="wide")
st.title("📄 Smart Assistant for Research Summarization")

# 📤 Upload PDF or TXT file
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    # 📄 Extract text
    if uploaded_file.name.endswith(".pdf"):
        content = read_pdf(uploaded_file)
    else:
        content = read_txt(uploaded_file)

    # 🧠 Auto Summary
    st.subheader("🧠 Auto Summary (≤ 150 words)")
    with st.spinner("Generating summary..."):
        summary = generate_summary(content)
    st.success("Summary Ready!")
    st.write(summary)

    # 📖 Expand to show full document text
    with st.expander("📖 View Full Document Text"):
        st.write(content[:2000])  # preview first 2000 characters

    # ❓ Ask Anything (Q&A)
    st.subheader("❓ Ask Anything About This Document")
    user_question = st.text_input("Enter your question:")

    if user_question:
        with st.spinner("Thinking..."):
            answer, score = ask_question(user_question, content)
        st.success(f"✅ Answer (Confidence: {score:.2f})")
        st.write(answer)

    # 🧠 Challenge Me Mode
    st.subheader("🧠 Challenge Me: Answer Logic Questions from This Document")

    # Store generated questions
    if "challenge_questions" not in st.session_state:
        st.session_state.challenge_questions = []
        st.session_state.challenge_references = []

    # Button to trigger question generation
    if st.button("🪄 Generate Questions"):
        st.session_state.challenge_questions, st.session_state.challenge_references = generate_questions(content)

    # Show questions
    if st.session_state.challenge_questions:
        st.write("Answer the following questions based on the uploaded document:")

        for i, (q, ref) in enumerate(zip(st.session_state.challenge_questions, st.session_state.challenge_references)):
            st.markdown(f"**Q{i+1}:** {q}")
            user_input = st.text_input(f"Your answer to Q{i+1}", key=f"user_ans_{i}")

            if user_input:
                is_correct, justification = evaluate_answer(user_input, ref)
                if is_correct:
                    st.success(justification)
                else:
                    st.error(justification)

else:
    st.info("📁 Please upload a PDF or TXT file to begin.")
