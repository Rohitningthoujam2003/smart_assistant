from transformers import pipeline

# Load T5 model for question generation
question_generator = pipeline(
    "text2text-generation",
    model="valhalla/t5-base-qg-hl",
    framework="pt"  # PyTorch only to avoid TensorFlow conflicts
)

def generate_questions(context, num_questions=3):
    # Naively split sentences (no nltk needed)
    rough_sentences = [s.strip() for s in context.split('.') if len(s.strip()) > 20]

    questions = []
    references = []

    for sentence in rough_sentences:
        if len(questions) >= num_questions:
            break
        input_text = f"generate question: {sentence} </s>"
        result = question_generator(input_text, max_length=64, do_sample=False)
        question = result[0]['generated_text']
        questions.append(question)
        references.append(sentence)

    return questions, references

def evaluate_answer(user_answer, reference_answer):
    user_answer = user_answer.strip().lower()
    reference_answer = reference_answer.strip().lower()

    if user_answer in reference_answer:
        return True, f"✅ Correct! This is supported by: \"{reference_answer}\""
    else:
        return False, f"❌ Incorrect. Refer to this part of the document: \"{reference_answer}\""
