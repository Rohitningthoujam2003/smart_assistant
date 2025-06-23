from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

# Load embedding model (for semantic similarity)
embedding_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Load QA model with PyTorch only (avoids TensorFlow issues)
qa_model = pipeline(
    "question-answering",
    model="distilbert-base-uncased-distilled-squad",
    framework="pt"  # Force PyTorch-only to avoid inspect/TF issues
)

def embed(text):
    """Generate a vector embedding for a given text chunk."""
    inputs = embedding_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

def split_text(text, max_words=150):
    """Split the input text into manageable chunks."""
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def ask_question(question, context):
    """Answer the user's question by selecting the most relevant chunk."""
    chunks = split_text(context)
    chunk_embeddings = [embed(chunk) for chunk in chunks]
    question_embedding = embed(question)

    # Compute similarities between the question and each chunk
    similarities = [cosine_similarity(question_embedding, chunk_emb)[0][0] for chunk_emb in chunk_embeddings]
    best_index = int(np.argmax(similarities))
    best_chunk = chunks[best_index]

    # Run QA on the best chunk
    result = qa_model(question=question, context=best_chunk)
    return result['answer'], result['score']
