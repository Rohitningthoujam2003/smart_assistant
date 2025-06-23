from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# Load tokenizer and model explicitly (no pipeline)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

def generate_summary(text, max_words=150):
    # Shorten text if too long for model
    if len(text) > 1024:
        text = text[:1024]

    # Tokenize input text
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)

    # Generate summary (no sampling)
    summary_ids = model.generate(inputs, max_length=max_words, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode and return summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
