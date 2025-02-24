from transformers import AutoTokenizer, AutoModel
import torch

model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


text = ["Hello, world!", "How are you?"]
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # Mean pooling

print(embeddings)