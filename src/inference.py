from transformers import pipeline
from huggingface_hub import login

login(token=("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"))

text = "The film's storyline is well-structured, and the acting is competent, but it doesn't stand out in any remarkable way. Overall, it's neither exceptionally good nor particularly bad—it just is."
classifier = pipeline("sentiment-analysis", model="Dolmer/my_awesome_model")
result = classifier(text)
print(result)
