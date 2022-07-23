from transformers import pipeline

classifier = pipeline("sentiment-analysis")

res = classifier("I hate you")
print(res)

# bert-base-german-cased