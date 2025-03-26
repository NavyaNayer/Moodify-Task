<<<<<<< HEAD
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("thethinkmachine/Maxwell-Task-Complexity-Scorer-v0.2")
model = AutoModelForSequenceClassification.from_pretrained("thethinkmachine/Maxwell-Task-Complexity-Scorer-v0.2")

# Example task
task_description = "find a new theory"

# Tokenize the input
inputs = tokenizer(task_description, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    complexity_score = torch.sigmoid(outputs.logits).item()

print(f"Task Complexity Score: {complexity_score:.4f}")
=======
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("thethinkmachine/Maxwell-Task-Complexity-Scorer-v0.2")
model = AutoModelForSequenceClassification.from_pretrained("thethinkmachine/Maxwell-Task-Complexity-Scorer-v0.2")

# Example task
task_description = "find a new theory"

# Tokenize the input
inputs = tokenizer(task_description, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    complexity_score = torch.sigmoid(outputs.logits).item()

print(f"Task Complexity Score: {complexity_score:.4f}")
>>>>>>> b1313c5d084e410cadf261f2fafd8929cb149a4f
