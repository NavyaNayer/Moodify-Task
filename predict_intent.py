import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from collections import Counter

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset and get correct label names
dataset = load_dataset("clinc_oos", "plus")
label_names = dataset["train"].features["intent"].names  # Ensure correct order

# Debugging check
print(f"Total labels: {len(label_names)}")  # Should print 151
print("Sample labels:", label_names[:10])  # Print first 10 labels

# Load the trained model
num_labels = len(label_names)  # Should be 151
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
model.load_state_dict(torch.load("intent_classifier.pth", map_location=device))
model.to(device)
model.eval()

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def predict_intent(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]

    if predicted_class >= len(label_names):  # Prevent out-of-range errors
        print(f"Warning: Predicted class {predicted_class} is out of range!")
        return predicted_class, "Unknown Label"

    return predicted_class, label_names[predicted_class]

# Example usage
sentence = "I need to attend a meeting but so tired but important"
predicted_intent, predicted_label_name = predict_intent(sentence)
print(f"Predicted intent for '{sentence}': {predicted_intent} ({predicted_label_name})")

# # Fix: Count labels correctly from dataset["train"]
# label_counts = Counter([label_names[label] for label in dataset["train"]["intent"]])  
# print("Label distribution:", label_counts)  # Print top 10 most common labels
