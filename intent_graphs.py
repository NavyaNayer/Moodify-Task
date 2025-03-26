import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.preprocessing import label_binarize
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = load_dataset("clinc_oos", "plus")
label_names = dataset["train"].features["intent"].names  # Ensure correct order

# Load model
num_labels = len(label_names)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
model.load_state_dict(torch.load("intent_classifier.pth", map_location=device))
model.to(device)
model.eval()

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Prepare data
true_labels = []
pred_labels = []
all_probs = []

for example in dataset["test"]:
    sentence = example["text"]
    true_label = example["intent"]

    # Tokenize
    inputs = tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        predicted_class = np.argmax(probs)

    # Store results
    true_labels.append(true_label)
    pred_labels.append(predicted_class)
    all_probs.append(probs)

# Convert to numpy arrays
true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)
all_probs = np.array(all_probs)

# Compute confusion matrix
conf_matrix = confusion_matrix(true_labels, pred_labels)

# Plot confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=False, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Intent Classification")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

print("Confusion matrix saved as confusion_matrix.png")

# --- Multi-Class Precision-Recall Curve ---
# Binarize true labels for multi-class PR calculation
true_labels_bin = label_binarize(true_labels, classes=np.arange(num_labels))

# Plot Precision-Recall Curve for multiple classes
plt.figure(figsize=(10, 8))
for i in range(num_labels):
    precision, recall, _ = precision_recall_curve(true_labels_bin[:, i], all_probs[:, i])
    plt.plot(recall, precision, lw=1, alpha=0.7, label=f"Class {i}: {label_names[i]}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Multi-Class Precision-Recall Curve")
plt.legend(loc="best", fontsize=6, ncol=2, frameon=True)
plt.grid(True)
plt.savefig("precision_recall_curve.png", dpi=300, bbox_inches="tight")
plt.close()

print("Precision-Recall curve saved as precision_recall_curve.png")
