<<<<<<< HEAD
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# Load dataset
dataset = load_dataset("go_emotions")

# Print dataset columns
print("Dataset Columns Before Preprocessing:", dataset["train"].column_names)

# Ensure labels exist
if "labels" not in dataset["train"].column_names:
    raise KeyError("Column 'labels' is missing! Check dataset structure.")

# Load tokenizer
model_checkpoint = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Preprocessing function (Take only the first label for single-label classification)
def preprocess_data(batch):
    encoding = tokenizer(batch["text"], padding="max_length", truncation=True)
    
    # Take only the first label (for single-label classification)
    encoding["labels"] = batch["labels"][0] if batch["labels"] else 0  # Default to 0 if empty
    return encoding

# Tokenize dataset
encoded_dataset = dataset.map(preprocess_data, batched=False, remove_columns=["text"])

# Set format for PyTorch
encoded_dataset.set_format("torch")

# Load model for single-label classification (28 classes)
num_labels = 28  # Change based on dataset labels
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

# Training arguments
args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    logging_strategy="no",
    per_device_train_batch_size=32,  # Increase batch size
    per_device_eval_batch_size=32,  
    num_train_epochs=2,  # Reduce epochs
    weight_decay=0.01,
    load_best_model_at_end=True,
    fp16=True,  # Mixed precision for speedup
    gradient_accumulation_steps=2,  # Helps with large batch sizes
)


# Compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    # Convert logits to class predictions
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    
    return {"accuracy": accuracy, "f1": f1}

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    compute_metrics=compute_metrics
)

# Train model
trainer.train()
print("Training completed!")

# Save model and tokenizer
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")
print("Model and tokenizer saved!")

# ====== Evaluation on Test Set ======
print("\nEvaluating model on test set...")

# Get test dataset
test_dataset = encoded_dataset["test"]

# Make predictions
predictions = trainer.predict(test_dataset)
logits = predictions.predictions

# Convert logits to class predictions
y_pred = np.argmax(logits, axis=-1)
y_true = test_dataset["labels"].numpy()

# Compute accuracy and F1-score
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="weighted")

# Print evaluation results
print("\nEvaluation Results:")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")

# Print classification report
print("\nClassification Report:\n", classification_report(y_true, y_pred))

# Save test results
pd.DataFrame({"true_labels": y_true.tolist(), "predicted_labels": y_pred.tolist()}).to_csv("test_results.csv", index=False)
print("Test results saved to 'test_results.csv'!")
=======
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# Load dataset
dataset = load_dataset("go_emotions")

# Print dataset columns
print("Dataset Columns Before Preprocessing:", dataset["train"].column_names)

# Ensure labels exist
if "labels" not in dataset["train"].column_names:
    raise KeyError("Column 'labels' is missing! Check dataset structure.")

# Load tokenizer
model_checkpoint = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Preprocessing function (Take only the first label for single-label classification)
def preprocess_data(batch):
    encoding = tokenizer(batch["text"], padding="max_length", truncation=True)
    
    # Take only the first label (for single-label classification)
    encoding["labels"] = batch["labels"][0] if batch["labels"] else 0  # Default to 0 if empty
    return encoding

# Tokenize dataset
encoded_dataset = dataset.map(preprocess_data, batched=False, remove_columns=["text"])

# Set format for PyTorch
encoded_dataset.set_format("torch")

# Load model for single-label classification (28 classes)
num_labels = 28  # Change based on dataset labels
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

# Training arguments
args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    logging_strategy="no",
    per_device_train_batch_size=32,  # Increase batch size
    per_device_eval_batch_size=32,  
    num_train_epochs=2,  # Reduce epochs
    weight_decay=0.01,
    load_best_model_at_end=True,
    fp16=True,  # Mixed precision for speedup
    gradient_accumulation_steps=2,  # Helps with large batch sizes
)


# Compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    # Convert logits to class predictions
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    
    return {"accuracy": accuracy, "f1": f1}

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    compute_metrics=compute_metrics
)

# Train model
trainer.train()
print("Training completed!")

# Save model and tokenizer
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")
print("Model and tokenizer saved!")

# ====== Evaluation on Test Set ======
print("\nEvaluating model on test set...")

# Get test dataset
test_dataset = encoded_dataset["test"]

# Make predictions
predictions = trainer.predict(test_dataset)
logits = predictions.predictions

# Convert logits to class predictions
y_pred = np.argmax(logits, axis=-1)
y_true = test_dataset["labels"].numpy()

# Compute accuracy and F1-score
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="weighted")

# Print evaluation results
print("\nEvaluation Results:")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")

# Print classification report
print("\nClassification Report:\n", classification_report(y_true, y_pred))

# Save test results
pd.DataFrame({"true_labels": y_true.tolist(), "predicted_labels": y_pred.tolist()}).to_csv("test_results.csv", index=False)
print("Test results saved to 'test_results.csv'!")
>>>>>>> b1313c5d084e410cadf261f2fafd8929cb149a4f
