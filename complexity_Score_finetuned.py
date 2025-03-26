<<<<<<< HEAD
import torch
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.utils.data import DataLoader
from transformers import AdamW
from sklearn.metrics import r2_score, f1_score, mean_absolute_error

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Load DEITA-Complexity dataset
dataset = load_dataset("hkust-nlp/deita-complexity-scorer-data")
val_data = dataset["validation"]

# Initialize tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Preprocessing function
def preprocess_function(examples):
    return tokenizer(examples["input"], truncation=True, padding="max_length", max_length=128)

# Tokenize validation dataset
val_encodings = val_data.map(preprocess_function, batched=True)

# Inspect the structure of val_encodings
print("Validation Encodings Structure:")
print(val_encodings)

# Convert dataset to PyTorch format
class ComplexityDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        # Create a dictionary for the inputs
        item = {
            "input_ids": torch.tensor(self.encodings['input_ids'][idx]),
            "attention_mask": torch.tensor(self.encodings['attention_mask'][idx]),
            # Convert target to float if it's a string
            "labels": torch.tensor(float(self.encodings['target'][idx]), dtype=torch.float)  # Ensure 'target' is numeric
        }
        return item

val_dataset = ComplexityDataset(val_encodings)

# Load pre-trained DistilBERT model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1)

# Freeze first 4 transformer layers
for layer in model.distilbert.transformer.layer[:4]:
    for param in layer.parameters():
        param.requires_grad = False

# Define optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# DataLoader for batching
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Evaluation function
def evaluate_model(model, val_loader):
    model.eval()
    val_loss = 0.0
    total_mae = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            batch = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**batch)
            loss = torch.nn.functional.mse_loss(outputs.logits.squeeze(), batch["labels"])

            val_loss += loss.item()
            total_mae += torch.nn.functional.l1_loss(outputs.logits.squeeze(), batch["labels"], reduction="sum").item()

            all_predictions.extend(outputs.logits.squeeze().cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    avg_val_mae = total_mae / len(val_loader.dataset)

    # Calculate additional metrics
    r2 = r2_score(all_labels, all_predictions)
    f1 = f1_score(np.round(all_labels), np.round(all_predictions), average='weighted')

    return avg_val_loss, avg_val_mae, r2, f1, all_predictions, all_labels

# Evaluate the model
val_loss, val_mae, r2, f1, predictions, labels = evaluate_model(model, val_loader)

print(f"Validation Loss = {val_loss:.4f}, Validation MAE = {val_mae:.4f}, R² Score = {r2:.4f}, F1 Score = {f1:.4f}")

# Testing the model (inference on the validation set)
def test_model(model, val_loader):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Testing", leave=False):
            batch = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**batch)

            all_predictions.extend(outputs.logits.squeeze().cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    return np.array(all_predictions), np.array(all_labels)

# Get predictions and labels from the test function
test_predictions, test_labels = test_model(model, val_loader)

# You can also calculate the evaluation metrics on the test predictions
test_r2 = r2_score(test_labels, test_predictions)
test_f1 = f1_score(np.round(test_labels), np.round(test_predictions), average='weighted')

print(f"Test R² Score = {test_r2:.4f}, Test F1 Score = {test_f1:.4f}")

# Save the fine-tuned model
model.save_pretrained("fine_tuned_deita_model")
tokenizer.save_pretrained("fine_tuned_deita_model")

print("✅ Evaluation and testing complete! Model saved at 'fine_tuned_deita_model'.")
=======
import torch
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.utils.data import DataLoader
from transformers import AdamW
from sklearn.metrics import r2_score, f1_score, mean_absolute_error

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Load DEITA-Complexity dataset
dataset = load_dataset("hkust-nlp/deita-complexity-scorer-data")
val_data = dataset["validation"]

# Initialize tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Preprocessing function
def preprocess_function(examples):
    return tokenizer(examples["input"], truncation=True, padding="max_length", max_length=128)

# Tokenize validation dataset
val_encodings = val_data.map(preprocess_function, batched=True)

# Inspect the structure of val_encodings
print("Validation Encodings Structure:")
print(val_encodings)

# Convert dataset to PyTorch format
class ComplexityDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        # Create a dictionary for the inputs
        item = {
            "input_ids": torch.tensor(self.encodings['input_ids'][idx]),
            "attention_mask": torch.tensor(self.encodings['attention_mask'][idx]),
            # Convert target to float if it's a string
            "labels": torch.tensor(float(self.encodings['target'][idx]), dtype=torch.float)  # Ensure 'target' is numeric
        }
        return item

val_dataset = ComplexityDataset(val_encodings)

# Load pre-trained DistilBERT model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1)

# Freeze first 4 transformer layers
for layer in model.distilbert.transformer.layer[:4]:
    for param in layer.parameters():
        param.requires_grad = False

# Define optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# DataLoader for batching
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Evaluation function
def evaluate_model(model, val_loader):
    model.eval()
    val_loss = 0.0
    total_mae = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            batch = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**batch)
            loss = torch.nn.functional.mse_loss(outputs.logits.squeeze(), batch["labels"])

            val_loss += loss.item()
            total_mae += torch.nn.functional.l1_loss(outputs.logits.squeeze(), batch["labels"], reduction="sum").item()

            all_predictions.extend(outputs.logits.squeeze().cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    avg_val_mae = total_mae / len(val_loader.dataset)

    # Calculate additional metrics
    r2 = r2_score(all_labels, all_predictions)
    f1 = f1_score(np.round(all_labels), np.round(all_predictions), average='weighted')

    return avg_val_loss, avg_val_mae, r2, f1, all_predictions, all_labels

# Evaluate the model
val_loss, val_mae, r2, f1, predictions, labels = evaluate_model(model, val_loader)

print(f"Validation Loss = {val_loss:.4f}, Validation MAE = {val_mae:.4f}, R² Score = {r2:.4f}, F1 Score = {f1:.4f}")

# Testing the model (inference on the validation set)
def test_model(model, val_loader):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Testing", leave=False):
            batch = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**batch)

            all_predictions.extend(outputs.logits.squeeze().cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    return np.array(all_predictions), np.array(all_labels)

# Get predictions and labels from the test function
test_predictions, test_labels = test_model(model, val_loader)

# You can also calculate the evaluation metrics on the test predictions
test_r2 = r2_score(test_labels, test_predictions)
test_f1 = f1_score(np.round(test_labels), np.round(test_predictions), average='weighted')

print(f"Test R² Score = {test_r2:.4f}, Test F1 Score = {test_f1:.4f}")

# Save the fine-tuned model
model.save_pretrained("fine_tuned_deita_model")
tokenizer.save_pretrained("fine_tuned_deita_model")

print("✅ Evaluation and testing complete! Model saved at 'fine_tuned_deita_model'.")
>>>>>>> b1313c5d084e410cadf261f2fafd8929cb149a4f
