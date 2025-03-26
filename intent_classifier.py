import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load CLINC-OOS Dataset (Correct Config)
dataset = load_dataset("clinc_oos", "plus")

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Preprocess Dataset
class IntentDataset(Dataset):
    def __init__(self, dataset_split):
        self.texts = dataset_split["text"]
        self.labels = dataset_split["intent"]
        self.label_map = {label: i for i, label in enumerate(set(self.labels))}  # Create label mapping

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = tokenizer(self.texts[idx], padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        label = self.labels[idx]
        if label not in self.label_map:
            raise ValueError(f"Unexpected label {label} found in dataset")  # Debugging step
        return {key: val.squeeze(0) for key, val in inputs.items()}, torch.tensor(self.label_map[label])

# Create Dataloaders
batch_size = 16
train_dataset = IntentDataset(dataset["train"])
test_dataset = IntentDataset(dataset["test"])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Load Pretrained BERT Model
num_labels = len(set(dataset["train"]["intent"]))
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels).to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# Training Loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training"):
        inputs, labels = batch
        inputs = {key: val.to(device) for key, val in inputs.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(**inputs).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    train_accuracy = correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

# Evaluation on Test Set
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        inputs, labels = batch
        inputs = {key: val.to(device) for key, val in inputs.items()}
        labels = labels.to(device)

        outputs = model(**inputs).logits
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute Metrics
accuracy = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted")

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

# Save Model
torch.save(model.state_dict(), "intent_classifier.pth")
