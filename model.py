import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("go_emotions")

# Extract text and labels
texts = dataset["train"]["text"][:20000]  # Increased dataset size
labels = dataset["train"]["labels"][:20000]  # Increased dataset size

# Convert labels to categorical
def fix_labels(labels):
    labels = [max(label) if label else 0 for label in labels]  # Convert multi-label to single-label
    return torch.tensor(labels, dtype=torch.long)

labels = fix_labels(labels)

# Split dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize text
def tokenize(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

train_encodings = tokenize(train_texts)
val_encodings = tokenize(val_texts)
train_encodings = {key: val.to(device) for key, val in train_encodings.items()}
val_encodings = {key: val.to(device) for key, val in val_encodings.items()}

class EmotionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

train_dataset = EmotionDataset(train_encodings, train_labels)
val_dataset = EmotionDataset(val_encodings, val_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

class BertGRUClassifier(nn.Module):
    def __init__(self, bert_model="bert-base-uncased", hidden_dim=128, num_classes=28):
        super(BertGRUClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.gru = nn.GRU(self.bert.config.hidden_size, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)  # Added dropout layer
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        gru_output, _ = self.gru(bert_output.last_hidden_state)
        output = self.fc(self.dropout(gru_output[:, -1, :]))  # Apply dropout
        return output

model = BertGRUClassifier()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)  # Added learning rate scheduler

def evaluate_model(model, data_loader):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    return acc, f1

def train_model(model, train_loader, val_loader, epochs=10):  # Increased number of epochs
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()  # Step the scheduler

        train_acc, train_f1 = evaluate_model(model, train_loader)
        val_acc, val_f1 = evaluate_model(model, val_loader)
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        # Save the model after each epoch
        torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")

train_model(model, train_loader, val_loader)

# Assuming you have a test dataset
test_texts = dataset["test"]["text"]
test_labels = fix_labels(dataset["test"]["labels"])
test_encodings = tokenize(test_texts)
test_encodings = {key: val.to(device) for key, val in test_encodings.items()}
test_dataset = EmotionDataset(test_encodings, test_labels)
test_loader = DataLoader(test_dataset, batch_size=16)

test_acc, test_f1 = evaluate_model(model, test_loader)
print(f"Test Accuracy: {test_acc:.4f}, Test F1 Score: {test_f1:.4f}")
