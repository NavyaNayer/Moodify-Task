import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load GoEmotions dataset
dataset = load_dataset("go_emotions", split="train")
dataset = dataset.map(lambda x: {"label": x["labels"][0]})  # Convert multi-label to single-label

labels = list(set(dataset["label"]))  # Unique labels
num_labels = len(labels)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class MoodDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = tokenizer(self.texts[idx], return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        return {key: val.squeeze(0) for key, val in inputs.items()}, torch.tensor(labels.index(self.labels[idx]))

dataset = MoodDataset(dataset["text"], dataset["label"])
train_size = int(0.8 * len(dataset))
train_set, test_set = random_split(dataset, [train_size, len(dataset) - train_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels).to(device)
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    epoch_loss, correct, total = 0, 0, 0
    preds, labels_list = [], []

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training"):
        optimizer.zero_grad()
        inputs = {key: val.to(device) for key, val in batch[0].items()}
        labels = batch[1].to(device)

        outputs = model(**inputs).logits
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

        preds.extend(outputs.argmax(dim=1).cpu().numpy())
        labels_list.extend(labels.cpu().numpy())

    train_acc = accuracy_score(labels_list, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels_list, preds, average="weighted")

    print(f"Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# **Evaluate on Test Set**
model.eval()
test_preds, test_labels = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating on Test Set"):
        inputs = {key: val.to(device) for key, val in batch[0].items()}
        labels = batch[1].to(device)

        outputs = model(**inputs).logits
        test_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

test_acc = accuracy_score(test_labels, test_preds)
precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average="weighted")

print(f"Test Accuracy: {test_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

# Save model
model.save_pretrained("mood_classifier")
