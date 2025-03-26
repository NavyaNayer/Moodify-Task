import torch
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertForSequenceClassification
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the intent classifier model and tokenizer
num_intent_labels = 151  # Set the correct number of labels for the intent classifier
intent_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_intent_labels)
intent_model.load_state_dict(torch.load("intent_classifier.pth"))
intent_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
intent_model.to(device)
intent_model.eval()

# Load the emotions model and tokenizer
emotions_model = DistilBertForSequenceClassification.from_pretrained("./saved_model")
emotions_tokenizer = BertTokenizer.from_pretrained("./saved_model")
emotions_model.to(device)
emotions_model.eval()

# Define the label names for emotions
emotion_label_names = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]

def predict_intent(sentence):
    inputs = intent_tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = intent_model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]

    return predicted_class

def predict_emotion(sentence):
    inputs = emotions_tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items() if key != "token_type_ids"}

    with torch.no_grad():
        outputs = emotions_model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]

    return predicted_class, emotion_label_names[predicted_class]

def calculate_priority_score(intent, emotion, time_remaining):
    # Example priority score calculation
    intent_weight = 0.4
    emotion_weight = 0.3
    time_weight = 0.3

    # Normalize time_remaining to a score between 0 and 1
    time_score = max(0, min(1, 1 - (time_remaining.total_seconds() / (24 * 3600))))

    # Calculate priority score
    priority_score = (intent * intent_weight) + (emotion * emotion_weight) + (time_score * time_weight)
    return priority_score

def prioritize_task(task_description, due_date_time, predicted_emotion, predicted_label_name):
    predicted_intent = predict_intent(task_description)
    
    # Calculate time remaining until the due date and time
    due_date_time = datetime.strptime(due_date_time, "%Y-%m-%d %H:%M:%S")
    time_remaining = due_date_time - datetime.now()
    
    priority_score = calculate_priority_score(predicted_intent, predicted_emotion, time_remaining)
    
    return {
        "description": task_description,
        "due_date_time": due_date_time,
        "time_remaining": time_remaining,
        "predicted_intent": predicted_intent,
        "predicted_emotion": predicted_emotion,
        "predicted_label_name": predicted_label_name,
        "priority_score": priority_score
    }

# Example tasks
tasks = [
    {"description": "Finish the report by tomorrow.", "due_date_time": "2025-03-02 09:00:00"},
    {"description": "meeting", "due_date_time": "2025-03-02 12:00:00"},
    {"description": "listen to music.", "due_date_time": "2025-03-02 15:00:00"},
    {"description": "daily linkedin queens game.", "due_date_time": "2025-03-02 18:00:00"},
    {"description": "prepare ppt", "due_date_time": "2025-03-02 21:00:00"}
]

# Overall emotion sentence
emotion_sentence = "I am feeling very tired and stressed now"
predicted_emotion, predicted_label_name = predict_emotion(emotion_sentence)

# Prioritize tasks
prioritized_tasks = []
for task in tasks:
    prioritized_tasks.append(prioritize_task(task["description"], task["due_date_time"], predicted_emotion, predicted_label_name))

# Reorder tasks based on priority score (descending order)
prioritized_tasks.sort(key=lambda x: x["priority_score"], reverse=True)

# Print prioritized tasks
for task in prioritized_tasks:
    print(f"Task Description: '{task['description']}'")
    print(f"Due Date and Time: {task['due_date_time']}")
    print(f"Time Remaining: {task['time_remaining']}")
    print(f"Predicted Intent: {task['predicted_intent']}")
    print(f"Predicted Emotion: {task['predicted_emotion']} ({task['predicted_label_name']})")
    print(f"Priority Score: {task['priority_score']:.4f}")
    print()
