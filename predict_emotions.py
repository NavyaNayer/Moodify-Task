import torch
from transformers import BertTokenizer, DistilBertForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model and tokenizer
try:
    model = DistilBertForSequenceClassification.from_pretrained("./saved_model")
    tokenizer = BertTokenizer.from_pretrained("./saved_model")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit()

model.to(device)
model.eval()

# Define the sentences
sentences = [
    "I am so happy today!",
    "This is the worst day ever.",
    "I feel so loved and appreciated.",
    "I am really angry right now.",
    "I am so done cant take this anymore",
    "i have to finish this report by tomorrow but so tired",
    "let's do it",
    "i have got this,, yayyyy",
    "energetic",
    "worst tired lazy",
    "I am feeling very sad and lonely."
]

# Define the label names
label_names = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]

def predict_emotion(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items() if key != "token_type_ids"}

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]

    return predicted_class, label_names[predicted_class]

# Predict emotions for the sentences
for sentence in sentences:
    predicted_emotion, predicted_label_name = predict_emotion(sentence)
    print(f"Predicted emotion for '{sentence}': {predicted_emotion} ({predicted_label_name})")
