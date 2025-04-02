import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def download_base_models():
    # Create models directory
    os.makedirs("pretrained_models", exist_ok=True)
    
    print("Downloading BERT base model...")
    # Download and save BERT base model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    
    # Save models locally
    tokenizer.save_pretrained("pretrained_models/bert-base-uncased")
    model.save_pretrained("pretrained_models/bert-base-uncased")
    print("Base models downloaded successfully!")

if __name__ == "__main__":
    download_base_models()
