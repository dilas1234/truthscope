# text_verification.py

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)  # 2 classes 
(Misinformation/True)

# Example function to load the fine-tuned model
def load_model():
    model_path = 'path_to_your_model'  # replace with the actual model path
    model = BertForSequenceClassification.from_pretrained(model_path)
    return model

# Function to predict whether a text is misinformation
def verify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions.item()  # 0: true, 1: misinformation

# Example function for training a text classifier (if you need it)
def train_text_model(training_data, labels):
    # Preprocess the training data and labels
    encoded_data = tokenizer(training_data, truncation=True, padding=True, max_length=512)
    
    # Train model here...
    # This is an example, you would need labeled data to train properly.
    pass

