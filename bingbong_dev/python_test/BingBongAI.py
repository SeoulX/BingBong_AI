import json
import os
import random
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv, dotenv_values 
load_dotenv()

genai.configure(api_key=os.getenv("API_KEY"))


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
folder_path_json = os.path.join(BASE_DIR, 'json')
file_path_json = os.path.join(folder_path_json, 'bingbong_combined.json')

with open(file_path_json, 'r') as f:
    data = json.load(f)
    
df = pd.DataFrame(data['intents'])

dic = {"tag":[], "patterns":[], "responses":[]}
for i in range(len(df)):
    ptrns = df[df.index == i]['patterns'].values[0]
    rspns = df[df.index == i]['responses'].values[0]
    tag = df[df.index == i]['tag'].values[0]
    for j in range(len(ptrns)):
        dic['tag'].append(tag)
        dic['patterns'].append(ptrns[j])
        dic['responses'].append(rspns)
        
df = pd.DataFrame.from_dict(dic)
df_responses = df.explode('responses')
all_patterns = ' '.join(df['patterns'])

# Preprocessing the Dataset============================================

import re

def preprocess_text(s):
    s = re.sub('[^a-zA-Z\']', ' ', s)
    s = s.lower() 
    s = s.split() 
    s = " ".join(s)
    return s

df['patterns'] = df['patterns'].apply(preprocess_text)
df['tag'] = df['tag'].apply(preprocess_text)

df['tag'].unique()

len(df['tag'].unique())

X = df['patterns']
y = df['tag']

X = df['patterns']
y = df['tag']

# Implementation of NLP & Sentiment Analysis (BERT) ===================

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 128

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_labels = len(np.unique(y_encoded))

np.save(f'{BASE_DIR}/label_encoder_classes.npy', label_encoder.classes_)

def load_model_and_tokenizer(num_labels):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    model_save_path = f'{BASE_DIR}/saved_models/bingbong_combined_10E.pth'
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.to(device)
    return model, tokenizer


def load_label_encoder(label_encoder_path=f'{BASE_DIR}/label_encoder_classes.npy'):
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(label_encoder_path, allow_pickle=True)
    return label_encoder


def predict_intent(text, model, tokenizer, label_encoder, max_len=128):
    encoded_dict = tokenizer.encode_plus(
        text,                      
        add_special_tokens=True,   
        max_length=max_len,        
        padding='max_length',    
        return_attention_mask=True,
        return_tensors='pt',       
    )

    input_ids = encoded_dict['input_ids'].to(device)
    attention_mask = encoded_dict['attention_mask'].to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    probabilities = F.softmax(logits, dim=1)
    predicted_class_index = torch.argmax(probabilities).item()

    predicted_label = label_encoder.inverse_transform([predicted_class_index])[0]
    confidence = probabilities[0][predicted_class_index].item()
    return predicted_label, confidence

model, tokenizer = load_model_and_tokenizer(num_labels)
label_encoder = load_label_encoder()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_response(user_input):
    global model, tokenizer, label_encoder
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    folder_path_json = os.path.join(BASE_DIR, 'json')
    file_path_json = os.path.join(folder_path_json, 'bingbong_combined.json')

    with open(file_path_json, 'r') as f:
        data_json = json.load(f)

    if user_input.lower() == "quit":
        return "Goodbye!"

    predicted_label, probability = predict_intent(user_input, model, tokenizer, label_encoder)

    print(f"Predicted Label: {predicted_label}") 

    for intent in data_json['intents']:
        tag = preprocess_text(intent['tag'])
        if  tag == predicted_label:
            responses = intent['responses']
            best_response = np.random.choice(responses)
            break
    else:
        best_response = "I'm not sure I understand. Could you rephrase that?"
            
    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
    }

    modelAI = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    # safety_settings = Adjust safety settings
    # See https://ai.google.dev/gemini-api/docs/safety-settings
    )
    chat_session = modelAI.start_chat(
        history=[
        ]
        )
    
    print("User:", user_input)
    print("Model:", best_response)
    promp = f"User Input: {user_input} \n Model Input: {best_response} \n Give a paraphrased version of the model output,. Always remember you are BingBong a friendly bot. Remove quotation marks on your reply"
    
    response = chat_session.send_message(promp)
    
    print(response.text)
    
    return response.text