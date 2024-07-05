import json
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, 'bingbong.json')

with open(file_path, 'r') as f:
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

# =====================================================================

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
# Generate a word cloud image
wordcloud = WordCloud(background_color='white', max_words=100, contour_width=3, contour_color='steelblue').generate(all_patterns)

# # Display the generated image:
# plt.figure(figsize=(10, 6))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')  # Do not show axes to keep it clean
# plt.title('Word Cloud for Patterns')
# plt.show()

# # =====================================================================

# df['pattern_length'] = df['patterns'].apply(len)
# plt.figure(figsize=(10, 6))
# sns.histplot(df['pattern_length'], bins=30, kde=True)
# plt.title('Distribution of Pattern Lengths')
# plt.xlabel('Length of Patterns')
# plt.ylabel('Frequency')
# plt.show()

# # =====================================================================

# plt.figure(figsize=(22, 16))
# sns.countplot(y='tag', data=df, order=df['tag'].value_counts().index)
# plt.title('Distribution of Intents')
# plt.xlabel('Number of Patterns')
# plt.ylabel('Intent')
# plt.show()

# # =====================================================================

# # 3. Number of Unique Responses per Intent
# df_unique_responses = df_responses.groupby('tag')['responses'].nunique().reset_index(name='unique_responses')
# plt.figure(figsize=(22, 16))
# sns.barplot(x='unique_responses', y='tag', data=df_unique_responses.sort_values('unique_responses', ascending=False))
# plt.title('Number of Unique Responses per Intent')
# plt.xlabel('Number of Unique Responses')
# plt.ylabel('Intent')
# plt.show()

# # =====================================================================

# # Calculating response lengths from the exploded DataFrame
# df_responses['response_length'] = df_responses['responses'].apply(len)
# plt.figure(figsize=(12, 8))
# sns.histplot(df_responses['response_length'], bins=30, kde=True)
# plt.title('Distribution of Response Lengths')
# plt.xlabel('Length of Responses')
# plt.ylabel('Frequency')
# plt.show()

# =====================================================================

import re

# Preprocessing function
def preprocess_text(s):
    s = re.sub('[^a-zA-Z\']', ' ', s)  # Keep only alphabets and apostrophes
    s = s.lower()  # Convert to lowercase
    s = s.split()  # Split into words
    s = " ".join(s)  # Rejoin words to ensure clean spacing
    return s

# Apply preprocessing to the patterns
df['patterns'] = df['patterns'].apply(preprocess_text)
df['tag'] = df['tag'].apply(preprocess_text)

df['tag'].unique()

len(df['tag'].unique())

X = df['patterns']
y = df['tag']

# =========================================

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

X = df['patterns']
y = df['tag']
# Tokenization and Encoding the Data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 128  # Max sequence length

def encode_texts(texts, max_len):
    input_ids = []
    attention_masks = []
    
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    
    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)


# Encoding labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_labels = len(np.unique(y_encoded))

np.save(f'{BASE_DIR}/label_encoder_classes.npy', label_encoder.classes_)

# Encode the patterns
input_ids, attention_masks = encode_texts(X, max_len)
labels = torch.tensor(y_encoded)

# Splitting the dataset into training and validation
dataset = torch.utils.data.TensorDataset(input_ids, attention_masks, labels)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
validation_dataloader = DataLoader(val_dataset, batch_size=16)

# Model and Optimization
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
optimizer = AdamW(model.parameters(), lr=2e-5)

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 10
best_val_loss = float('inf')
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    all_train_labels = []
    all_train_preds = []
    for batch in train_dataloader:
        b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
        
        b_labels = b_labels.long()

        optimizer.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]

        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()

        # Calculate and store training accuracy
        logits = outputs.logits
        _, preds = torch.max(logits, dim=1)
        all_train_labels.extend(b_labels.cpu().numpy())
        all_train_preds.extend(preds.cpu().numpy())

    avg_train_loss = total_train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)
    train_accuracy = accuracy_score(all_train_labels, all_train_preds)
    train_accuracies.append(train_accuracy)
    print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")

    # Validation Loop
    model.eval()
    total_eval_loss = 0
    all_val_labels = []
    all_val_preds = []
    with torch.no_grad():
        for batch in validation_dataloader:
            b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
            
            b_labels = b_labels.long()

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]

            total_eval_loss += loss.item()

            # Calculate and store validation accuracy
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)
            all_val_labels.extend(b_labels.cpu().numpy())
            all_val_preds.extend(preds.cpu().numpy())
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    val_losses.append(avg_val_loss)
    val_accuracy = accuracy_score(all_val_labels, all_val_preds)
    val_accuracies.append(val_accuracy)
    print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model_directory = os.path.join(BASE_DIR, "saved_models/BingBongTrain.pth")
        torch.save(model.state_dict(), model_directory)
        print("Model saved!")


# Plot losses
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot accuracies
plt.figure(figsize=(12, 6))
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()