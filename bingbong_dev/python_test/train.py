import json
import random
import re
import spacy
from fuzzywuzzy import process

nlp = spacy.load("en_core_web_sm")

with open('kb.json', 'r') as file:
    kb_data = json.load(file)

context_list = []

nlp_cache = {}

def get_nlp_doc(text):
    if text not in nlp_cache:
        nlp_cache[text] = nlp(text)
    return nlp_cache[text]

def clean_text(text):
    return re.sub(r'[^\w\s]', '', text.lower().strip())

def add_to_context(user_input):
    if len(context_list) < 5:
        context_list.append(user_input)
    else:
        context_list.pop(0)
        context_list.append(user_input)

def determine_emotional_tone(user_input):
    doc = nlp(user_input)
    emotions = {
        "sad": 0, "happy": 0, "angry": 0, "neutral": 0,
        "fear": 0, "surprise": 0, "disgust": 0,
        "hopeful": 0, "excited": 0, "calm": 0,
        "confused": 0, "lonely": 0, "content": 0,
        "nostalgic": 0, "grateful": 0, "proud": 0
    }
    
    for token in doc:
        if token.pos_ == "ADJ":
            if "sad" in token.text or "depressed" in token.text:
                emotions["sad"] += 1
            elif "happy" in token.text or "joyful" in token.text:
                emotions["happy"] += 1
            elif "angry" in token.text or "mad" in token.text:
                emotions["angry"] += 1
            elif "fear" in token.text or "scared" in token.text:
                emotions["fear"] += 1
            elif "surprise" in token.text or "surprised" in token.text:
                emotions["surprise"] += 1
            elif "disgust" in token.text or "disgusted" in token.text:
                emotions["disgust"] += 1
            elif "hopeful" in token.text:
                emotions["hopeful"] += 1
            elif "excited" in token.text:
                emotions["excited"] += 1
            elif "calm" in token.text:
                emotions["calm"] += 1
            elif "confused" in token.text:
                emotions["confused"] += 1
            elif "lonely" in token.text:
                emotions["lonely"] += 1
            elif "content" in token.text:
                emotions["content"] += 1
            elif "nostalgic" in token.text:
                emotions["nostalgic"] += 1
            elif "grateful" in token.text:
                emotions["grateful"] += 1
            elif "proud" in token.text:
                emotions["proud"] += 1
            else:
                emotions["neutral"] += 1
    
    return max(emotions, key=emotions.get)

def get_response(user_input):
    user_input = clean_text(user_input)
    user_input_doc = get_nlp_doc(user_input)
    add_to_context(user_input)

    best_match = None
    best_score = 0
    best_tag = None

    all_patterns = [(pattern.lower(), intent['tag']) for intent in kb_data['intents'] for pattern in intent['patterns']]

    fuzzy_match = process.extractOne(user_input, [pattern for pattern, _ in all_patterns], score_cutoff=40)
    if fuzzy_match:
        fuzzy_best_pattern, _ = fuzzy_match
        best_tag = next(tag for pattern, tag in all_patterns if pattern == fuzzy_best_pattern)

    for pattern, tag in all_patterns:
        pattern_doc = get_nlp_doc(pattern)
        if user_input_doc.has_vector and pattern_doc.has_vector:
            score = user_input_doc.similarity(pattern_doc)
            if score > best_score:
                best_score = score
                best_match = pattern
                best_tag = tag

    if best_tag:
        emotional_tone = determine_emotional_tone(user_input)
        
        for intent in kb_data['intents']:
            if intent['tag'] == best_tag:
                responses = intent['responses']
                context_relevant_responses = [response for response in responses if emotional_tone in response.lower()]
                if context_relevant_responses:
                    response = random.choice(context_relevant_responses)
                else:
                    response = random.choice(responses)
                
                context_list[:] = [user_input for user_input in context_list if best_tag in [intent['tag'] for intent in kb_data['intents'] if intent['tag'] == best_tag]]
                return response

    return "Sorry, I didn't understand that. Could you please rephrase?"

if __name__ == '__main__':
    print("Bot: Hello! How can I assist you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Bot: Goodbye!")
            break
        response = get_response(user_input)
        print("Bot:", response)
