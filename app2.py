import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import json


# Load configuration
with open("model/config.json", "r") as f:
    config = json.load(f)

# Load metadata
with open("model/metadata.json", "r") as f:
    metadata = json.load(f)

from tensorflow.keras.models import load_model
model = load_model('model/Ner_model.h5')



import pickle
with open('model/word2idx.pkl', 'rb') as f:
    word2idx = pickle.load(f)
with open('model/tag2idx.pkl', 'rb') as f:
    tag2idx = pickle.load(f)
    

# Reverse mappings
 # convert list to dict
word2idx = {w: i for i, w in enumerate(word2idx)}
idx2word = {i: w for w, i in word2idx.items()}

tag2idx = {tag: idx for idx, tag in enumerate(tag2idx)}
idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    
def get_words_list(word2idx):
    words = list(word2idx.keys())
    if "ENDPAD" not in words:
        words.append("ENDPAD")
    return words

words = get_words_list(word2idx)
num_words = len(words)



max_len = 40  
def preprocess_text(text):
    
    sentences = text
    X = [[word2idx[w[0]] for w in s] for s in sentences]
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=num_words-1)

    y = [[tag2idx[w[1]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])
#padded_sequence = pad_sequences( maxlen=max_len,  padding="post", value= word2idx['ENDPAD'])

    return X

def predict_tags(sequence):
    # Get predictions
    predictions = model.predict(sequence)
    predicted_indices = np.argmax(predictions, axis=-1)[0]
    
    # Map indices to tags
    predicted_tags = [idx2tag[idx] for idx in predicted_indices[:len(words)]]
    
    return predicted_tags

st.title("Arabic Named Entity Recognition")

st.write("""
Enter an Arabic sentence to identify named entities such as people, organizations, locations, etc.
""")

# Input text area
text_input = st.text_area("Enter Arabic text:", height=100)

# Process button
if st.button("Identify Entities"):
    if text_input:
        # Preprocess text
        sequence = preprocess_text(text_input)
        
        # Get predictions
        tags = predict_tags(sequence)
        
        # Display results
        st.subheader("Results:")
        
        # Create a DataFrame for better visualization
        import pandas as pd
        results = []
        for word, tag in zip(words, tags):
            results.append({"Word": word, "Entity Type": tag})
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)
        
        # Highlight entities in text
        st.subheader("Highlighted Text:")
        highlighted_text = ""
        current_entity = None
        
        for word, tag in zip(words, tags):
            if tag == "O":
                highlighted_text += word + " "
                current_entity = None
            else:
                # Extract entity type (removing B- or I- prefix)
                entity_type = tag[2:] if tag.startswith("B-") or tag.startswith("I-") else tag
                
                # Color coding based on entity type
                if entity_type != current_entity:
                    highlighted_text += f":red[{word}]" + " "
                    current_entity = entity_type
                else:
                    highlighted_text += f":red[{word}]" + " "
        
        st.markdown(highlighted_text)
    else:
        st.warning("Please enter some text to analyze.")

# Add some information about the model

st.sidebar.header("About")
st.sidebar.info("""
This app uses a Bidirectional LSTM model trained on Arabic text for Named Entity Recognition.
It can identify various entity types in Arabic text.
""")






