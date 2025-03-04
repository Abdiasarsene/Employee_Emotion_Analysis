# Importing librairies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from collections import Counter
from transformers import pipeline

# Importing text data
text_data = pd.read_csv(r'D:\Projects\IT\Data Science & IA\Employee_Emotion_Analysis\Data\emotion.csv')

# Loading and previewing data
text_data.tail() #Print data
text_data.info() #Info about the text data
text_data.count() #Number of Observation

# Preprocessing with SpaCy
nlp = spacy.load("en_core_web_sm")

    # Define stop_words

custom_stop_word =  {
    "i", "you", "he", "she", "we", "they", "it", "my", "your", "his", "her", "our", "their",
    "me", "him", "us", "them", "is", "are", "was", "were", "be", "been", "being", 
    "a", "an", "the", "this", "that", "these", "those", "and", "but", "or", "if", "because", "so",
    "on", "in", "at", "by", "with", "about", "against", "between", "into", "through", "over", "under",
    "again", "further", "then", "once", "can", "will", "just", "should", "would", "could", "may", "might", "must","include", "mention", "already", "quickly", "soon", "allow", "out", "second", "far",
    "market", "million", "stock", "general", "industry", "economy", "nation", "education",
    "moment", "parent", ""
}
    # Cleaning and pre-treatment function
def preprocess_text(text):
    doc = nlp(text.lower()) #Convert to lower case
    tokens = [token.lemma_ for  token in doc if token.is_alpha and token.text not in custom_stop_word]
    return " ".join(tokens)

# Apply pre_treatment to the column [Message]
text_data['Message_cleaned'] = text_data['Message'].astype(str).apply(preprocess_text)

# Print
text_data[['Message', 'Message_cleaned']]
