# Importing librairies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from collections import Counter
from transformers import pipeline
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex

# Importing text data
text = pd.read_csv(r'D:\Projects\IT\Data Science & IA\Employee_Emotion_Analysis\Data\message_cleaned.csv')

# Vader Sentiment
sid = SentimentIntensityAnalyzer()  
def analyze_sentiment(text):
    scores = sid.polarity_scores(text)
    return scores['compound']  # Valeur entre -1 et 1, où -1 est négatif et 1 est positif

# Apply vader sentiment
text['sentiment_score'] = text['Message_cleaned'].apply(analyze_sentiment)

# Define cateogorize of sentiment
def categorize_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

text['sentiment'] = text['sentiment_score'].apply(categorize_sentiment)

# Counting feelings
sentiments_counts = text['sentiment'].value_counts()

# Graphical visualization
sns.set(style='whitegrid', palette='pastel',font_scale='1.5',color_codes=True)
plt.figure(figsize=(12,5))
sns.barplot(x=sentiments_counts.index, y=sentiments_counts.values)
plt.xlabel('Sentiments')
plt.ylabel('Number of occurrences')
plt.title('Distribution of sentiments')
plt.show()

