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

# Emotion analysis
def analyze_emotion(text):
    emotion =NRCLex(text)
    return emotion.raw_emotion_scores

text['emotion_score']=text['Message_cleaned'].apply(analyze_emotion)

# Extract specific emotion
def extract_specific_emotion(score):
    relevant_emotion ={
        'stress' : score.get('stress', 0),
        'anticipation': score.get('anticipation',0),
        'sadness': score.get('sadness',0),
        'joy' : score.get('joy',0),
        'fear': score.get('fear',0),
        'trust' : score.get('trust',0),
        'disgust' : score.get('disgust',0),
        'surprise' : score.get('surprise',0),
        'positive' : score.get('positive',0)
    }
    return relevant_emotion

# Apply extracting secific emotion
text['specific_emotion'] = text['emotion_score'].apply(extract_specific_emotion)

# Create specific emotion as a DataFrame
emotion_specific = text['specific_emotion'].apply(lambda x: max(x, key=x.get))

# Data viz of distribution of specific emotion
plt.figure(figsize=(12,5))
emotion_specific.sum().sort_values(ascending=False).plot(kind='bar', colormap='viridis')
plt.title('Distribution of emotion')
plt.xlabel('Emotions')
plt.ylabel('Frequences')

# Choose emotions with a high score
text['dominant_emotion'] = text['specific_emotion'].apply(lambda x: max(x, key=x.get) if x else "unknown")


# Convertir 'Date' en datetime
text['Date'] = pd.to_datetime(text['Date'])

# Extraire le trimestre et l'année sous forme de chaîne
text['Trimestre'] = text['Date'].dt.to_period('Q').astype(str)

# Compter les émotions par trimestre
emotion_trend = text.groupby(['Trimestre', 'dominant_emotion']).size().unstack(fill_value=0)

# Tracer le lineplot
plt.figure(figsize=(14, 6))
sns.lineplot(data=emotion_trend, marker='o')

plt.title("Évolution des émotions par trimestre")
plt.xlabel("Trimestre")
plt.ylabel("Nombre d'occurrences")
plt.xticks(rotation=45)  # Rotation pour lisibilité
plt.grid()
plt.legend(title="Émotions", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Compter les émotions par département
emotion_by_dept = text.groupby(['Département', 'dominant_emotion']).size().unstack(fill_value=0)

# Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(emotion_by_dept, cmap="coolwarm", annot=True, fmt="d")
plt.title("Émotions par département")
plt.xlabel("Émotions")
plt.ylabel("Département")
plt.xticks(rotation=45)
plt.show()

# Compter les émotions par source (Slack, Email, Teams)
emotion_by_source = text.groupby(['Source', 'dominant_emotion']).size().unstack(fill_value=0)

# Barplot empilé
emotion_by_source.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title("Répartition des émotions par plateforme")
plt.xlabel("Source")
plt.ylabel("Nombre d'occurrences")
plt.xticks(rotation=0)
plt.legend(title="Émotions", bbox_to_anchor=(1.05, 1), loc='upper left')