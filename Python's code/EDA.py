# Importing librairies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from collections import Counter
from transformers import pipeline
from wordcloud import WordCloud

# Importing text data
text = pd.read_csv(r'D:\Projects\IT\Data Science & IA\Employee_Emotion_Analysis\Data\message_cleaned.csv')

# Count the most frequent words
word_freq = Counter([word for tokens in text['Message_cleaned'] for word in tokens.split()])

# Print 20 most common word
word_most_common = word_freq.most_common(20)

# Break down for visualization
words, counts = zip(*word_most_common)

# Graphique des mots les plus fréquents
plt.figure(figsize=(12,6))
plt.barh(words, counts, color="skyblue")
plt.gca().invert_yaxis()  # Inverser l'axe pour que le plus fréquent soit en haut
plt.xlabel("Number of occurrences")
plt.title("Most frequent words in messages")
plt.show()

all_text = " ".join(text["Message_cleaned"])

# Generate word cloud
wordcloud = WordCloud(height=400, width=800, background_color='white', colormap='coolwarm').generate(all_text)

# Graphical Visualization
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word cloud of employee's message")
plt.show()