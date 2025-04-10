# Quick Summary of What You Did:
# Step	Description
# ✅ Load & label data	Fake = 1, True = 0
# ✅ Clean text	Lowercase, remove punctuation, stopwords
# ✅ Visualize	Count plot + Word clouds
# ✅ Feature extraction	TF-IDF vectorization
# ✅ Model training	Logistic Regression
# ✅ Evaluation	Accuracy + classification report

import pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
# Load CSVs
fake_dtf = pd.read_csv('Fake.csv')
true_dtf = pd.read_csv('True.csv')

fake_dtf["label"] = 1
true_dtf["label"] = 0

df = pd.concat([fake_dtf, true_dtf], ignore_index=True)
df = df[["text", "label"]]

# Preprocessing without nltk
def preprocess(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation and numbers
    tokens = text.split()
    tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)

# Apply preprocessing
df["clean_text"] = df["text"].apply(preprocess)

# Save cleaned file
df.to_csv("cleaned_data.csv", index=False)
print("Preprocessing done. Cleaned data saved to cleaned_data.csv")

##Samples for true and fake news
plt.figure(figsize=(6,4))
sns.countplot(data=df,x="label")
plt.xticks([0,1],["True news","Fake news"])
plt.title("Distribution of True vs Fake News")
plt.xlabel("News Type")
plt.ylabel("Count")
plt.savefig("Label Distribution")
plt.show()

## wordcloud for fake and real news

fake_text = " ".join(df[df["label"]==1]["clean_text"])
true_text = " ".join(df[df["label"]==0]["clean_text"])

plt.figure(figsize=(10,5))
wordcloud_fake = WordCloud(width=800,height=400,background_color='black',colormap='Blues').generate(fake_text)
plt.imshow(wordcloud_fake,interpolation='bilinear')
plt.axis("off")
plt.title("WordCloud - Fake News")
plt.savefig("Fake News wordcloud")
plt.show()


plt.figure(figsize=(10,5))
wordcloud_true = WordCloud(width=800, height = 400, background_color='black',colormap='Oranges').generate(true_text)
plt.imshow(wordcloud_true,interpolation='bilinear')
plt.axis("off")
plt.title("WordCloud- True News")
plt.savefig("True News wordcloud")
plt.show()


vectorizer = TfidfVectorizer(stop_words='english',max_df=0.7)
X = vectorizer.fit_transform(df['text'])

## x hold the numerical features extracted from the text

y = df['label']
X_train, X_test , y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
## Logistic regression is good for binary classification
model = LogisticRegression()
model.fit(X_train,y_train)


y_prediction = model.predict(X_test)
print("Accuracy: ", accuracy_score(y_test,y_prediction))
print("Report:\n", classification_report(y_test,y_prediction))


with open('logistic_model.pkl','wb') as file:
    pickle.dump(model,file)

with open('tfidf_vectorizer.pkl','wb') as file:
    pickle.dump(vectorizer,file)


print("Model and vectorizer model")