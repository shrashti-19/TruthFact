# 📰 Fake News Detection using TF-IDF and Logistic Regression

This project uses **Logistic Regression** and **TF-IDF vectorization** to classify news articles as either **Fake** or **True**. It includes preprocessing, visualization, model training, evaluation, and deployment-ready code using `Streamlit`.

## 📌 Features

- Data loading from `Fake.csv` and `True.csv`
- Text preprocessing: lowercase, punctuation removal, stopword removal
- Visualizations: 
  - Count plot of fake vs real news
  - Word clouds for both fake and real news
- Feature Extraction using **TF-IDF**
- Binary classification using **Logistic Regression**
- Model performance metrics: accuracy, classification report
- Model and vectorizer saved using `pickle` for deployment

---

## 📁 Project Structure
├── app.py 
# Streamlit web app 
├── model_training.py 
# Training and evaluation code 
├── cleaned_data.csv # Preprocessed dataset 
├── logistic_model.pkl
# Trained Logistic Regression model
├── tfidf_vectorizer.pkl 
# Trained TF-IDF vectorizer 
├── Fake.csv # Fake news data 
├── True.csv # True news data 
├── distributions/ 
# Plots (word clouds, countplot) 
README.md # Project documentation 
.gitignore # Ignore pkl, csv files if needed


## 🚀 How to Run

1. Clone the repository
```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd TruthFact
### 2. Install Dependencies
pip install -r requirements.txt

3. Train the Model
python model_training.py

4. Run the Streamlit App
streamlit run app.py

🧠 Model Details
Algorithm: Logistic Regression

Vectorization: TF-IDF (Term Frequency-Inverse Document Frequency)

Train-Test Split: 80/20

Stopwords: sklearn's built-in English stopwords

📊 Sample Output
Accuracy: ~98% (depending on the dataset)

Classification Report: Includes precision, recall, and F1-score




