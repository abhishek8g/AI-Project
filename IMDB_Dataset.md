# 1. Project Content & Description
The project involves building a machine learning model to classify movie reviews as either positive or negative sentiment. It covers the following key steps:

Data Loading: Reading a CSV dataset containing IMDB movie reviews and their corresponding sentiments.
Text Preprocessing: Cleaning the text data by handling contractions, converting text to lowercase, removing HTML tags, special characters, and stop words. It also includes stemming and lemmatization.
Feature Extraction: Converting text data into numerical features using techniques like TF-IDF (Term Frequency-Inverse Document Frequency).
Model Training: Training sentiment classification models, specifically a Naive Bayes classifier and a Logistic Regression classifier.
Model Evaluation: Assessing the performance of the trained models.
Model Interpretation: Using SHAP (SHapley Additive exPlanations) to explain the predictions of the sentiment analysis model, showing how individual words contribute to the positive or negative classification of a review.
# 2. Project Code
The project code is written in Python within a Jupyter Notebook. Key aspects of the code include:

Importing necessary libraries such as pandas, numpy, nltk, re, contractions, scikit-learn (for TfidfVectorizer, CountVectorizer, BernoulliNB, LogisticRegression, accuracy_score), and shap.
Functions for text cleaning (e.g., remove_html, remove_special_chars, text_preprocessing).
Code for splitting data into training and testing sets.
Implementation of TF-IDF vectorization.
Training and prediction logic for Naive Bayes and Logistic Regression models.
SHAP explainer for linear models, applied to specific document predictions.
### Cell 1: Install contractions library
!pip install contractions

### Cell 2: Import necessary libraries
import pandas as pd
import numpy as np
import nltk
import re
import contractions
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import shap # Make sure shap is installed: pip install shap

### Download NLTK resources (if not already downloaded)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

### Cell 3: Load the dataset
### Assuming the dataset is named 'IMDB Dataset.csv' and is in the same directory
try:
    df = pd.read_csv('IMDB Dataset.csv')
    print("Dataset loaded successfully.")
    print(df.head())
    print(df.shape)
except FileNotFoundError:
    print("Error: 'IMDB Dataset.csv' not found. Please ensure the file is in the correct directory.")
    # Create a dummy DataFrame for demonstration if file is not found
    data = {'review': [
        "A wonderful movie, really enjoyed it!",
        "This film was terrible, a complete waste of time.",
        "It's an average film, nothing special.",
        "Absolutely brilliant! Highly recommend.",
        "Worst movie ever, so boring and predictable."
    ], 'sentiment': ['positive', 'negative', 'negative', 'positive', 'negative']}
    df = pd.DataFrame(data)
    print("Using dummy dataset for demonstration.")
    print(df.head())
    print(df.shape)


### Cell 4: Text Preprocessing Functions
def remove_html(text):
    """Removes HTML tags from the text."""
    pattern = re.compile('<.*?>')
    return pattern.sub(r'', text)

def remove_special_chars(text):
    """Removes special characters and digits from the text."""
    pattern = re.compile('[^a-zA-Z\s]')
    return pattern.sub(r'', text)

def convert_lower(text):
    """Converts text to lowercase."""
    return text.lower()

def expand_contractions(text):
    """Expands contractions in the text."""
    return contractions.fix(text)

def remove_stopwords(text):
    """Removes common English stopwords from the text."""
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def apply_stemming(text):
    """Applies Porter Stemming to the text."""
    ps = PorterStemmer()
    words = text.split()
    stemmed_words = [ps.stem(word) for word in words]
    return ' '.join(stemmed_words)

def apply_lemmatization(text):
    """Applies WordNet Lemmatization to the text."""
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

def text_preprocessing(text):
    """Applies a sequence of preprocessing steps to the text."""
    text = remove_html(text)
    text = convert_lower(text)
    text = expand_contractions(text)
    text = remove_special_chars(text)
    text = remove_stopwords(text)
    text = apply_lemmatization(text) # Using lemmatization as it's generally preferred over stemming for better accuracy
    return text

### Cell 5: Apply preprocessing to the 'review' column
df['cleaned_review'] = df['review'].apply(text_preprocessing)
print("\nReviews after preprocessing:")
print(df[['review', 'cleaned_review']].head())

### Cell 6: Convert sentiment to numerical (0 for negative, 1 for positive)
df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1})
print("\nSentiment mapped to numerical values:")
print(df.head())

### Cell 7: Split data into features (X) and target (y)
X = df['cleaned_review']
y = df['sentiment']

### Cell 8: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

### Cell 9: Feature Extraction (TF-IDF Vectorization)
vectorizer = TfidfVectorizer(max_features=5000) # Limiting features for better performance and interpretability
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print(f"\nShape of vectorized training data: {X_train_vectorized.shape}")
print(f"Shape of vectorized testing data: {X_test_vectorized.shape}")

### Cell 10: Train and Evaluate Bernoulli Naive Bayes Classifier
bnb_classifier = BernoulliNB()
bnb_classifier.fit(X_train_vectorized, y_train)
y_pred_bnb = bnb_classifier.predict(X_test_vectorized)
accuracy_bnb = accuracy_score(y_test, y_pred_bnb)
print(f"\nBernoulli Naive Bayes Classifier Accuracy: {accuracy_bnb:.4f}")

### Cell 11: Train and Evaluate Logistic Regression Classifier
lr_classifier = LogisticRegression(max_iter=1000) # Increase max_iter for convergence
lr_classifier.fit(X_train_vectorized, y_train)
y_pred_lr = lr_classifier.predict(X_test_vectorized)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Classifier Accuracy: {accuracy_lr:.4f}")

### Cell 12: SHAP Explainer for Logistic Regression (Example for one prediction)
print("\n--- SHAP Explanations ---")

### Choose a document from the test set to explain
### For demonstration, let's pick the first document from the test set
idx_to_explain = 0
original_review = X_test.iloc[idx_to_explain]
true_sentiment = y_test.iloc[idx_to_explain]
predicted_sentiment = lr_classifier.predict(X_test_vectorized[idx_to_explain])[0]

print(f"\nOriginal Review (Test Set Index {idx_to_explain}):\n{original_review}")
print(f"True Sentiment: {'Positive' if true_sentiment == 1 else 'Negative'}")
print(f"Predicted Sentiment: {'Positive' if predicted_sentiment == 1 else 'Negative'}")

### Create a SHAP explainer for the Logistic Regression model
### Using feature_perturbation="interventional" for linear models with sparse data
explainer = shap.LinearExplainer(lr_classifier, X_train_vectorized, feature_perturbation="interventional")

### Calculate SHAP values for the chosen document
shap_values = explainer.shap_values(X_test_vectorized[idx_to_explain])

### Get feature names (words) from the vectorizer
feature_names = vectorizer.get_feature_names_out()

### Create a mapping from feature index to feature name
feature_map = {i: name for i, name in enumerate(feature_names)}

### Get the SHAP values for the specific instance
instance_shap_values = shap_values[0] # For binary classification, shap_values is a list of arrays

### Get the indices of non-zero SHAP values (i.e., words that contributed)
contributing_indices = np.where(instance_shap_values != 0)[0]

### Sort contributing features by absolute SHAP value to see most impactful words
sorted_indices = contributing_indices[np.argsort(np.abs(instance_shap_values[contributing_indices]))[::-1]]

print("\nTop contributing words and their SHAP values:")
for i in sorted_indices[:10]: # Display top 10 contributing words
    word = feature_map[i]
    shap_value = instance_shap_values[i]
    print(f"  - '{word}': {shap_value:.4f}")



### Cell 13: Example of explaining another random document
print("\n--- Explaining another random document ---")
random_idx = np.random.randint(0, len(X_test))
original_review_rand = X_test.iloc[random_idx]
true_sentiment_rand = y_test.iloc[random_idx]
predicted_sentiment_rand = lr_classifier.predict(X_test_vectorized[random_idx])[0]

print(f"\nOriginal Review (Test Set Index {random_idx}):\n{original_review_rand}")
print(f"True Sentiment: {'Positive' if true_sentiment_rand == 1 else 'Negative'}")
print(f"Predicted Sentiment: {'Positive' if predicted_sentiment_rand == 1 else 'Negative'}")

shap_values_rand = explainer.shap_values(X_test_vectorized[random_idx])
instance_shap_values_rand = shap_values_rand[0]

contributing_indices_rand = np.where(instance_shap_values_rand != 0)[0]
sorted_indices_rand = contributing_indices_rand[np.argsort(np.abs(instance_shap_values_rand[contributing_indices_rand]))[::-1]]

print("\nTop contributing words and their SHAP values:")
for i in sorted_indices_rand[:10]:
    word = feature_map[i]
    shap_value = instance_shap_values_rand[i]
    print(f"  - '{word}': {shap_value:.4f}")


# 3. Key Technologies
The project leverages the following key technologies and Python libraries:

Data Manipulation: pandas, numpy
Text Preprocessing: contractions, nltk (for stop words, stemming, lemmatization), re (regular expressions)
Machine Learning: scikit-learn (for TfidfVectorizer, CountVectorizer, BernoulliNB, LogisticRegression, accuracy_score, train_test_split)
Model Interpretability: shap (SHapley Additive exPlanations)
Interactive Development Environment: Jupyter Notebook
# 4. Output
The notebook's outputs include:

Confirmation of library installations.
Display of the initial dataset (e.g., df.head()).
Outputs from text preprocessing steps.
Model accuracy scores for both Naive Bayes and Logistic Regression.
SHAP values and plots (though not fully visible in the provided content snippet, the code indicates SHAP value calculation and plotting).
# 5. Further Research
Based on this project, potential areas for further research could include:

Exploring other advanced NLP techniques: Investigating transformer-based models (e.g., BERT, GPT) for sentiment analysis for potentially higher accuracy.
Hyperparameter tuning: Optimizing the hyperparameters of the current models (Naive Bayes, Logistic Regression) and vectorizers (TF-IDF) for better performance.
Different feature extraction methods: Experimenting with word embeddings (Word2Vec, GloVe) or FastText as alternatives to TF-IDF.
More comprehensive model evaluation: Utilizing other metrics like precision, recall, F1-score, and ROC curves, and performing cross-validation.
Advanced interpretability techniques: Applying other interpretability methods (e.g., LIME, saliency maps) and comparing their insights with SHAP.
Handling imbalanced datasets: If the sentiment classes are imbalanced, exploring techniques like oversampling or undersampling.
Deployment: Investigating how to deploy the trained model as an API for real-time sentiment prediction.

