import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load the SMS dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# Convert labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Separate features and target labels
X = df['message']
y = df['label']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text into numerical feature vectors using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Multinomial Naive Bayes model
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vec, y_train)

# Evaluate the model
y_pred = nb_classifier.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Save the model and vectorizer
joblib.dump(nb_classifier, 'model/nb_classifier.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')
