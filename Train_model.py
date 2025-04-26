import pandas as pd
print("Pandas library imported successfully.")
df_fake = pd.read_csv('Fake.csv')
df_true = pd.read_csv('True.csv')
print("Datasets loaded successfully.")
print("\nFirst 5 rows of Fake News Data:")
print(df_fake.head())

print("\nFirst 5 rows of True News Data:")
print(df_true.head())

print("\nInfo for Fake News Data:")
df_fake.info()

print("\nInfo for True News Data:")
df_true.info()
print(f"\nShape of Fake News Data: {df_fake.shape}")
print(f"Shape of True News Data: {df_true.shape}")

df_fake['label'] = 0
df_true['label'] = 1

print("\nAdded 'label' column.")
print("Fake News Data head with label:")
print(df_fake.head())
print("\nTrue News Data head with label:")
print(df_true.head())

df_fake_selected = df_fake[['text', 'label']]
df_true_selected = df_true[['text', 'label']]

df_combined = pd.concat([df_fake_selected, df_true_selected], ignore_index=True)

print(f"\nCombined DataFrame shape: {df_combined.shape}")
print("Combined DataFrame head:")
print(df_combined.head())
print("\nCombined DataFrame tail (to see both labels):")
print(df_combined.tail())

df_shuffled = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nShuffled DataFrame head:")
print(df_shuffled.head())
print("\nShuffled DataFrame tail:")
print(df_shuffled.tail())

print("\nMissing values in shuffled DataFrame:")
print(df_shuffled.isnull().sum())

import re
print("\nImported 're' library for text cleaning.")

def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r'[^a-z\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text

print("Defined text preprocessing function.")

print("\nStarting text preprocessing...")
df_shuffled['cleaned_text'] = df_shuffled['text'].apply(preprocess_text)
print("Text preprocessing complete.")

print("\nOriginal vs Cleaned Text (First 5 rows):")
print(df_shuffled[['text', 'cleaned_text']].head())


from sklearn.feature_extraction.text import TfidfVectorizer
print("\nImported TfidfVectorizer.")


tfidf_vectorizer = TfidfVectorizer(max_features=5000) 

print("Initialized TfidfVectorizer.")

print("\nStarting TF-IDF vectorization...")
X_tfidf = tfidf_vectorizer.fit_transform(df_shuffled['cleaned_text'])
print("TF-IDF vectorization complete.")

print(f"\nShape of TF-IDF matrix (X): {X_tfidf.shape}")


y = df_shuffled['label']

print(f"Shape of labels (y): {y.shape}")
print("\nFirst 5 labels:")
print(y.head())

from sklearn.model_selection import train_test_split
print("\nImported train_test_split.")


X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

print("Data split into training and testing sets complete.")

print(f"\nShape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

from sklearn.linear_model import LogisticRegression
print("\nImported LogisticRegression model.")


model = LogisticRegression(random_state=42, max_iter=1000)
print("Initialized Logistic Regression model.")


print("\nStarting model training...")
model.fit(X_train, y_train)
print("Model training complete.")

print("\nMaking predictions on the test set...")
y_pred = model.predict(X_test)
print("Predictions made.")


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print("\nImported evaluation metrics.")

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)


print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Fake (0)', 'True (1)']))




def predict_news(news_text):
    processed_text = preprocess_text(news_text)
  
    vectorized_text = tfidf_vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)
    return prediction[0]

print("\nDefined prediction function.")


print("\n--- Test the model with your news ---")
user_news = input("Please enter the news text you want to check: ")

if user_news.strip(): 
    user_prediction = predict_news(user_news)

    print(f"\nPrediction for your news: {user_prediction}")
    if user_prediction == 0:
        print("Result: Model predicts this is FAKE news.")
    else:
        print("Result: Model predicts this is TRUE news.")
else:
    print("You did not enter any news text.")


print("\n--- Starting Step 9: Save Model & Vectorizer ---")
import joblib

model_filename = 'fake_news_model.pkl'
vectorizer_filename = 'tfidf_vectorizer.pkl'

print(f"Saving the trained model to {model_filename}...")
joblib.dump(model, model_filename)
print("Model saved successfully.")

print(f"Saving the TF-IDF vectorizer to {vectorizer_filename}...")
joblib.dump(tfidf_vectorizer, vectorizer_filename) 
print("Vectorizer saved successfully.")
print("--- Step 9 Complete ---")

print("\n=== TRAINING SCRIPT FINISHED ===")