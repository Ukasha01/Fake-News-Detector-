import re
import joblib 
import os

print("Libraries imported successfully.")


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Defined text preprocessing function.")

print("\n--- Loading saved model and vectorizer ---")
model_filename = 'fake_news_model.pkl'
vectorizer_filename = 'tfidf_vectorizer.pkl'

if os.path.exists(model_filename) and os.path.exists(vectorizer_filename):
    try:
        model = joblib.load(model_filename)
        tfidf_vectorizer = joblib.load(vectorizer_filename)
        print("Model and vectorizer loaded successfully.")
    except Exception as e:
        print(f"Error loading files: {e}")
        print("Please ensure the model files are correct and run the training script again if needed.")
        exit()
else:
    print(f"Error: Model file '{model_filename}' or vectorizer file '{vectorizer_filename}' not found.")
    print("Please run the training script first to create these files.")
    exit()

def predict_news(news_text):
    processed_text = preprocess_text(news_text)
    vectorized_text = tfidf_vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)
    return prediction[0]

print("Defined prediction function using loaded components.")

print("\n--- Test the model with your news ---")
try:
    while True:
        user_news = input("Please enter the news text (or type 'exit' to quit): ")

        if user_news.lower() == 'exit':
            break

        if user_news.strip():
            user_prediction = predict_news(user_news)

            print(f"\nPrediction: {user_prediction}")
            if user_prediction == 0:
                print("Result: Model predicts this is FAKE news.")
            else:
                print("Result: Model predicts this is TRUE news.")
            print("-" * 30) 
        else:
            print("You did not enter any news text. Please try again or type 'exit'.")

except KeyboardInterrupt:
    print("\nExiting prediction app.")

print("\n=== PREDICTION SCRIPT FINISHED ===")