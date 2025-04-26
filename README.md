# 🔍 Fake News Detection System  

<div align="center">
  
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)
![ML](https://img.shields.io/badge/ML-Logistic_Regression-FF6F00)
![Accuracy](https://img.shields.io/badge/Accuracy-98.6%25-brightgreen)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-4CAF50)

</div>

## 🚀 10-Second Pitch  
**What:** ML system detecting fake news with **98.6% accuracy**  
**Why Matters:** Combats misinformation using NLP  
**Core Tech:** Python · Scikit-learn · Pandas  


## Workflow / Steps Implemented
1.  **Data Loading:** Loaded the `Fake.csv` and `True.csv` datasets using the Pandas library.
2.  **Data Preparation:** Assigned binary labels (0 for Fake, 1 for True), combined the datasets into a single DataFrame, and randomly shuffled the entries. Checked for and handled any missing values.
3.  **Text Preprocessing:** Cleaned the news article text by converting to lowercase, removing punctuation and numerical characters using regular expressions, and normalizing whitespace.
4.  **Feature Extraction:** Employed `TfidfVectorizer` from Scikit-learn to convert the cleaned text into numerical TF-IDF features, considering the top 5000 most frequent terms.
5.  **Train/Test Split:** Partitioned the vectorized data and corresponding labels into training (80%) and testing (20%) sets using `train_test_split`.
6.  **Model Training:** Trained a `LogisticRegression` classification model using the training data (`X_train`, `y_train`).
7.  **Model Evaluation:** Assessed the trained model's performance on the unseen test set (`X_test`, `y_test`) using standard metrics: Accuracy, Confusion Matrix, Precision, Recall, and F1-Score.
8.  **Model Persistence:** Saved the trained Logistic Regression model and the fitted TF-IDF vectorizer to disk using Joblib for future use.


---


| **Aspect**           |    **What I Delivered**                                 | **Your Benefit**                          |  
|----------------------|---------------------------------------------------|-------------------------------------------|  
| **Production Skills** | End-to-end pipeline (Data → Model → Deployment)    | Reduces onboarding time for ML projects   |  
| **Technical Depth**  | TF-IDF + Regex + Logistic Regression              | Proves ability to implement core NLP flows|  
| **Code Quality**     | PEP8-compliant · Modular · Documented             | Ready for team collaboration              |  
| **Business Impact**  | 98.6% accuracy on real-world data                 | Risk reduction in misinformation handling |  

---

## 🛠️ Technical Breakdown  

### 🔑 Key Components  
| **Component**       | **Tech Used**                | **Purpose**                     |  
|----------------------|------------------------------|----------------------------------|  
| Text Cleaning        | Regex · Lowercasing          | Noise reduction in news text    |  
| Feature Engineering  | TF-IDF (5000 features)       | Convert text → Numerical vectors|  
| Model Training       | Scikit-learn LogisticRegression | Binary classification          |  
| Deployment           | Joblib serialization         | Reusable model artifacts        |  

### 📊 Performance Snapshot  
| Metric        | Fake News (0) | Real News (1) |  
|---------------|---------------|---------------|  
| **Precision** | 0.99          | 0.98          |  
| **Recall**    | 0.98          | 0.99          |  

---

## ⚡ 3-Step Implementation  
# 1. Install (5s)
pip install pandas scikit-learn joblib

# 2. Train (2min)
python train_model.py  # Generates model.pkl

# 3. Predict (Live Demo)
python predict_news_app.py  

+ # Input: "Aliens control White House!" → Output: 🔴 FAKE



🌟 Why This Project?
For Recruiters	                  For My Growth
Proves I can ship ML systems	First hands-on NLP experience
Shows documentation skills    	Learned TF-IDF vectorization
98.6% = Technical capability	Confidence to tackle BERT next


## Learning & Future Scope
This project provided valuable hands-on experience with the fundamental workflow of an NLP text classification task. Potential future enhancements include:
* Implementing more advanced text preprocessing techniques (e.g., stop word removal, stemming/lemmatization).
* Experimenting with different classification models (e.g., Naive Bayes, SVM, Random Forest).
* Exploring word embeddings (like Word2Vec or GloVe) or transformer-based models for potentially higher accuracy.
* Developing a simple web interface (e.g., using Flask or Streamlit) for easier interaction.





