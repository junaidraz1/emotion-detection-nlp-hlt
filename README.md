# Emotion Detection using Machine Learning | NLP Project

This repository contains the final project for the course **Introduction to Human Language Technology**, focused on detecting emotions in English text using traditional machine learning techniques.

## 📌 Project Overview

This project aims to explore emotion classification using the [Emotion Dataset](https://huggingface.co/datasets/dair-ai/emotion) sourced from Hugging Face. It uses tweets labeled with six emotions: `anger`, `fear`, `joy`, `love`, `sadness`, and `surprise`.

### 🧠 Goals:
- Train an emotion classification model using a Linear Support Vector Machine (SVM).
- Preprocess informal text data (tweets) for training.
- Evaluate in-domain performance.
- Test out-of-domain robustness with external emotion-labeled sentences.
- Analyze limitations and opportunities for generalization.

---

## 📂 Dataset

- **Source**: [Emotion Dataset by DAIR AI](https://huggingface.co/datasets/dair-ai/emotion)
- **Type**: English-language tweets
- **Labels**: anger, fear, joy, love, sadness, surprise

---

## ⚙️ Tech Stack

- Python
- Scikit-learn
- Pandas, NumPy
- NLTK
- Hugging Face Datasets

---

## 🧪 Preprocessing

Text preprocessing steps included:
- Lowercasing
- Removal of punctuation and special characters
- Stopword removal
- Tokenization
- TF-IDF vectorization

---

## 🤖 Model

- **Vectorizer**: TF-IDF
- **Classifier**: Linear SVM
- **Hyperparameter Tuning**: GridSearchCV

---

## 📊 Results

| Evaluation Type     | Accuracy | Macro F1-Score |
|---------------------|----------|----------------|
| In-Domain (Validation) | ~87%     | ~0.86           |
| Out-of-Domain       | ~58-60%  | Significant drop|

> ⚠️ The model failed to generalize well on out-of-domain data due to domain-specific features learned from tweet language (hashtags, slang, emojis, etc.). This emphasizes the need for domain adaptation or more generalized representations.

---

## 🔍 Key Learnings

- Traditional ML models can perform well on clean, domain-specific data.
- Preprocessing is essential for noisy social media text.
- Out-of-domain evaluation reveals overfitting risks.
- Domain adaptation and transformer models are promising next steps.

---

## 📈 Next Steps

- Experiment with BERT or DistilBERT for better generalization.
- Introduce domain adaptation techniques.
- Augment training data with more varied emotional texts.
