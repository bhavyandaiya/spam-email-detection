### Spam Email Detection

A machine learning project that classifies emails as **Spam** or **Not Spam** using Natural Language Processing (NLP) and a Naive Bayes classifier.


## Project Overview
This project demonstrates an end-to-end spam email detection pipeline:
- Text preprocessing: **NLTK**
- Feature extraction: **TF-IDF**
- Classification: **Multinomial Naive Bayes**

- Model persistence with **pickle**
- Inference using a standalone Python script


## Tech Stack
- Python
- scikit-learn
- NLTK
- Pandas
- NumPy


## How It Works
1. Raw email text is cleaned:
   - Lowercasing
   - Tokenization
   - Stopword & Punctuation removal
   - Stemming
2. Cleaned text is vectorized usingTF-ID
3. A Multinomial Naive Bayes model predicts spam or not spam
