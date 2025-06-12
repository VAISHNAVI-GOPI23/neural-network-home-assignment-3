# 🧠 CS5720 - Home Assignment 3: Neural Networks and Deep Learning

**University of Central Missouri**  
**Department of Computer Science & Cybersecurity**  
**Course:** CS5720 Neural Networks and Deep Learning  
**Semester:** Summer 2025  

---

## 👤 Student Information

- **Name:** Vaishnavi Gopi
- **Student id** 700754518
- **Assignment:** Home Assignment 3  

---

## 📄 Assignment Overview

Topics covered include RNNs, NLP pipelines, Named Entity Recognition, Attention Mechanisms, and Sentiment Analysis using Transformers.

---

## 🔍 Program Summaries

### ✅ Q1: Text Generation using LSTM
- Trains an LSTM-based Recurrent Neural Network on a text dataset.
- Learns character sequences and predicts the next character in a sequence.
- Generates new text using a seed input.
- Temperature scaling controls the creativity/randomness of the generated output.

### ✅ Q2: NLP Preprocessing Pipeline
- Takes a sentence and processes it through three key NLP steps:
  - **Tokenization**: Splits the sentence into words and punctuation.
  - **Stopword Removal**: Removes common non-informative words (e.g., "the", "in").
  - **Stemming**: Reduces words to their root form (e.g., "running" → "run").
- Helps prepare clean and concise text for NLP models.

### ✅ Q3: Named Entity Recognition using spaCy
- Uses spaCy to extract named entities like people, places, and organizations from a sentence.
- Outputs each entity's label (e.g., PERSON, DATE) and its position in the text.
- Useful for extracting structured information from raw text (e.g., news, documents).

### ✅ Q4: Scaled Dot-Product Attention
- Implements the core attention mechanism used in Transformers.
- Calculates attention scores using Query (Q), Key (K), and Value (V) matrices.
- Applies scaling and softmax to produce attention weights.
- Outputs a weighted sum that helps the model focus on relevant parts of the input.

### ✅ Q5: Sentiment Analysis using HuggingFace Transformers
- Loads a pre-trained sentiment analysis model from the HuggingFace library.
- Analyzes the sentiment (positive/negative) of a given sentence.
- Prints both the sentiment label and the confidence score.
- Ideal for quickly understanding opinions in reviews, feedback, or social media.

---


