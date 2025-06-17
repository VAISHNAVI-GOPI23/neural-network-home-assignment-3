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
- 
# SHORT ANSWERS
---

## Q1: Implementing an RNN for Text Generation

**(No short answer questions were specified for Q1 in your prompt.)**

---

## Q2: NLP Preprocessing Pipeline - Short Answer Questions

### 1. What is the difference between stemming and lemmatization? Provide examples with the word “running.”

* **Stemming** is a crude heuristic process that chops off suffixes (and sometimes prefixes) from words, often resulting in a "root" form that is not a valid word itself. It's faster but less accurate.
    * **Example with "running":**
        * "running" $\rightarrow$ "run"
        * "runner" $\rightarrow$ "runner" (might not always be reduced further)
        * "ran" $\rightarrow$ "ran" (stemming might not handle irregular forms)
* **Lemmatization** is a more sophisticated process that uses vocabulary and morphological analysis to return the base or dictionary form of a word, known as a lemma. The lemma is always a valid word. It's slower but more accurate.
    * **Example with "running":**
        * "running" $\rightarrow$ "run"
        * "runner" $\rightarrow$ "runner"
        * "ran" $\rightarrow$ "run"

### 2. Why might removing stop words be useful in some NLP tasks, and when might it actually be harmful?

* **Why it's useful:**
    * **Reduces Dimensionality:** Stop words (like "the", "a", "is", "in") are extremely common but carry little semantic meaning for many NLP tasks. Removing them significantly reduces the number of unique words in the vocabulary, making models smaller and faster to train.
    * **Focus on Meaningful Words:** For tasks like text classification, information retrieval, or topic modeling, stop words can act as noise. Removing them allows the model to focus on the more content-bearing words that truly differentiate documents or topics.
    * **Improved Performance:** By reducing noise and focusing on important terms, models can sometimes achieve better accuracy or more relevant results.

* **When it might be harmful:**
    * **Sentiment Analysis:** Stop words can be crucial for conveying sentiment. For example, "not good" has a different meaning than "good". Removing "not" would alter the sentiment.
    * **Machine Translation:** The grammatical structure and meaning of a sentence often heavily rely on stop words. Removing them would break the sentence structure and lead to incorrect translations.
    * **Part-of-Speech Tagging or Dependency Parsing:** These tasks require understanding the grammatical role of every word, including stop words, to build correct syntactic trees.
    * **Question Answering:** Queries often contain stop words that are essential for understanding the nature of the question (e.g., "What *is* the capital of France?").
    * **Text Generation:** When generating text, stop words are necessary to create grammatically correct and natural-sounding sentences.

---

## Q3: Named Entity Recognition with SpaCy - Short Answer Questions

### 1. How does NER differ from POS tagging in NLP?

* **Part-of-Speech (POS) Tagging:**
    * **Purpose:** Identifies the grammatical category of each word in a sentence (e.g., noun, verb, adjective, adverb, pronoun).
    * **Output:** Assigns a single grammatical tag to each word.
    * **Example:** "Barack (NNP) Obama (NNP) served (VBD) as (IN) the (DT) 44th (JJ) President (NN)." (NNP: Proper Noun, VBD: Verb Past Tense, IN: Preposition, DT: Determiner, JJ: Adjective, NN: Noun)
* **Named Entity Recognition (NER):**
    * **Purpose:** Identifies and classifies "named entities" in text into pre-defined categories such as persons, organizations, locations, dates, monetary values, etc.
    * **Output:** Extracts spans of text (one or more words) that represent an entity and assigns a categorical label to that span.
    * **Example:** "[Barack Obama] (PERSON) served as the 44th President of [the United States] (GPE) and won the [Nobel Peace Prize] (EVENT) in [2009] (DATE)."

In essence, POS tagging focuses on the *grammatical role* of individual words, while NER focuses on identifying and categorizing *real-world objects or concepts* (which can span multiple words).

### 2. Describe two applications that use NER in the real world (e.g., financial news, search engines).

1.  **Information Extraction and Summarization:**
    * **Application:** Automatically extracting key information from large volumes of unstructured text, such as news articles, legal documents, or medical records.
    * **How NER is used:** NER can quickly identify all mentions of people, organizations, locations, dates, and events. For instance, in financial news, it can pinpoint company names, merger dates, and executives involved in a deal. This extracted information can then be used to populate databases, create structured summaries, or link related pieces of information.
    * **Example:** A system analyzing news about mergers and acquisitions could use NER to identify the acquiring company (ORG), the target company (ORG), the effective date (DATE), and the individuals involved (PERSON).

2.  **Customer Support and Chatbots:**
    * **Application:** Enhancing the ability of chatbots and virtual assistants to understand user queries and provide relevant responses.
    * **How NER is used:** When a user types a query like "What is the return policy for orders placed on Amazon last week?", NER can identify "Amazon" as an `ORG`, "last week" as a `DATE`, and "return policy" as a key concept. This allows the chatbot to route the query to the correct department, fetch relevant policy documents, or ask clarifying questions based on the identified entities.
    * **Example:** In a travel booking chatbot, NER helps identify entities like "London" (GPE), "July 15th" (DATE), "two adults" (CARDINAL), which are crucial for processing the booking request.

---

## Q4: Scaled Dot-Product Attention - Short Answer Questions

### 1. Why do we divide the attention score by $\sqrt{d_k}$ in the scaled dot-product attention formula?

We divide the dot product of Q and K by $\sqrt{d_k}$ (where $d_k$ is the dimension of the key vectors) to **prevent the dot products from growing too large**, especially when $d_k$ is large.

Here's why this is important:
* **Impact on Softmax:** The softmax function (which converts raw scores into probabilities/weights) is very sensitive to large input values. If the dot products become very large, the softmax function will push the gradients towards zero for most values, making them extremely small for all but the largest input. This leads to very sharp probability distributions (one hot vector), which can hinder the model's ability to learn and explore different attention weights during training (leading to vanishing gradients).
* **Stabilizing Gradients:** By scaling down the values, we keep the arguments to the softmax function in a more stable range, which helps to maintain more meaningful and diverse gradients, allowing for more effective learning.
* **Preventing Saturation:** It prevents the softmax function from "saturating" too quickly, ensuring that the attention weights remain well-distributed and that the model can learn to attend to multiple relevant parts of the input, rather than focusing too strongly on just one.

In essence, scaling helps to make the attention mechanism more numerically stable and prevents the attention weights from becoming too concentrated on a single item, especially in high-dimensional spaces.

### 2. How does self-attention help the model understand relationships between words in a sentence?

Self-attention allows the model to weigh the importance of all other words in the input sequence when processing each word. It helps the model understand relationships between words in a sentence in the following ways:

* **Capturing Long-Range Dependencies:** Unlike traditional RNNs that process sequentially, self-attention can directly connect any two words in a sequence, regardless of their distance. This is crucial for understanding dependencies where related words might be far apart (e.g., "The animal didn't cross the street because *it* was too tired." — *it* refers to *animal*).
* **Contextual Understanding:** For each word, self-attention calculates an "attention score" with every other word in the sentence. These scores determine how much focus the model should put on other words when encoding the current word. This means the representation of a word becomes a weighted sum of the representations of all other words, effectively encoding its context.
* **Handling Polysemy:** Words with multiple meanings (polysemy) can have their meaning clarified by the surrounding context. Self-attention allows the model to incorporate this context. For example, the meaning of "bank" in "river bank" versus "money bank" is differentiated by attending to "river" or "money".
* **Parallel Processing:** While not directly about understanding relationships, the ability to calculate attention in parallel for all words (unlike sequential processing in RNNs) significantly speeds up training and allows for the processing of much longer sequences.

By allowing each word to "look at" and "attend to" every other word in the sequence, self-attention creates rich, context-aware representations that implicitly capture complex semantic and syntactic relationships without relying on fixed-size windows or sequential memory.

---

## Q5: Sentiment Analysis using HuggingFace Transformers - Short Answer Questions

### 1. What is the main architectural difference between BERT and GPT? Which uses an encoder and which uses a decoder?

The main architectural difference between BERT (Bidirectional Encoder Representations from Transformers) and GPT (Generative Pre-trained Transformer) lies in their reliance on the Transformer architecture and their specific components:

* **BERT** uses an **encoder-only** architecture from the Transformer model. It is designed to understand the context of a word based on all other words in a sentence, both to its left and right (bidirectionally). This means it can "see" the entire sentence at once to build a rich representation of each word's meaning in context. Its pre-training tasks (Masked Language Model and Next Sentence Prediction) reflect this bidirectional understanding.
* **GPT** uses a **decoder-only** architecture from the Transformer model. It is designed for generative tasks and predicts the next word in a sequence based only on the preceding words (unidirectionally or autoregressively). It cannot "look ahead" to future tokens. Its pre-training primarily involves predicting the next token in a sequence.

### 2. Explain why using pre-trained models (like BERT or GPT) is beneficial for NLP applications instead of training from scratch.

Using pre-trained models like BERT or GPT offers several significant benefits for NLP applications compared to training models from scratch:

1.  **Reduced Training Time and Computational Resources:**
    * Training large NLP models from scratch (especially foundation models) requires immense computational power (hundreds of GPUs/TPUs), vast amounts of data, and considerable time (days to weeks or even months).
    * Pre-trained models have already undergone this intensive training, saving developers and researchers significant computational costs and time.

2.  **Access to Vast Linguistic Knowledge:**
    * Pre-trained models are typically trained on enormous and diverse text corpora (e.g., billions of words from books, articles, web pages like Wikipedia, Common Crawl).
    * This extensive pre-training allows them to learn a rich and general understanding of language, including grammar, syntax, semantic relationships, common facts, and even stylistic nuances.

3.  **Improved Performance (Transfer Learning):**
    * The process of using a pre-trained model and then adapting it to a specific, often smaller, task is known as **transfer learning**.
    * Pre-trained models provide excellent initial weights, serving as a powerful starting point. This enables them to achieve significantly higher accuracy and better generalization on new, unseen data, even when the task-specific dataset is relatively small. Training from scratch on a small dataset often leads to overfitting and poor performance due to insufficient data to learn complex linguistic patterns.

4.  **Faster Development and Prototyping:**
    * Developers can quickly integrate pre-trained models into their applications, fine-tune them for specific needs, and achieve functional prototypes much faster than if they had to build and train models from the ground up. This significantly accelerates the development cycle.

5.  **Accessibility:**
    * Pre-trained models make advanced NLP capabilities accessible to a wider range of users, including those without extensive machine learning expertise, large datasets, or vast computational resources. They democratize access to powerful AI.
  

   ### video link ##
   https://1drv.ms/v/c/f0028c921af54720/ERFKW6P2ZRdDutB4OypHnkwB17j9yQxtaxG2WPTQEdXbfg?e=OzCo0j
