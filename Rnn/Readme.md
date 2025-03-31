# Reviews Sentiment analysis using Recurrent Neural Network

Group number: 23
Contributors:
- Amrit Agarwal (055004)
- Oishik Banerjee (055028)

---
## Objective

To design, implement, and evaluate a deep learning-based sentiment analysis model using RNN architecture. This model aims to classify movie reviews based on sentiment by leveraging the sequential patterns present in text data.

---

# **Dataset Description**

## **Overview**
The dataset contains movie reviews with sentiment labels, structured as follows:

### **Columns:**
1. **Movie Name** – The title of the movie being reviewed.
2. **Rating** – A numerical rating given by the reviewer (ranges from 1 to 10).
3. **Review** – The text-based review of the movie.
4. **Sentiment** – The sentiment classification of the review:
   - **Positive**: The review expresses a favorable opinion.
   - **Negative**: The review expresses an unfavorable opinion.

## **Key Characteristics**
- The dataset includes reviews for multiple movies, such as *La La Land, Pinocchio, and RRR*.
- Reviews vary in length and writing style, providing diverse textual data.
- The sentiment column acts as the **target variable** for classification.

### **Shape Of Data** - (50000,2)
### **Size of Data** - 64,477 kb
### **Source of Data** - Kaggle 
### **Link to Data** - https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?datasetId=134715&sortBy=dateRun&tab=profile
---

## Problem Statement

- Online movie reviews significantly influence public opinion.
- Classifying sentiment is challenging due to language complexity.
- The goal is to develop a machine learning model for sentiment analysis.
- An RNN-based approach will be used to capture contextual information.
- The model will classify reviews as positive, negative, or neutral.

---

## Key Tasks

### 1. Data Preprocessing

The data preprocessing stage prepares the movie review dataset to ensure compatibility with the RNN model. The key steps are:

#### **I) Sentiment Encoding**
- Positive Sentiment → Encoded as **1**
- Negative Sentiment → Encoded as **0**

#### **II) Text Normalization**
- **Removing Special Characters:** Stripping unnecessary characters (e.g., punctuation, special symbols) to clean the text.
- **Lowercasing:** Converting all reviews to lowercase for uniformity and consistency.

#### **III) Tokenization**
- Splitting the text into individual tokens (words).
- Using a vocabulary size of **20,000** most frequent words (`max_features=20000`). Any words outside this range are replaced with a placeholder token.

#### **IV) Sequence Padding**
- Ensuring all tokenized reviews are of the same length by:
  - Padding shorter sequences with zeros at the beginning or end.
  - Truncating longer sequences to a maximum length of **400** tokens (`max_length = 400`).

---

### 2. Model Development

To build a model that classifies movie reviews as positive or negative, we follow these steps:

#### **I) Using the Data**

**Training Data:**
- The training data consists of **50,000** records with the following **2 columns**:
  - **Reviews:** The textual review of the movie.
  - **Sentiment:** The sentiment label (positive or negative).
- A random sample of **40,000** reviews is selected using a random state of **xxxx** to ensure reproducibility.

**Dataset link:** IMDB Dataset of 50K Movie Reviews

**Testing Data:**
- The testing data consists of **151** records with the following **4 columns**:
  - **Movie Name:** The title of the movie.
  - **Rating:** The rating given to the movie.
  - **Reviews:** The textual review of the movie.
  - **Sentiment:** The sentiment label (positive or negative).
- This dataset was created by manually scraping the data on reviews and ratings of various movies from **Metacritic**.

#### **II) Model Structure**

The model is built step by step with these layers:

##### **Embedding Layer**
- **Input dimension:** 20,000 (vocabulary size)
- **Output dimension:** 128 (word embedding size)
- **Input length:** 400 (maximum sequence length)

##### **Recurrent Layer**
- **Type:** SimpleRNN
- **Number of units:** 64
- **Activation function:** Tanh
- **Return sequences:** False (since it’s a single RNN layer)
- **Regularization:** Dropout (0.2) to prevent overfitting

##### **Fully Connected Layer**
- **Type:** Dense layer
- **Number of neurons:** 1
- **Activation function:** Sigmoid (for binary classification)

#### **III) Training the Model**

The model is trained on IMDB reviews by splitting the sampled dataset of **40,000** reviews into **80%** for training and **20%** for testing, ensuring the model learns effectively while being evaluated on unseen data during training.

**Model Compilation and Training:**

- **Loss Function:** Binary Crossentropy (suitable for binary classification)
- **Optimizer:** Adam (learning rate = 0.001)
- **Batch Size:** 32
- **Epochs:** 15 (With early stopping)

**Early Stopping Criteria:**
- **Monitored metric:** Validation Loss
- **Patience:** 3 epochs
- **Best weights restored** if validation loss does not improve
- The model was trained for **10** epochs initially and then for an additional **5** epochs.

#### **IV) Testing the Model with Metacritic Data**
- After training on IMDB reviews, the model is tested on the **100 manually collected Metacritic reviews**.
- Performed data preprocessing, tokenization, and sequence padding as performed with the training dataset.

#### **V) Predicting Sentiment for New Reviews**
Once trained, the model can predict whether new reviews are **positive or negative**.

---

## 3. Observations

- **Training accuracy** increased steadily, reaching approximately **89%** after **10 epochs**.
- **Validation accuracy** remained stable at around **87%**, indicating good generalization.
- The final **test accuracy** on the IMDB test set was around **86%**, suggesting a well-trained model with slight room for improvement.
- Training loss started to decrease significantly with every epoch, with potential signs of overfitting mitigated by **early stopping and dropout**.
- The model performed similarly on the Metacritic dataset, achieving a **test accuracy of approximately 77%**, showing that it generalizes well across different review datasets but could improve if **LSTM was used instead of RNN**.
- Early stopping was triggered after a few epochs in both training phases, preventing overfitting and ensuring that the best model was retained.

---

## 4. Managerial Insights

### **Model Effectiveness & Business Implications**
- The RNN model performs well on the IMDB dataset but generalizes **poorly** on Metacritic reviews.
- This suggests that Metacritic reviews might have **different writing styles, slang, or review structures** compared to IMDB.

### **Improvement Areas**
- **Better Preprocessing:** Introduce techniques like stemming, lemmatization, stop-word removal, and n-grams to improve accuracy.
- **More Complex Architectures:** RNNs have limited long-term memory; switching to **LSTMs** may enhance generalization.
- **Larger Dataset & Augmentation:** Training on a combined dataset of IMDB and Metacritic reviews may improve model robustness.
- **Domain Adaptation:** Fine-tuning the model specifically on Metacritic reviews could improve cross-domain accuracy.

### **Business Applications**
- **Customer Sentiment Monitoring:** Companies can use this model to analyze movie, product, or service reviews to gauge public opinion.
- **Brand Reputation Analysis:** Identifying sentiment trends can help businesses manage PR crises and improve customer engagement.
- **Automated Review Filtering:** Businesses can filter out fake reviews or spam using an improved sentiment classification model.

---

## 5. Conclusion & Recommendations

### **Immediate Steps:**
- Improve text preprocessing by handling stop words and using TF-IDF weights.
- Fine-tune the model using **transfer learning** with additional datasets.
- Consider switching to **LSTM/GRU-based models** for improved generalization.

### **Long-Term Strategy:**
- Expand training data by incorporating reviews from multiple platforms.
- Implement **real-time sentiment tracking** in a dashboard for actionable insights.
- Conduct **A/B testing** with different architectures to find the best-performing model.

By implementing these recommendations, the sentiment analysis model can achieve **higher accuracy (target: 75%+)** and be effectively deployed for business use cases.
