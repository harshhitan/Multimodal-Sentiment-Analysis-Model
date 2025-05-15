# Multimodal Sentiment Analysis and Meme Understanding

This repository contains code and resources for a comprehensive project on sentiment analysis and meme understanding using both text and image data. The project covers fake news detection, movie review sentiment analysis with BERT, and multimodal meme sentiment classification, with deployment via a Streamlit web app.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Datasets](#datasets)
- [Methods](#methods)
- [Results](#results)
- [Deployment](#deployment)
- [Contributors](#contributors)

---

## Project Overview

The goal of this project is to develop and evaluate machine learning and deep learning models for sentiment analysis and meme classification using both textual and visual data. The work includes:
- Fake news detection using classical machine learning methods.
- Sentiment analysis of movie reviews using BERT.
- Multimodal meme sentiment analysis combining text and image features.

## Features

- **Fake News Detection:**  
  Text preprocessing and classification using Logistic Regression, Naive Bayes, and SVM, with performance comparison and confusion matrix visualization.

- **Movie Review Sentiment Analysis:**  
  Fine-tuned BERT model achieving 91% accuracy on sentiment classification, deployed as an interactive Streamlit web application.

- **Multimodal Meme Sentiment Analysis:**  
  Combined text and image features to classify memes from Twitter as positive, negative, or neutral, and predicted humour, sarcasm, and offensive scores.

## Datasets

- **Fake News Detection:**  
[Kaggle Fake News Dataset]

- **Movie Reviews:**  
  [IMDB Movie Reviews Dataset]

- **Meme Sentiment Analysis:**  
  [Twitter Meme Dataset with sentiment labels]

## Methods

- **Text Preprocessing:**  
  Tokenization, stopword removal, and feature extraction for classical ML models.

- **Classical Machine Learning:**  
  Logistic Regression, Naive Bayes, and SVM for fake news detection.

- **Deep Learning:**  
  Fine-tuned BERT for movie review sentiment analysis.

- **Multimodal Learning:**  
  Combined BERT embeddings for text with CNN-based image features for meme sentiment classification.

- **Evaluation:**  
  Used accuracy, F1-score, and confusion matrices to compare model performance.

## Results

- **Fake News Detection:**  
  Compared multiple classifiers; see notebooks for detailed performance metrics.

- **Movie Review Sentiment Analysis:**  
  Achieved 91% accuracy using BERT.

- **Meme Sentiment Analysis:**  
  Accurately classified meme sentiment and predicted humour, sarcasm, and offensive scores.

## Deployment

- The movie review sentiment analysis model is deployed as a [Streamlit](https://sentiment-classification-model-using-transformers.streamlit.app/) web application for interactive use. 
- [Watch the demo video on Google Drive](https://drive.google.com/file/d/1AGoqZmr7QuzmOkpS6cuIXlx8iaLlPjsi/view)

## Contributors

- [Harshit Anand](https://www.linkedin.com/in/harshit-anand-a33172284)  

