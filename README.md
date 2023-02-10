# Fine-tuned-BERT-for-Sentiment-Analysis

This project incorporates Exploratory Data Analysis, XGBoost and Fine-tuned BERT approaches to classify emotion phrases in the Emotions NLP dataset.

## Dataset

The data attached to the project comes from the [Kaggle](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp?select=train.txt). It includes phrases and labels. Labels(emotions): anger, fear, joy, love, sadness, suprise.

## EDA

Exploratory Data Analysis at this point shows the distribution of classes, the number of words in phrases, the most frequent words for each emotion and the words with the highest cTfidf value for each emotion.

## XGBoost Classifier

The approach uses Tfidf statistics for classification. The hyperparameters were optimized using the [Optuna](https://optuna.org) library. The accuracy on the test dataset is about 87%.

## Fine-tunned BERT

The model in this approach contains a pre-trained BERT taken from [the Hugging Face transformers API](https://huggingface.co/docs/transformers/index). The accuracy on the test dataset is above 92%.
