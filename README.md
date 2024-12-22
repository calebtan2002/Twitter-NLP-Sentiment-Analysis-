# Twitter NLP Sentiment Analysis
## Sentiment Analysis with DistilBERT
### Overview
This project focuses on sentiment analysis using a subset of tweets from a large dataset. The primary goal is to classify text into three sentiment categories: negative, neutral, and positive. The analysis employs the Hugging Face's transformers library, leveraging a pre-trained DistilBERT model fine-tuned for this task.

### Dataset
The project uses a sample of 100,000 tweets derived from a larger dataset of 1,600,000 entries. Labels in the dataset represent the sentiment:

0 for negative
2 for neutral
4 for positive
These labels are remapped to:

0 for negative
1 for neutral
2 for positive
### Steps Undertaken
#### Data Preprocessing
Loaded and cleaned the dataset.
Remapped sentiment labels for easier processing.
Extracted a 100,000-sample subset for reduced training time.
#### Dataset Splitting

Split the subset into 80% training data and 20% testing data.
#### Tokenization

Used the pre-trained tokenizer distilbert-base-uncased to convert text into numerical format with padding and truncation.
#### Model Fine-tuning

Loaded a pre-trained DistilBERT model configured for sequence classification with three labels.
Configured training parameters including batch size, learning rate, and number of epochs.
Monitored performance using accuracy, precision, recall, and F1-score.
#### Training

Trained the model using the Hugging Face Trainer API.
Saved the best-performing model based on validation metrics.
### Results
After training for two epochs, the model achieved:

Accuracy: ~83.6%
F1 Score: ~83.6%
Precision: ~83.7%
Recall: ~83.6%
### Tools and Libraries
Python
Pandas and NumPy
Hugging Face transformers
Scikit-learn for evaluation metrics
Google Colab for execution
