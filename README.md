# Twitter NLP Sentiment Analysis
## Sentiment Analysis with DistilBERT
### Overview
This project utilizes the Sentiment140 dataset from Kaggle, which contains 1,600,000 tweets extracted using the Twitter API (https://www.kaggle.com/datasets/kazanova/sentiment140?resource=download). The aim to classify tweets into three sentiment categories: positive, negative, and neutral. The NLP model of choice is DistilBERT as it is a lightweight and efficient model that retains 97% of BERT's performance while being faster and requiring fewer resources, making it ideal for sentiment analysis. Its ability to understand language nuances and ease of fine-tuning ensures accurate and scalable results. 


### Dataset
The project uses a sample of 100,000 tweets derived from a larger dataset of 1,600,000 entries. Labels in the dataset represent the sentiment:

- 0 for negative

- 2 for neutral

- 4 for positive

These labels are remapped to:

- 0 for negative

- 1 for neutral

- 2 for positive

### Steps Undertaken
#### 1. Data Preprocessing
- Loaded and cleaned the dataset.

- Remapped sentiment labels for easier processing.
- Extracted a 100,000-sample subset for reduced training time.
#### 2. Dataset Splitting

- Split the subset into 80% training data and 20% testing data.
#### 3. Tokenization

- Used the pre-trained tokenizer distilbert-base-uncased to convert text into numerical format with padding and truncation.
#### 4. Model Fine-tuning

- Loaded a pre-trained DistilBERT model configured for sequence classification with three labels.
- Configured training parameters including batch size, learning rate, and number of epochs.
- Monitored performance using accuracy, precision, recall, and F1-score.
#### 5. Training

- Trained the model using the Hugging Face Trainer API.
- Saved the best-performing model based on validation metrics.
## Results
After training for two epochs, the model achieved:

- **Accuracy: ~83.6%**

- **F1 Score: ~83.6%**

- **Precision: ~83.7%**

- **Recall: ~83.6%**
### Tools and Libraries
- Python

- Pandas and NumPy

- Hugging Face transformers

- Scikit-learn for evaluation metrics

- Google Colab for execution

## Future Improvements
- Utilize the full dataset for training to enhance model generalization.
  
- Experiment with other pre-trained models such as BERT or RoBERTa.
  
- Implement hyperparameter tuning for optimal performance.
