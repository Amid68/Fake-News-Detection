# Enhanced Documentation for MobileBERT Fine-tuning on ISOT Dataset

In this notebook, I'll build on our previous exploratory data analysis and feature engineering work to fine-tune a MobileBERT model for fake news detection. While our engineered features achieved impressive results, transformer models can capture complex linguistic patterns that might further improve performance or provide better generalization to new data. MobileBERT is specifically designed for mobile applications, offering a better trade-off between model size, inference speed, and accuracy compared to larger models like BERT or RoBERTa.

## 1. Setup and Library Installation

First, I'll install the required packages:


```python
# Install required packages
!pip install transformers datasets torch evaluate scikit-learn
```

Now, let's import the basic libraries:


```python
# Import basic libraries
import numpy as np
import pandas as pd
import torch
import random
import time
import os
import warnings
warnings.filterwarnings('ignore')
```

Import the transformer-specific libraries:


```python
# Import transformer-specific libraries
from torch.utils.data import Dataset, DataLoader
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from datasets import Dataset as HFDataset
```

Import evaluation libraries:


```python
# Import evaluation libraries
import evaluate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
```

Set up reproducibility and check for GPU availability:


```python
# Set random seeds for reproducibility
# This ensures that our experiments can be replicated with the same results
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Check if GPU is available
# MobileBERT can run efficiently on CPU, but GPU will significantly speed up training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

## 2. Load and Prepare the Dataset

Load the preprocessed datasets:


```python
# Load the preprocessed datasets
# These datasets have already been cleaned and split in our previous data preprocessing steps
# The preprocessing included removing the "(Reuters)" pattern to prevent data leakage
try:
    train_df = pd.read_csv('/kaggle/input/isot-processed-and-splitted/train_fake_news.csv')
    val_df = pd.read_csv('/kaggle/input/isot-processed-and-splitted/val_fake_news.csv') 
    test_df = pd.read_csv('/kaggle/input/isot-processed-and-splitted/test_fake_news.csv')
    
    print(f"Training set: {train_df.shape}")
    print(f"Validation set: {val_df.shape}")
    print(f"Test set: {test_df.shape}")
except FileNotFoundError:
    print("Preprocessed files not found. Please run the data preprocessing from Part 2 first.")
```

Examine the data format:


```python
# Display sample data to understand the structure
print("Sample of training data:")
train_df.head(3)
```

Define a function to convert pandas DataFrames to HuggingFace Datasets:


```python
# Function to convert pandas DataFrames to HuggingFace Datasets
# This is necessary because the Transformers library works best with HuggingFace Datasets
def convert_to_hf_dataset(df):
    # For MobileBERT, we'll use both title and text
    # Combining title and text provides more context for the model
    df['text'] = df['title'] + " " + df['enhanced_cleaned_text']
    
    # Convert to HuggingFace Dataset format
    dataset = HFDataset.from_pandas(df[['text', 'label']])
    return dataset
```

Apply the conversion function:


```python
# Convert our datasets to HuggingFace format
train_dataset = convert_to_hf_dataset(train_df)
val_dataset = convert_to_hf_dataset(val_df)
test_dataset = convert_to_hf_dataset(test_df)

print(f"Training dataset: {len(train_dataset)} examples")
print(f"Validation dataset: {len(val_dataset)} examples")
print(f"Test dataset: {len(test_dataset)} examples")
```

## 3. Prepare Tokenizer and Model

Check data format and types:


```python
# Check first few examples in your dataset to ensure proper formatting
print("First example in train_dataset:", train_dataset[0])

# Debug the content types to catch any potential issues
print("Text type for first example:", type(train_dataset[0]['text']))
```

Define a cleaning function:


```python
# Define a cleaning function for the dataset
# This ensures all text entries are properly formatted strings
def clean_dataset(example):
    example['text'] = str(example['text']) if example['text'] is not None else ""
    return example
```

Apply cleaning to the datasets:


```python
# Apply cleaning to all datasets
train_dataset = train_dataset.map(clean_dataset)
val_dataset = val_dataset.map(clean_dataset)
test_dataset = test_dataset.map(clean_dataset)
```

Initialize the MobileBERT tokenizer:


```python
# Initialize the MobileBERT tokenizer
# We use the uncased version as case is typically not important for fake news detection
tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')

# Define the maximum sequence length
# 512 is the maximum that BERT models can handle
max_length = 512
```

Define the tokenization function:


```python
# Function to tokenize the dataset
# This converts text into the numerical format that the model can process
def tokenize_function(examples):
    # Convert all text entries to strings and handle potential None values
    texts = [str(text) if text is not None else "" for text in examples['text']]
    
    return tokenizer(
        texts,
        padding='max_length',  # Pad to max_length to create uniform batch sizes
        truncation=True,       # Truncate texts longer than max_length
        max_length=max_length,
        return_tensors=None    # Don't return tensors in batch mode
    )
```

Apply tokenization to our datasets:


```python
# Apply tokenization to our datasets
train_tokenized = train_dataset.map(tokenize_function, batched=True)
val_tokenized = val_dataset.map(tokenize_function, batched=True)
test_tokenized = test_dataset.map(tokenize_function, batched=True)
```

Set the format for PyTorch:


```python
# Set the format for PyTorch after tokenization
# This ensures compatibility with the PyTorch-based Transformers library
train_tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
val_tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
```

## 4. Define Metrics and Evaluation Strategy

Define our evaluation metrics:


```python
# Function to compute metrics
# This will be used to evaluate model performance during and after training
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
```

## 5. Initialize Model for Fine-tuning

Initialize the MobileBERT model:


```python
# Initialize the MobileBERT model for sequence classification
# We use the pre-trained model and add a classification head for our binary task
model = MobileBertForSequenceClassification.from_pretrained(
    'google/mobilebert-uncased',
    num_labels=2  # Binary classification: 0 for fake, 1 for real
)
```

Move the model to the appropriate device:


```python
# Move model to device (GPU if available)
# This significantly speeds up training if a GPU is available
model.to(device)
```

## 6. Define Training Arguments and Trainer

Configure the training parameters:


```python
# Define training arguments
# These hyperparameters were selected based on empirical testing and literature recommendations
training_args = TrainingArguments(
    output_dir='./results',          # Output directory for model checkpoints
    num_train_epochs=3,              # Number of training epochs - 3 is typically sufficient for this task
    per_device_train_batch_size=16,  # Batch size for training - MobileBERT is efficient enough for larger batches
    per_device_eval_batch_size=32,   # Batch size for evaluation
    warmup_steps=500,                # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # Strength of weight decay for regularization
    logging_dir='./logs',            # Directory for storing logs
    logging_steps=100,               # Log every X steps
    eval_strategy="epoch",           # Evaluate every epoch
    save_strategy="epoch",           # Save model checkpoint every epoch
    load_best_model_at_end=True,     # Load the best model at the end of training
    metric_for_best_model="f1",      # Use F1 score to determine the best model
    push_to_hub=False,               # Don't push to Hugging Face Hub
    report_to="none",                # Disable reporting to avoid wandb or other services
    learning_rate=2e-5               # Learning rate - 2e-5 is a common value for fine-tuning transformers
)
```

Create the Trainer:


```python
# Create the Trainer
# The Trainer handles the training loop, evaluation, and early stopping
trainer = Trainer(
    model=model,                         # The instantiated model to train
    args=training_args,                  # Training arguments
    train_dataset=train_tokenized,       # Training dataset
    eval_dataset=val_tokenized,          # Evaluation dataset
    compute_metrics=compute_metrics,     # The function to compute metrics
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Early stopping to prevent overfitting
)
```

## 7. Fine-tune the Model

Start the timer to measure training time:


```python
# Start the timer to measure training time
# This helps us compare efficiency across different models
start_time = time.time()
```

Train the model:


```python
# Train the model
# This will fine-tune the pre-trained MobileBERT on our fake news dataset
trainer.train()
```

Calculate and display the training time:


```python
# Calculate training time
training_time = time.time() - start_time
print(f"Training completed in {training_time/60:.2f} minutes")
```

Save the fine-tuned model:


```python
# Save the fine-tuned model for later use
trainer.save_model("./mobilebert-fake-news-detector")
```

## 8. Evaluate Model Performance

Evaluate the model on the test set:


```python
# Evaluate the model on the test set
test_results = trainer.evaluate(test_tokenized)
print(f"Test results: {test_results}")
```

Get predictions on the test set:


```python
# Get predictions on the test set
test_pred = trainer.predict(test_tokenized)
y_preds = np.argmax(test_pred.predictions, axis=1)
y_true = test_pred.label_ids
```

Create confusion matrix:


```python
# Create confusion matrix to visualize model performance
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_true, y_preds)
print("Confusion Matrix:")
print(cm)
```

Plot the confusion matrix:


```python
# Plot confusion matrix for better visualization
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('MobileBERT Confusion Matrix')
plt.savefig('mobilebert_confusion_matrix.png')
plt.show()
```

Print the classification report:


```python
# Print classification report for detailed performance metrics
print("\nClassification Report:")
print(classification_report(y_true, y_preds, target_names=['Fake News', 'Real News']))
```

## 9. Analyze Misclassified Examples

Find and count misclassified examples:


```python
# Get indices of misclassified examples
misclassified_indices = np.where(y_preds != y_true)[0]
print(f"Number of misclassified examples: {len(misclassified_indices)}")
```

Analyze misclassified examples if any exist:


```python
# If there are misclassifications, analyze a few to understand model limitations
if len(misclassified_indices) > 0:
    # Get the original text and predictions
    misclassified_texts = []
    for idx in misclassified_indices[:5]:  # Examine up to 5 examples
        # Convert numpy.int64 to Python int
        idx_int = int(idx)
        
        # Now use the converted index
        original_idx = test_dataset[idx_int]['__index_level_0__'] if '__index_level_0__' in test_dataset[idx_int] else idx_int
        
        text = test_df.iloc[original_idx]['title']
        true_label = "Real" if y_true[idx] == 1 else "Fake"
        pred_label = "Real" if y_preds[idx] == 1 else "Fake"
        
        misclassified_texts.append({
            'Title': text,
            'True Label': true_label,
            'Predicted Label': pred_label
        })
    
    # Display misclassified examples
    print("\nSample of misclassified examples:")
    display(pd.DataFrame(misclassified_texts))
```

## 10. Model Performance Comparison and Conclusions

Create a comparison table with previous models:


```python
# Create a comparison table of model performances
models = ['TF-IDF + ML', 'DistilBERT', 'TinyBERT', 'MobileBERT']
accuracy = [0.984, 0.9996, 0.9991, test_results['eval_accuracy']] 
f1_scores = [0.984, 0.9996, 0.9991, test_results['eval_f1']]
training_times = ['39.18 minutes', '48.69 minutes', '8.99 minutes', f"{training_time/60:.2f} minutes"]

comparison_df = pd.DataFrame({
    'Model': models,
    'Accuracy': accuracy,
    'F1 Score': f1_scores,
    'Training Time': training_times
})

print("Model Performance Comparison:")
display(comparison_df)
```

## Conclusion and Discussion

In this notebook, I've fine-tuned a MobileBERT model for fake news detection on the ISOT dataset. Here are the key findings and insights:

### Performance Analysis

MobileBERT achieves excellent accuracy, comparable to our previous models using engineered features, DistilBERT, and TinyBERT. This demonstrates that lightweight transformer models can maintain high performance while requiring fewer computational resources.

### Efficiency Considerations

MobileBERT is specifically designed for mobile and edge devices, offering a good balance between model size, inference speed, and accuracy. With approximately 25M parameters (compared to BERT's 110M and DistilBERT's 67M), it's significantly more efficient while maintaining strong performance.

### Practical Applications

The high accuracy combined with MobileBERT's efficiency makes it suitable for deployment in resource-constrained environments like mobile applications or edge devices. This enables real-time fake news detection without requiring powerful hardware.

### Limitations and Future Work

While the model performs exceptionally well on the ISOT dataset, real-world deployment would benefit from:

1. Testing on more diverse datasets to ensure generalization
2. Implementing explainability techniques to understand model decisions
3. Exploring quantization and pruning for further efficiency improvements
4. Developing ensemble approaches combining traditional ML and transformer models

This work demonstrates that lightweight transformer models like MobileBERT can effectively detect fake news with high accuracy while maintaining reasonable computational requirements, making them practical for real-world applications.

