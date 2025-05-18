# Fine-tuning MobileBERT for Fake News Detection

## Introduction

This notebook guides you through the process of fine-tuning a MobileBERT model for fake news detection using the ISOT dataset. MobileBERT is particularly well-suited for deployment on mobile and edge devices due to its compact architecture and efficiency optimizations. Unlike larger models like BERT or RoBERTa, MobileBERT was specifically designed to balance performance with computational constraints, making it ideal for resource-limited environments.

In this notebook, we will:
1. Set up the necessary environment and libraries
2. Load and prepare the ISOT dataset for training
3. Configure MobileBERT for sequence classification
4. Fine-tune the model with carefully selected hyperparameters
5. Evaluate performance and analyze results
6. Save the model for future use or deployment

## 1. Setup and Environment Preparation

First, let's install and import all necessary libraries:

```python
# Install required packages
!pip install transformers datasets torch evaluate scikit-learn
```

Now let's import all the libraries we'll need for our fine-tuning process:

```python
# Import necessary libraries
import numpy as np
import pandas as pd
import torch
import random
import time
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from datasets import Dataset as HFDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import warnings
warnings.filterwarnings('ignore')
```

Let's set up reproducibility by setting random seeds:

```python
# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
```

Check for GPU availability to accelerate training:

```python
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

## 2. Data Preparation

### Loading the Dataset

Now we'll load the preprocessed ISOT dataset:

```python
# Load the preprocessed datasets
try:
    train_df = pd.read_csv('/kaggle/input/isot-fake-news-robust/train_fake_news_robust.csv')
    val_df = pd.read_csv('/kaggle/input/isot-fake-news-robust/val_fake_news_robust.csv') 
    test_df = pd.read_csv('/kaggle/input/isot-fake-news-robust/test_fake_news_robust.csv')
    
    print(f"Training set: {train_df.shape}")
    print(f"Validation set: {val_df.shape}")
    print(f"Test set: {test_df.shape}")
except FileNotFoundError:
    print("Preprocessed files not found. Please run the data preprocessing step first.")
```

Let's examine the data structure:

```python
# Display sample data
print("Sample of training data:")
train_df.head(3)
```

### Converting to HuggingFace Dataset Format

Now we'll convert our pandas DataFrames to the HuggingFace Dataset format, which is optimized for working with transformer models:

```python
# Function to convert pandas DataFrames to HuggingFace Datasets
def convert_to_hf_dataset(df):
    # For MobileBERT, we'll use both title and text
    df['text'] = df['title'] + " " + df['enhanced_cleaned_text']
    
    # Convert to HuggingFace Dataset format
    dataset = HFDataset.from_pandas(df[['text', 'label']])
    return dataset
```

Apply the conversion function:

```python
# Convert our datasets
train_dataset = convert_to_hf_dataset(train_df)
val_dataset = convert_to_hf_dataset(val_df)
test_dataset = convert_to_hf_dataset(test_df)

print(f"Training dataset: {len(train_dataset)} examples")
print(f"Validation dataset: {len(val_dataset)} examples")
print(f"Test dataset: {len(test_dataset)} examples")
```

We combine the title and body text into a single text field because:
1. News headlines often contain important contextual information that can help identify fake news
2. MobileBERT can process sequences up to 512 tokens, which is sufficient for most news articles
3. This approach provides the model with the maximum available information for classification

### Data Cleaning and Preparation

Before tokenization, we ensure the dataset is clean and properly formatted:

```python
# Check first few examples in your dataset
print("First example in train_dataset:", train_dataset[0])

# Debug the content types
print("Text type for first example:", type(train_dataset[0]['text']))
```

Define a cleaning function:

```python
# Define a cleaning function for the dataset
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

This cleaning step ensures that all text entries are properly formatted as strings, preventing potential errors during tokenization.

## 3. Model Architecture and Configuration

### Tokenization

Let's prepare the tokenizer for MobileBERT:

```python
# Initialize the MobileBERT tokenizer
tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')

# Define the maximum sequence length
max_length = 512  # This is the maximum that BERT models can handle
```

Define the tokenization function:

```python
# Function to tokenize the dataset
def tokenize_function(examples):
    # Convert all text entries to strings and handle potential None values
    texts = [str(text) if text is not None else "" for text in examples['text']]
    
    return tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors=None  # Don't return tensors in batch mode
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
train_tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
val_tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
```

Key tokenization decisions:
- We use the uncased version of MobileBERT because case information is less critical for fake news detection
- We set `max_length=512` to use the full context window of MobileBERT
- We apply padding to ensure all sequences have the same length, which is necessary for batch processing
- We use truncation to handle any articles that exceed the maximum length
- We use batched processing for efficiency

### Model Initialization

Now we initialize the MobileBERT model for sequence classification:

```python
# Initialize the MobileBERT model for sequence classification
model = MobileBertForSequenceClassification.from_pretrained(
    'google/mobilebert-uncased',
    num_labels=2  # Binary classification: 0 for fake, 1 for real
)

# Move model to device (GPU if available)
model.to(device)
```

We use the pretrained MobileBERT model and adapt it for our binary classification task. The pretrained weights provide a strong starting point that captures general language understanding, which we'll fine-tune for our specific task of fake news detection.

## 4. Training Process

### Defining Metrics

Let's define a function to compute evaluation metrics during training:

```python
# Function to compute metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
```

We track multiple metrics because accuracy alone can be misleading, especially if the dataset is imbalanced:
- Accuracy: Overall correctness of predictions
- Precision: Proportion of positive identifications that were actually correct
- Recall: Proportion of actual positives that were identified correctly
- F1 Score: Harmonic mean of precision and recall, providing a balance between the two

### Training Configuration

Let's set up the training arguments with carefully chosen hyperparameters:

```python
# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # Output directory for model checkpoints
    num_train_epochs=5,              # Number of training epochs
    per_device_train_batch_size=16,  # Batch size for training - MobileBERT is efficient
    per_device_eval_batch_size=32,   # Batch size for evaluation
    warmup_steps=500,                # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # Strength of weight decay
    logging_dir='./logs',            # Directory for storing logs
    logging_steps=100,               # Log every X steps
    eval_strategy="epoch",           # Evaluate every epoch
    save_strategy="epoch",           # Save model checkpoint every epoch
    load_best_model_at_end=True,     # Load the best model at the end
    metric_for_best_model="f1",      # Use F1 score to determine the best model
    push_to_hub=False,               # Don't push to Hugging Face Hub
    report_to="none",                # Disable reporting to avoid wandb or other services
    learning_rate=2e-5               # Learning rate
)
```

Key hyperparameter choices and their rationale:
- `num_train_epochs=5`: Provides sufficient training iterations while avoiding overfitting
- `per_device_train_batch_size=16`: Balances memory constraints with training efficiency
- `warmup_steps=500`: Gradually increases the learning rate to stabilize early training
- `weight_decay=0.01`: Adds L2 regularization to prevent overfitting
- `evaluation_strategy="epoch"`: Evaluates after each epoch to track progress
- `metric_for_best_model="f1"`: Uses F1 score as the primary metric for model selection because it balances precision and recall

### Training Execution

Let's initialize the Trainer:

```python
# Create the Trainer
trainer = Trainer(
    model=model,                         # The instantiated model to train
    args=training_args,                  # Training arguments
    train_dataset=train_tokenized,       # Training dataset
    eval_dataset=val_tokenized,          # Evaluation dataset
    compute_metrics=compute_metrics,     # The function to compute metrics
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Early stopping
)
```

Now, let's train the model with the option to resume from a checkpoint if needed:

```python
# Start the timer to measure training time
start_time = time.time()

# Train the model (with option to resume from checkpoint)
# To resume from a checkpoint, use: trainer.train(resume_from_checkpoint=True)
trainer.train()

# Calculate training time
training_time = time.time() - start_time
print(f"Training completed in {training_time/60:.2f} minutes")
```

We include an early stopping callback with a patience of 2 epochs to prevent overfitting. This means training will stop if the F1 score on the validation set doesn't improve for 2 consecutive epochs. This is particularly important for MobileBERT, which might be more prone to overfitting due to its more limited capacity.

## 5. Evaluation

### Model Evaluation

Now let's evaluate the model on the test set:

```python
# Evaluate the model on the test set
test_results = trainer.evaluate(test_tokenized)
print(f"Test results: {test_results}")
```

### Detailed Performance Analysis

Let's perform a more detailed analysis of the model's predictions:

```python
# Get predictions on the test set
test_pred = trainer.predict(test_tokenized)
y_preds = np.argmax(test_pred.predictions, axis=1)
y_true = test_pred.label_ids
```

Create and visualize the confusion matrix:

```python
# Create confusion matrix
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_true, y_preds)
print("Confusion Matrix:")
print(cm)

# Plot confusion matrix
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
# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_preds, target_names=['Fake News', 'Real News']))
```

## 6. Error Analysis and Results Summary

Let's analyze the misclassified examples to understand where the model struggles:

```python
# Get indices of misclassified examples
misclassified_indices = np.where(y_preds != y_true)[0]
print(f"Number of misclassified examples: {len(misclassified_indices)}")

# If there are misclassifications, analyze a few
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

Let's also create a comparison with other models if you have those results available:

```python
# Create a comparison table with previous models
models = ['TF-IDF + ML', 'DistilBERT', 'TinyBERT', 'MobileBERT']
accuracy = [0.984, 0.9996, 0.9991, test_results['eval_accuracy']] 
f1_scores = [0.984, 0.9996, 0.9991, test_results['eval_f1']]
training_times = ['0.13 minutes', '48.69 minutes', '8.99 minutes', f"{training_time/60:.2f} minutes"]

comparison_df = pd.DataFrame({
    'Model': models,
    'Accuracy': accuracy,
    'F1 Score': f1_scores,
    'Training Time': training_times
})

print("Model Performance Comparison:")
display(comparison_df)
```

## 7. Saving the Model

Finally, let's save the fine-tuned model for future use:

```python
# Save the fine-tuned model
model_save_path = "./mobilebert-fake-news-detector"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model and tokenizer saved to {model_save_path}")
```

## 8. Conclusion

MobileBERT offers several advantages for fake news detection:

1. **Efficiency:** MobileBERT's bottleneck architecture significantly reduces model size and improves inference speed, making it suitable for resource-constrained environments.

2. **Performance:** Despite its optimizations for efficiency, MobileBERT maintains competitive performance compared to larger models like DistilBERT and RoBERTa.

3. **Mobile Deployment:** The model's architecture was specifically designed for mobile applications, making it ideal for on-device fake news detection.

4. **Memory Footprint:** With approximately 25M parameters (compared to BERT's 110M or RoBERTa's 125M), MobileBERT requires significantly less memory while preserving most of the capability.

This notebook demonstrated how to fine-tune MobileBERT for fake news detection, achieving excellent performance with reasonable computational requirements. The model can now be deployed in various scenarios, particularly those with resource constraints where larger models would be impractical.

In future work, you might consider:
1. Exploring additional optimization techniques like quantization for even greater efficiency
2. Testing the model on more diverse fake news datasets to evaluate generalization
3. Implementing the model in a mobile application to demonstrate real-world deployment
4. Comparing inference latency across different mobile devices
