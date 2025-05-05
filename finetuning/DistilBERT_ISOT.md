# Part 3: Fine-tuning DistilBERT for Fake News Detection

In this notebook, I'll build on our previous exploratory data analysis and feature engineering work to fine-tune a DistilBERT model for fake news detection. While our engineered features achieved impressive results, transformer models like DistilBERT can capture more complex linguistic patterns that might further improve performance or provide better generalization to new data.

## 1. Setup and Library Installation

First, I'll import the necessary libraries and install any missing packages.

```python
# Install required packages
!pip install transformers datasets torch evaluate scikit-learn
```

```python
# Import necessary libraries
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from datasets import Dataset as HFDataset
import evaluate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

## 2. Load and Prepare the Dataset

I'll load the preprocessed datasets from our previous work. If you're running this notebook independently, make sure you have the processed files from Part 2, or run the data preprocessing steps from the previous notebooks first.

```python
# Load the preprocessed datasets
try:
    train_df = pd.read_csv('train_fake_news.csv')
    val_df = pd.read_csv('val_fake_news.csv') 
    test_df = pd.read_csv('test_fake_news.csv')
    
    print(f"Training set: {train_df.shape}")
    print(f"Validation set: {val_df.shape}")
    print(f"Test set: {test_df.shape}")
except FileNotFoundError:
    print("Preprocessed files not found. Please run the data preprocessing from Part 2 first.")
```

Let's examine the data format to ensure it's what we expect:

```python
# Display sample data
print("Sample of training data:")
train_df.head(3)
```

Next, I'll convert our pandas DataFrames to the Hugging Face Dataset format, which is optimized for working with the transformers library:

```python
# Function to convert pandas DataFrames to HuggingFace Datasets
def convert_to_hf_dataset(df):
    # For DistilBERT, we'll use both title and text
    df['text'] = df['title'] + " " + df['enhanced_cleaned_text']
    
    # Convert to HuggingFace Dataset format
    dataset = HFDataset.from_pandas(df[['text', 'label']])
    return dataset

# Convert our datasets
train_dataset = convert_to_hf_dataset(train_df)
val_dataset = convert_to_hf_dataset(val_df)
test_dataset = convert_to_hf_dataset(test_df)

print(f"Training dataset: {len(train_dataset)} examples")
print(f"Validation dataset: {len(val_dataset)} examples")
print(f"Test dataset: {len(test_dataset)} examples")
```

## 3. Prepare Tokenizer and Model

Now I'll set up the DistilBERT tokenizer and model:

```python
# Initialize the DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Define the maximum sequence length
# Most news articles are quite long, but we need to balance information retention with computational efficiency
max_length = 512  # This is the maximum that BERT models can handle

# Function to tokenize the dataset
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

# Apply tokenization to our datasets
train_tokenized = train_dataset.map(tokenize_function, batched=True)
val_tokenized = val_dataset.map(tokenize_function, batched=True)
test_tokenized = test_dataset.map(tokenize_function, batched=True)

# Set the format for PyTorch
train_tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
val_tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
```

## 4. Define Metrics and Evaluation Strategy

I'll define our evaluation metrics to track model performance during training:

```python
# Function to compute metrics
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

Now I'll initialize the DistilBERT model for sequence classification:

```python
# Initialize the DistilBERT model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2  # Binary classification: 0 for fake, 1 for real
)

# Move model to device (GPU if available)
model.to(device)
```

## 6. Define Training Arguments and Trainer

Next, I'll configure the training parameters and create a Trainer:

```python
# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # Output directory for model checkpoints
    num_train_epochs=3,              # Number of training epochs
    per_device_train_batch_size=8,   # Batch size for training
    per_device_eval_batch_size=16,   # Batch size for evaluation
    warmup_steps=500,                # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # Strength of weight decay
    logging_dir='./logs',            # Directory for storing logs
    logging_steps=100,               # Log every X steps
    evaluation_strategy="epoch",     # Evaluate every epoch
    save_strategy="epoch",           # Save model checkpoint every epoch
    load_best_model_at_end=True,     # Load the best model at the end
    metric_for_best_model="f1",      # Use F1 score to determine the best model
    push_to_hub=False,               # Don't push to Hugging Face Hub
    report_to="none"                 # Disable reporting to avoid wandb or other services
)

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

## 7. Fine-tune the Model

Now I'll fine-tune the model:

```python
# Start the timer to measure training time
start_time = time.time()

# Train the model
trainer.train()

# Calculate training time
training_time = time.time() - start_time
print(f"Training completed in {training_time/60:.2f} minutes")

# Save the fine-tuned model
trainer.save_model("./distilbert-fake-news-detector")
```

## 8. Evaluate Model Performance

I'll evaluate the model on the test set:

```python
# Evaluate the model on the test set
test_results = trainer.evaluate(test_tokenized)
print(f"Test results: {test_results}")
```

Let's also look at the confusion matrix to get a better understanding of the errors:

```python
# Get predictions on the test set
test_pred = trainer.predict(test_tokenized)
y_preds = np.argmax(test_pred.predictions, axis=1)
y_true = test_pred.label_ids

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
plt.title('DistilBERT Confusion Matrix')
plt.savefig('distilbert_confusion_matrix.png')
plt.show()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_preds, target_names=['Fake News', 'Real News']))
```

## 9. Compare Results with Previous Models

Let's compare the DistilBERT results with our previous models:

```python
# Define the results from our previous models
previous_results = {
    'Engineered Features': {
        'accuracy': 0.9998,
        'precision': 1.0,
        'recall': 1.0,
        'f1': 1.0
    },
    'TF-IDF': {
        'accuracy': 0.984,
        'precision': 0.985,
        'recall': 0.984,
        'f1': 0.984
    }
}

# Add DistilBERT results
model_results = {
    'Engineered Features': previous_results['Engineered Features'],
    'TF-IDF': previous_results['TF-IDF'],
    'DistilBERT': {
        'accuracy': test_results['eval_accuracy'],
        'precision': test_results['eval_precision'],
        'recall': test_results['eval_recall'],
        'f1': test_results['eval_f1']
    }
}

# Convert to DataFrame for plotting
results_df = pd.DataFrame(model_results).T

# Create comparative bar chart
plt.figure(figsize=(12, 8))
results_df.plot(kind='bar', figsize=(12, 8))
plt.title('Model Performance Comparison')
plt.xlabel('Model')
plt.ylabel('Score')
plt.ylim(0.90, 1.01)  # Set y-axis to focus on the high performance range
plt.xticks(rotation=0)
plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()
```

## 10. Analyze Misclassified Examples

Let's analyze some misclassified examples to understand where the model struggles:

```python
# Get indices of misclassified examples
misclassified_indices = np.where(y_preds != y_true)[0]
print(f"Number of misclassified examples: {len(misclassified_indices)}")

# If there are misclassifications, analyze a few
if len(misclassified_indices) > 0:
    # Get the original text and predictions
    misclassified_texts = []
    for idx in misclassified_indices[:5]:  # Examine up to 5 examples
        original_idx = test_dataset[idx]['__index_level_0__'] if '__index_level_0__' in test_dataset[idx] else idx
        
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
    pd.DataFrame(misclassified_texts)
```

## 11. Use the Model for Predictions

Now I'll show how to use our fine-tuned model to make predictions on new data:

```python
# Function to make predictions on new texts
def predict_fake_news(texts, model, tokenizer, device):
    # Tokenize the texts
    inputs = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predicted class (0 for fake, 1 for real)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_classes = predictions.argmax(dim=1).cpu().numpy()
    confidence = predictions.max(dim=1).values.cpu().numpy()
    
    # Convert to human-readable labels
    results = []
    for i, pred_class in enumerate(pred_classes):
        label = "Real News" if pred_class == 1 else "Fake News"
        results.append({
            'text': texts[i],
            'prediction': label,
            'confidence': confidence[i],
            'class': pred_class
        })
    
    return pd.DataFrame(results)

# Example texts to test
sample_texts = [
    # Example of real news (from a reputable source)
    "Senate Passes Bipartisan Infrastructure Bill. The bill would provide funding for roads, bridges and other physical infrastructure.",
    
    # Example of fake news (made-up sensationalist headline)
    "BOMBSHELL: Government Admits Mind Control Program Targeting Citizens! Secret documents reveal shocking truth."
]

# Make predictions
sample_predictions = predict_fake_news(sample_texts, model, tokenizer, device)

# Display predictions
print("Predictions on sample texts:")
print(sample_predictions[['text', 'prediction', 'confidence']])
```

## 12. Conclusions and Next Steps

```python
# Print final summary and recommendations
conclusions = """
# Conclusions from DistilBERT Fine-tuning

In this notebook, I've fine-tuned a DistilBERT model for fake news detection on the ISOT dataset. Here are the key findings:

1. **Performance Comparison**: The DistilBERT model achieved [insert accuracy here] accuracy, which is [better/worse/comparable] to our previous models using engineered features (99.98%) and TF-IDF (98.4%).

2. **Training Efficiency**: Despite being more complex, DistilBERT is quite efficient for fine-tuning, with the process completing in approximately [insert time] minutes.

3. **Error Analysis**: Analysis of misclassified examples shows that DistilBERT struggles with [insert observations about errors].

4. **Generalization Potential**: Transformer models like DistilBERT likely have better generalization capabilities to new and unseen fake news, as they understand context and semantic meaning more deeply.

## Next Steps

1. **Experiment with Other Pretrained Models**: Try fine-tuning larger models like BERT-base or RoBERTa to see if they offer improvements.

2. **Combined Approach**: Develop an ensemble model that combines our engineered features with transformer-based features.

3. **External Validation**: Test the model on different fake news datasets to evaluate cross-dataset generalization.

4. **Model Explainability**: Implement techniques like LIME or SHAP to understand which parts of text the model relies on for classification.

5. **Deployment Considerations**: Optimize the model for inference time if it's to be used in a real-time application.

The transformer-based approach offers a powerful complement to our feature engineering work, potentially providing better generalization to evolving fake news tactics and new domains.
"""

print(conclusions)
```

This notebook provides a comprehensive approach to fine-tuning DistilBERT for fake news detection, building on our previous work of exploratory data analysis and feature engineering. The transformer-based approach captures complex linguistic patterns that may complement our engineered features and improve model robustness.
