# MobileBERT for Fake News Detection

## 1. Introduction

This notebook demonstrates how to finetune MobileBERT for detecting fake news using the WELFake dataset. MobileBERT is an efficient and lightweight transformer model specifically designed for mobile applications, combining architectural innovations with knowledge distillation to achieve performance comparable to BERT while requiring significantly fewer computational resources.

MobileBERT has approximately 25 million parameters, making it about 4.5x smaller than BERT-base, while maintaining comparable accuracy on many natural language understanding tasks. Its unique architecture features bottleneck structures and carefully designed attention mechanisms that reduce both model size and computational requirements. This efficiency makes it an excellent candidate for fake news detection applications that need to run on mobile devices or other resource-constrained environments.

### Why MobileBERT for Fake News Detection?

Transformer models like BERT have revolutionized NLP tasks with their ability to capture contextual relationships in text. However, their size and computational requirements can be prohibitive for many real-world applications, especially on mobile devices. MobileBERT addresses these limitations through a combination of architectural innovations and knowledge distillation.

For fake news detection, we need both accuracy and efficiency:
- Accuracy is critical to avoid falsely flagging legitimate news
- Efficiency enables deployment on mobile devices with limited computational power
- Fast inference speed allows for real-time content analysis
- Low memory footprint is essential for deployment on resource-constrained devices

## 2. Environment Setup

First, let's import the necessary libraries and set up our environment. We'll need PyTorch for deep learning, Hugging Face Transformers for the model, and various utilities for data processing and evaluation.


```python
# Import basic utilities
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

### PyTorch and System Libraries
We import PyTorch for our deep learning framework and set up system utilities for file management and timing.


```python
# Import PyTorch
import torch
import os
import time
import random
```

### Hugging Face Libraries
The Transformers library provides pre-trained models and utilities for fine-tuning, while the Datasets library helps with efficient data handling.


```python
# Import Hugging Face libraries
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from datasets import Dataset
```

    2025-05-19 08:40:29.546120: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    E0000 00:00:1747644029.750414      35 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    E0000 00:00:1747644029.812909      35 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered


### Evaluation Libraries
These libraries will help us measure our model's performance with standard metrics.


```python
# Import evaluation libraries
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix
```

### Utilities for Clean Output
We'll suppress warnings to keep our notebook clean and focused on the results.


```python
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
```

### Setting Up Reproducibility
Setting a random seed ensures our results are reproducible across different runs.


```python
# Set seeds for reproducibility
def set_seed(seed_value=42):
    """
    Set seeds for all random number generators to ensure reproducibility.
    This affects random, numpy, PyTorch CPU and GPU operations.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.backends.cudnn.deterministic = True
```


```python
# Apply seed
set_seed()
```

### Checking Hardware Availability
We'll check if a GPU is available to accelerate training.


```python
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

    Using device: cuda


## 3. Loading and Exploring the Dataset

The WELFake dataset combines real and fake news articles from multiple sources. Let's load it and understand its structure and distribution before proceeding with model training.


```python
# Load the cleaned dataset
df = pd.read_csv('/kaggle/input/welfake-cleaned/WELFake_cleaned.csv')
```

### Dataset Size and Shape
First, let's check the overall size of the dataset to understand how much data we're working with.


```python
# Display basic information
print(f"Dataset shape: {df.shape}")
```

    Dataset shape: (71537, 10)


### Class Distribution
It's important to check the balance between real and fake news articles to ensure our model doesn't develop bias toward the majority class.


```python
# Check class distribution
class_distribution = df['label'].value_counts(normalize=True).mul(100).round(2)
print(f"Class distribution:")
print(class_distribution)
```

    Class distribution:
    label
    1    51.04
    0    48.96
    Name: proportion, dtype: float64


### Visualizing the Class Distribution
A visual representation helps us better understand the dataset balance.


```python
# Visualize class distribution
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='label', data=df, palette='viridis')
plt.title('Distribution of Real vs. Fake News')
plt.xlabel('Label (0: Real, 1: Fake)')
plt.ylabel('Count')

# Add count labels on top of the bars
for p in ax.patches:
    ax.annotate(f'{p.get_height():,}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'bottom')
plt.tight_layout()
plt.show()
```


    
![png](output_22_0.png)
    


### Sample Data
Examining a few samples helps us understand the content and structure of the articles.


```python
# Display sample data
print("\nSample data:")
print(df.head(3))
```

    
    Sample data:
       Unnamed: 0                                              title  \
    0           0  LAW ENFORCEMENT ON HIGH ALERT Following Threat...   
    1           2  UNBELIEVABLE! OBAMA’S ATTORNEY GENERAL SAYS MO...   
    2           3  Bobby Jindal, raised Hindu, uses story of Chri...   
    
                                                    text  label  title_length  \
    0  No comment is expected from Barack Obama Membe...      1           130   
    1   Now, most of the demonstrators gathered last ...      1           137   
    2  A dozen politically active pastors came here f...      0           105   
    
       text_length  word_count  title_has_allcaps  title_exclamation  \
    0         5049         871               True              False   
    1          216          34               True               True   
    2         8010        1321              False              False   
    
       title_question  
    0           False  
    1           False  
    2           False  


## 4. Data Preprocessing

For transformer-based models like MobileBERT, we need to carefully prepare our input data. We'll combine the article title and text to provide complete information to the model and then split our data into training, validation, and test sets.

### Combining Title and Text
For news articles, both the headline and body contain important information. By combining them, we provide the model with the complete content.


```python
# Combine title and text
df['full_text'] = df['title'] + " " + df['text']
```

### Data Splitting
We'll use a stratified split to maintain class balance across all datasets. We'll create three sets:
- Training set (70% of data): Used to train the model
- Validation set (15% of data): Used for hyperparameter tuning and early stopping
- Test set (15% of data): Used for final evaluation


```python
# Import train_test_split
from sklearn.model_selection import train_test_split
```


```python
# First split: 85% train+val, 15% test
train_val_df, test_df = train_test_split(df, test_size=0.15, stratify=df['label'], random_state=42)
```


```python
# Second split: 70% train, 15% val (82.35% of the remaining 85%)
train_df, val_df = train_test_split(train_val_df, test_size=0.1765, stratify=train_val_df['label'], random_state=42)
```

### Checking Dataset Sizes
Let's verify our data splits have the expected proportions.


```python
# Display dataset sizes
print(f"Training set: {train_df.shape[0]} examples")
print(f"Validation set: {val_df.shape[0]} examples")
print(f"Test set: {test_df.shape[0]} examples")
```

    Training set: 50073 examples
    Validation set: 10733 examples
    Test set: 10731 examples


### Converting to Hugging Face Datasets
The Hugging Face Trainer API works best with their Dataset format, which optimizes memory usage and provides efficient data loading.


```python
# Convert training data to Hugging Face dataset
train_dataset = Dataset.from_pandas(train_df[['full_text', 'label']])
```


```python
# Convert validation data to Hugging Face dataset
val_dataset = Dataset.from_pandas(val_df[['full_text', 'label']])
```


```python
# Convert test data to Hugging Face dataset
test_dataset = Dataset.from_pandas(test_df[['full_text', 'label']])
```

## 5. Tokenization and Data Preparation

Transformer models like MobileBERT work with tokenized input, not raw text. We need to convert our text data into token IDs and prepare attention masks.

### Loading the MobileBERT Tokenizer
We'll use the pretrained tokenizer that corresponds to our MobileBERT model.


```python
# Load MobileBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
```


    config.json:   0%|          | 0.00/847 [00:00<?, ?B/s]



    vocab.txt:   0%|          | 0.00/232k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/466k [00:00<?, ?B/s]


### Defining the Tokenization Function
This function will convert text to token IDs and handle padding and truncation to ensure all inputs have the same length.


```python
# Define tokenization function
def tokenize_function(examples):
    """
    Tokenize text data for MobileBERT with appropriate padding and truncation.
    
    Args:
        examples: Dictionary containing text examples
        
    Returns:
        Dictionary with tokenized inputs
    """
    return tokenizer(
        examples["full_text"],
        padding="max_length",
        truncation=True,
        max_length=512,  # MobileBERT can handle sequences up to 512 tokens
    )
```

### Tokenizing the Datasets
Now we'll apply the tokenization function to all our datasets.


```python
# Tokenize training dataset
tokenized_train = train_dataset.map(tokenize_function, batched=True)
```


    Map:   0%|          | 0/50073 [00:00<?, ? examples/s]



```python
# Tokenize validation dataset
tokenized_val = val_dataset.map(tokenize_function, batched=True)
```


    Map:   0%|          | 0/10733 [00:00<?, ? examples/s]



```python
# Tokenize test dataset
tokenized_test = test_dataset.map(tokenize_function, batched=True)
```


    Map:   0%|          | 0/10731 [00:00<?, ? examples/s]


### Setting Dataset Format for PyTorch
We need to specify which columns to use and convert them to PyTorch tensors.


```python
# Set format for training dataset
tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
```


```python
# Set format for validation dataset
tokenized_val.set_format("torch", columns=["input_ids", "attention_mask", "label"])
```


```python
# Set format for test dataset
tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])
```

## 6. Setting Up the MobileBERT Model

Now we'll load the pretrained MobileBERT model and configure it for our binary classification task.

### Loading Pretrained MobileBERT
We'll use the standard MobileBERT uncased model, which provides an excellent balance between performance and efficiency.


```python
# Load MobileBERT model
model = AutoModelForSequenceClassification.from_pretrained(
    "google/mobilebert-uncased",
    num_labels=2  # Binary classification: real (0) or fake (1)
)
```


    pytorch_model.bin:   0%|          | 0.00/147M [00:00<?, ?B/s]


    Some weights of MobileBertForSequenceClassification were not initialized from the model checkpoint at google/mobilebert-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.


### Moving Model to GPU
To accelerate training, we'll move the model to GPU if available.


```python
# Move model to the appropriate device (GPU if available)
model.to(device)
```




    MobileBertForSequenceClassification(
      (mobilebert): MobileBertModel(
        (embeddings): MobileBertEmbeddings(
          (word_embeddings): Embedding(30522, 128, padding_idx=0)
          (position_embeddings): Embedding(512, 512)
          (token_type_embeddings): Embedding(2, 512)
          (embedding_transformation): Linear(in_features=384, out_features=512, bias=True)
          (LayerNorm): NoNorm()
          (dropout): Dropout(p=0.0, inplace=False)
        )
        (encoder): MobileBertEncoder(
          (layer): ModuleList(
            (0-23): 24 x MobileBertLayer(
              (attention): MobileBertAttention(
                (self): MobileBertSelfAttention(
                  (query): Linear(in_features=128, out_features=128, bias=True)
                  (key): Linear(in_features=128, out_features=128, bias=True)
                  (value): Linear(in_features=512, out_features=128, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): MobileBertSelfOutput(
                  (dense): Linear(in_features=128, out_features=128, bias=True)
                  (LayerNorm): NoNorm()
                )
              )
              (intermediate): MobileBertIntermediate(
                (dense): Linear(in_features=128, out_features=512, bias=True)
                (intermediate_act_fn): ReLU()
              )
              (output): MobileBertOutput(
                (dense): Linear(in_features=512, out_features=128, bias=True)
                (LayerNorm): NoNorm()
                (bottleneck): OutputBottleneck(
                  (dense): Linear(in_features=128, out_features=512, bias=True)
                  (LayerNorm): NoNorm()
                  (dropout): Dropout(p=0.0, inplace=False)
                )
              )
              (bottleneck): Bottleneck(
                (input): BottleneckLayer(
                  (dense): Linear(in_features=512, out_features=128, bias=True)
                  (LayerNorm): NoNorm()
                )
                (attention): BottleneckLayer(
                  (dense): Linear(in_features=512, out_features=128, bias=True)
                  (LayerNorm): NoNorm()
                )
              )
              (ffn): ModuleList(
                (0-2): 3 x FFNLayer(
                  (intermediate): MobileBertIntermediate(
                    (dense): Linear(in_features=128, out_features=512, bias=True)
                    (intermediate_act_fn): ReLU()
                  )
                  (output): FFNOutput(
                    (dense): Linear(in_features=512, out_features=128, bias=True)
                    (LayerNorm): NoNorm()
                  )
                )
              )
            )
          )
        )
        (pooler): MobileBertPooler()
      )
      (dropout): Dropout(p=0.0, inplace=False)
      (classifier): Linear(in_features=512, out_features=2, bias=True)
    )



### Examining Model Architecture
Understanding the model architecture helps us appreciate its efficiency compared to larger models like BERT-base.


```python
# Print model architecture
print(model)
```

    MobileBertForSequenceClassification(
      (mobilebert): MobileBertModel(
        (embeddings): MobileBertEmbeddings(
          (word_embeddings): Embedding(30522, 128, padding_idx=0)
          (position_embeddings): Embedding(512, 512)
          (token_type_embeddings): Embedding(2, 512)
          (embedding_transformation): Linear(in_features=384, out_features=512, bias=True)
          (LayerNorm): NoNorm()
          (dropout): Dropout(p=0.0, inplace=False)
        )
        (encoder): MobileBertEncoder(
          (layer): ModuleList(
            (0-23): 24 x MobileBertLayer(
              (attention): MobileBertAttention(
                (self): MobileBertSelfAttention(
                  (query): Linear(in_features=128, out_features=128, bias=True)
                  (key): Linear(in_features=128, out_features=128, bias=True)
                  (value): Linear(in_features=512, out_features=128, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): MobileBertSelfOutput(
                  (dense): Linear(in_features=128, out_features=128, bias=True)
                  (LayerNorm): NoNorm()
                )
              )
              (intermediate): MobileBertIntermediate(
                (dense): Linear(in_features=128, out_features=512, bias=True)
                (intermediate_act_fn): ReLU()
              )
              (output): MobileBertOutput(
                (dense): Linear(in_features=512, out_features=128, bias=True)
                (LayerNorm): NoNorm()
                (bottleneck): OutputBottleneck(
                  (dense): Linear(in_features=128, out_features=512, bias=True)
                  (LayerNorm): NoNorm()
                  (dropout): Dropout(p=0.0, inplace=False)
                )
              )
              (bottleneck): Bottleneck(
                (input): BottleneckLayer(
                  (dense): Linear(in_features=512, out_features=128, bias=True)
                  (LayerNorm): NoNorm()
                )
                (attention): BottleneckLayer(
                  (dense): Linear(in_features=512, out_features=128, bias=True)
                  (LayerNorm): NoNorm()
                )
              )
              (ffn): ModuleList(
                (0-2): 3 x FFNLayer(
                  (intermediate): MobileBertIntermediate(
                    (dense): Linear(in_features=128, out_features=512, bias=True)
                    (intermediate_act_fn): ReLU()
                  )
                  (output): FFNOutput(
                    (dense): Linear(in_features=512, out_features=128, bias=True)
                    (LayerNorm): NoNorm()
                  )
                )
              )
            )
          )
        )
        (pooler): MobileBertPooler()
      )
      (dropout): Dropout(p=0.0, inplace=False)
      (classifier): Linear(in_features=512, out_features=2, bias=True)
    )


## 7. Defining the Evaluation Metrics

To assess our model's performance, we need to define appropriate evaluation metrics. For fake news detection, accuracy alone isn't sufficient—we also need to consider precision, recall, and F1 score.

### Metrics Computation Function
This function will calculate multiple performance metrics during training and evaluation.


```python
# Define the metrics computation function
def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for the model.
    
    Args:
        eval_pred: Tuple of predictions and label ids
        
    Returns:
        Dictionary of metrics including accuracy, precision, recall, and F1
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)
    
    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
```

## 8. Training Configuration

Configuring the training process properly is crucial for achieving good model performance efficiently.

### Output Directory
We'll specify where to save checkpoints and the final model.


```python
# Define output directory
output_dir = "./results/mobilebert_welfake"
```

### Training Arguments
The TrainingArguments object configures all aspects of the training process, from learning rate to batch size to evaluation strategy.


```python
# Define training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",         # Evaluate after each epoch
    save_strategy="epoch",         # Save checkpoint after each epoch
    learning_rate=5e-5,            # Recommended fine-tuning learning rate for BERT family models
    per_device_train_batch_size=16, # Batch size for training
    per_device_eval_batch_size=64,  # Larger batch size for evaluation (no gradients needed)
    num_train_epochs=5,             # Maximum number of epochs
    weight_decay=0.01,              # Weight decay for regularization
    load_best_model_at_end=True,    # Load the best model at the end of training
    metric_for_best_model="f1",     # Use F1 score to determine the best model
    push_to_hub=False,              # Don't push to Hugging Face Hub
    report_to="tensorboard",        # Generate TensorBoard logs
)
```

### Early Stopping
To prevent overfitting and save training time, we'll implement early stopping based on validation performance.


```python
# Create early stopping callback
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2)
```

### Initializing the Trainer
The Hugging Face Trainer handles the training loop, evaluation, and logging.


```python
# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback]
)
```

## 9. Training Process

Now we're ready to train our MobileBERT model on the WELFake dataset. We'll track the training time to assess efficiency.

### Starting Timer
We'll measure how long the training process takes.


```python
# Start timer
start_time = time.time()
```

### Training the Model
This will fine-tune MobileBERT on our fake news detection task.


```python
# Train the model
print("Starting training...")
train_result = trainer.train()
```

    Starting training...



    model.safetensors:   0%|          | 0.00/147M [00:00<?, ?B/s]




    <div>

      <progress value='1798' max='7825' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [1798/7825 25:25 < 1:25:19, 1.18 it/s, Epoch 1.15/5]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>0.067500</td>
      <td>0.022665</td>
      <td>0.991708</td>
      <td>0.991782</td>
      <td>0.991708</td>
      <td>0.991709</td>
    </tr>
  </tbody>
</table><p>



```python
train_result = trainer.train(resume_from_checkpoint=True)
```



    <div>

      <progress value='7825' max='7825' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [7825/7825 1:08:21, Epoch 5/5]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3</td>
      <td>0.097500</td>
      <td>0.022665</td>
      <td>0.996180</td>
      <td>0.996180</td>
      <td>0.996180</td>
      <td>0.996180</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.001900</td>
      <td>0.020062</td>
      <td>0.995807</td>
      <td>0.995810</td>
      <td>0.995807</td>
      <td>0.995807</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.000400</td>
      <td>0.022069</td>
      <td>0.996553</td>
      <td>0.996554</td>
      <td>0.996553</td>
      <td>0.996553</td>
    </tr>
  </tbody>
</table><p>


### Calculating Training Time
Let's check how long the training took.


```python
# Calculate training time
train_time = time.time() - start_time
print(f"Training completed in {train_time/60:.2f} minutes")
```

    Training completed in 129.33 minutes


### Training Metrics Summary
We'll examine the final training metrics.


```python
# Print training metrics
print(f"Training metrics: {train_result.metrics}")
```

    Training metrics: {'train_runtime': 4102.3187, 'train_samples_per_second': 61.03, 'train_steps_per_second': 1.907, 'total_flos': 1.650870793645056e+16, 'train_loss': 0.007448047980332908, 'epoch': 5.0}


## 10. Evaluation on Test Set

After training, we'll evaluate our model on the held-out test set to assess its performance on unseen data.

### Running Evaluation
This will calculate all our defined metrics on the test set.


```python
# Evaluate on test set
print("\nEvaluating on test set...")
test_results = trainer.evaluate(tokenized_test)
```

    
    Evaluating on test set...






### Displaying Test Results
Let's examine the test metrics to see how well our model generalizes.


```python
# Print test results
print(f"Test results: {test_results}")
```

    Test results: {'eval_loss': 0.021314876154065132, 'eval_accuracy': 0.9959929177150312, 'eval_precision': 0.9959981623001891, 'eval_recall': 0.9959929177150312, 'eval_f1': 0.9959930396207516, 'eval_runtime': 87.8405, 'eval_samples_per_second': 122.165, 'eval_steps_per_second': 0.956, 'epoch': 5.0}


### Generating and Processing Predictions
To create a confusion matrix and detailed classification report, we need the individual predictions.


```python
# Get predictions
predictions = trainer.predict(tokenized_test)
```


```python
# Process predictions
preds = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids
```

### Creating a Confusion Matrix
A confusion matrix helps us understand where the model makes errors.


```python
# Create confusion matrix
cm = confusion_matrix(labels, preds)
```


```python
# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Real News', 'Fake News'], 
            yticklabels=['Real News', 'Fake News'])
plt.title('MobileBERT Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()
```


    
![png](output_83_0.png)
    


### Detailed Classification Report
This provides precision, recall, and F1 score for each class.


```python
# Print classification report
print("\nClassification Report:")
print(classification_report(labels, preds, target_names=['Real News', 'Fake News']))
```

    
    Classification Report:
                  precision    recall  f1-score   support
    
       Real News       0.99      1.00      1.00      5254
       Fake News       1.00      0.99      1.00      5477
    
        accuracy                           1.00     10731
       macro avg       1.00      1.00      1.00     10731
    weighted avg       1.00      1.00      1.00     10731
    


## 11. Saving the Model

To use our trained model in applications, we need to save it to disk.

### Creating Save Path
This defines where we'll store the model files.


```python
# Define model save path
model_save_path = "./mobilebert_welfake_model"
```

### Saving Model and Tokenizer
We need to save both the model weights and the tokenizer for inference.


```python
# Save model
trainer.save_model(model_save_path)
```


```python
# Save tokenizer
tokenizer.save_pretrained(model_save_path)
```




    ('./mobilebert_welfake_model/tokenizer_config.json',
     './mobilebert_welfake_model/special_tokens_map.json',
     './mobilebert_welfake_model/vocab.txt',
     './mobilebert_welfake_model/added_tokens.json',
     './mobilebert_welfake_model/tokenizer.json')




```python
print(f"Model saved to {model_save_path}")
```

    Model saved to ./mobilebert_welfake_model


## 12. Error Analysis

Understanding where and why our model makes mistakes is crucial for improvement.

### Finding Misclassified Examples
Let's identify which examples the model got wrong.


```python
# Find misclassified examples
misclassified_indices = np.where(preds != labels)[0]
```

### Creating a DataFrame of Errors
This will help us analyze the misclassified examples.


```python
# Create DataFrame of misclassified examples
misclassified_df = test_df.iloc[misclassified_indices].reset_index(drop=True)
```

### Adding Predicted Labels
We'll add the model's predictions to compare with the true labels.


```python
# Add prediction column
misclassified_df['prediction'] = preds[misclassified_indices]
```

### Counting Misclassifications
Let's see how many examples were misclassified out of the total.


```python
# Display number of misclassified examples
print(f"Total misclassified examples: {len(misclassified_df)}")
```

    Total misclassified examples: 43


### Examining Error Examples
Looking at specific misclassified examples helps identify patterns in the model's errors.


```python
# Display sample of misclassified examples
print("\nSample of misclassified examples:")
for i in range(min(5, len(misclassified_df))):
    print(f"\nExample {i+1}:")
    print(f"Title: {misclassified_df.iloc[i]['title']}")
    print(f"True label: {'Real' if misclassified_df.iloc[i]['label'] == 0 else 'Fake'}")
    print(f"Predicted: {'Real' if misclassified_df.iloc[i]['prediction'] == 0 else 'Fake'}")
    print("-" * 80)
```

    
    Sample of misclassified examples:
    
    Example 1:
    Title: "Top Five Clinton Donors Are Jewish" - How Anti-Semitic Is This Fact?
    True label: Fake
    Predicted: Real
    --------------------------------------------------------------------------------
    
    Example 2:
    Title: Will Trump's presidency change the way America views Russia?
    True label: Fake
    Predicted: Real
    --------------------------------------------------------------------------------
    
    Example 3:
    Title: Emergency Survival Food Sales Soar as We Get Closer to Election Day
    True label: Fake
    Predicted: Real
    --------------------------------------------------------------------------------
    
    Example 4:
    Title: Here’s How Goldman Sachs Lays People Off | Financial Markets
    True label: Fake
    Predicted: Real
    --------------------------------------------------------------------------------
    
    Example 5:
    Title: Aziz Ansari Why Trump Makes Me Scared for My Family
    
    True label: Real
    Predicted: Fake
    --------------------------------------------------------------------------------


## 13. Resource Usage Analysis

For practical deployment, we need to understand the resource requirements of our model.

### Model Size Calculation
Let's measure the memory footprint of our model.


```python
# Define function to calculate model size
def get_model_size(model):
    """Calculate model size in MB"""
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    return model_size
```


```python
# Calculate model size
model_size_mb = get_model_size(model)
```

### Parameter Count
The number of parameters affects both model size and computational requirements.


```python
# Calculate parameter count
param_count = sum(p.numel() for p in model.parameters())
```

### Printing Size Statistics
Let's summarize the model's resource usage.


```python
# Print model size statistics
print(f"\nModel Analysis:")
print(f"Parameter count: {param_count:,}")
print(f"Model size: {model_size_mb:.2f} MB")
```

    
    Model Analysis:
    Parameter count: 24,582,914
    Model size: 93.78 MB


### Measuring Inference Time
Inference speed is critical for real-time applications, especially on mobile devices.


```python
# Define function to measure inference time
def measure_inference_time(model, dataset, batch_size=1, num_samples=100):
    """
    Measure average inference time per sample at different batch sizes.
    
    Args:
        model: The model to evaluate
        dataset: The dataset to use for inference
        batch_size: Batch size for inference
        num_samples: Number of samples to process
        
    Returns:
        Average inference time per sample in milliseconds
    """
    model.eval()
    dataloader = torch.utils.data.DataLoader(
        dataset.select(range(min(num_samples, len(dataset)))), 
        batch_size=batch_size
    )
    
    total_time = 0
    sample_count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            start_time = time.time()
            _ = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            batch_time = time.time() - start_time
            
            total_time += batch_time
            sample_count += batch['input_ids'].size(0)
    
    avg_time_per_sample = (total_time / sample_count) * 1000  # Convert to ms
    return avg_time_per_sample
```

### Testing Different Batch Sizes
Measuring inference time with different batch sizes helps optimize deployment, especially for mobile applications.


```python
# Measure inference time for batch size 1
bs1_time = measure_inference_time(model, tokenized_test, batch_size=1)
print(f"Average inference time (batch size 1): {bs1_time:.2f} ms per sample")
```

    Average inference time (batch size 1): 34.45 ms per sample



```python
# Measure inference time for batch size 8
bs8_time = measure_inference_time(model, tokenized_test, batch_size=8)
print(f"Average inference time (batch size 8): {bs8_time:.2f} ms per sample")
```

    Average inference time (batch size 8): 4.52 ms per sample



```python
# Measure inference time for batch size 32
bs32_time = measure_inference_time(model, tokenized_test, batch_size=32)
print(f"Average inference time (batch size 32): {bs32_time:.2f} ms per sample")
```

    Average inference time (batch size 32): 2.38 ms per sample


## 14. Conclusion and Performance Analysis

Our MobileBERT model demonstrates the potential for efficient fake news detection that could run on mobile devices. This is particularly valuable as mobile platforms are often the primary means through which people consume news content.

### Understanding the Results

Looking at the confusion matrix and classification report, we can analyze the model's performance in terms of:

1. **True Positives (Fake News correctly identified)**
2. **True Negatives (Real News correctly identified)**
3. **False Positives (Real News misclassified as Fake)**
4. **False Negatives (Fake News misclassified as Real)**

The balance between precision and recall is particularly important in a fake news detection context, where both false positives (incorrectly flagging legitimate news) and false negatives (failing to catch fake news) have different but significant costs.

### Efficiency Analysis

MobileBERT's key advantage is its efficiency, which we can see in:

1. **Model Size**: Approximately 100 MB (25 million parameters)
2. **Training Time**: To be measured during execution
3. **Inference Speed**: To be measured during execution, but expected to be 4x faster than BERT-base

These metrics demonstrate why MobileBERT is particularly suitable for mobile deployment, where both storage space and computational capacity are limited compared to server environments.

### Comparison with Other Models

Comparing MobileBERT with the previous models we've examined:

- **MobileBERT**: Designed specifically for mobile deployment with bottleneck structures
- **DistilBERT**: General-purpose compressed model using knowledge distillation (66M parameters)
- **TinyBERT**: Highly compressed model with significant size reduction (15M parameters)

Each model represents a different trade-off between performance, size, and inference speed, allowing developers to choose the most appropriate option based on their specific deployment constraints.

## 15. Next Steps and Practical Applications

MobileBERT's efficient design makes it particularly well-suited for on-device fake news detection, leading to several promising directions for deployment and enhancement:

### Mobile-Specific Deployment

1. **Android/iOS Integration**: Package the model for direct integration into mobile applications
2. **On-Device Fine-Tuning**: Explore personalized model adaptation while preserving privacy
3. **Battery Impact Analysis**: Measure and optimize power consumption for real-world usage

### Model Enhancement

1. **Quantization**: Further reduce model size through 8-bit or even 4-bit quantization
2. **Model Pruning**: Experiment with removing less important weights to improve efficiency
3. **Knowledge Distillation**: Create an even smaller model using MobileBERT as a teacher

### Real-World Applications

1. **News Reader Plugin**: Develop a component that can be integrated into mobile news applications
2. **Offline Detection**: Enable fake news detection even when the device is disconnected from the internet
3. **Multi-Modal Analysis**: Combine text analysis with image verification using mobile-optimized vision models

By leveraging MobileBERT's efficiency, we can bring sophisticated fake news detection capabilities directly to users' devices, helping combat the spread of misinformation at the point of consumption without requiring constant server connectivity or compromising user privacy.
