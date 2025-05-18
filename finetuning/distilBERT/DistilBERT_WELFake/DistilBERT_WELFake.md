# DistilBERT for Fake News Detection

## 1. Introduction

This notebook demonstrates how to finetune DistilBERT for detecting fake news using the WELFake dataset. DistilBERT is an efficient and lightweight transformer model that uses knowledge distillation to achieve performance comparable to larger models while requiring significantly fewer computational resources.

DistilBERT has approximately 66 million parameters, making it about 40% smaller than BERT-base, while maintaining 97% of BERT's performance on natural language understanding tasks. This efficiency makes it an excellent candidate for fake news detection applications that may need to run in environments with limited resources.

### Why DistilBERT for Fake News Detection?

Transformer models like BERT have revolutionized NLP tasks with their ability to capture contextual relationships in text. However, their size and computational requirements can be prohibitive for many real-world applications. DistilBERT addresses this limitation through knowledge distillation, where a smaller student model (DistilBERT) learns to mimic a larger teacher model (BERT).

For fake news detection, we need both accuracy and efficiency:
- Accuracy is critical to avoid falsely flagging legitimate news
- Efficiency enables deployment in real-time content moderation systems
- Smaller model size allows for deployment in environments with resource constraints

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

    2025-05-18 17:45:21.684638: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    E0000 00:00:1747590321.917080      35 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    E0000 00:00:1747590321.984477      35 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered


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

For transformer-based models like DistilBERT, we need to carefully prepare our input data. We'll combine the article title and text to provide complete information to the model and then split our data into training, validation, and test sets.

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

Transformer models like DistilBERT work with tokenized input, not raw text. We need to convert our text data into token IDs and prepare attention masks.

### Loading the DistilBERT Tokenizer
We'll use the pretrained tokenizer that corresponds to our DistilBERT model.


```python
# Load DistilBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
```


    tokenizer_config.json:   0%|          | 0.00/48.0 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/483 [00:00<?, ?B/s]



    vocab.txt:   0%|          | 0.00/232k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/466k [00:00<?, ?B/s]


### Defining the Tokenization Function
This function will convert text to token IDs and handle padding and truncation to ensure all inputs have the same length.


```python
# Define tokenization function
def tokenize_function(examples):
    """
    Tokenize text data for DistilBERT with appropriate padding and truncation.
    
    Args:
        examples: Dictionary containing text examples
        
    Returns:
        Dictionary with tokenized inputs
    """
    return tokenizer(
        examples["full_text"],
        padding="max_length",
        truncation=True,
        max_length=512,  # DistilBERT can handle sequences up to 512 tokens
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

## 6. Setting Up the DistilBERT Model

Now we'll load the pretrained DistilBERT model and configure it for our binary classification task.

### Loading Pretrained DistilBERT
We'll use the standard DistilBERT base model, which provides a good balance between performance and efficiency.


```python
# Load DistilBERT model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2  # Binary classification: real (0) or fake (1)
)
```


    model.safetensors:   0%|          | 0.00/268M [00:00<?, ?B/s]


    Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.


### Moving Model to GPU
To accelerate training, we'll move the model to GPU if available.


```python
# Move model to the appropriate device (GPU if available)
model.to(device)
```




    DistilBertForSequenceClassification(
      (distilbert): DistilBertModel(
        (embeddings): Embeddings(
          (word_embeddings): Embedding(30522, 768, padding_idx=0)
          (position_embeddings): Embedding(512, 768)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (transformer): Transformer(
          (layer): ModuleList(
            (0-5): 6 x TransformerBlock(
              (attention): DistilBertSdpaAttention(
                (dropout): Dropout(p=0.1, inplace=False)
                (q_lin): Linear(in_features=768, out_features=768, bias=True)
                (k_lin): Linear(in_features=768, out_features=768, bias=True)
                (v_lin): Linear(in_features=768, out_features=768, bias=True)
                (out_lin): Linear(in_features=768, out_features=768, bias=True)
              )
              (sa_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (ffn): FFN(
                (dropout): Dropout(p=0.1, inplace=False)
                (lin1): Linear(in_features=768, out_features=3072, bias=True)
                (lin2): Linear(in_features=3072, out_features=768, bias=True)
                (activation): GELUActivation()
              )
              (output_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            )
          )
        )
      )
      (pre_classifier): Linear(in_features=768, out_features=768, bias=True)
      (classifier): Linear(in_features=768, out_features=2, bias=True)
      (dropout): Dropout(p=0.2, inplace=False)
    )



### Examining Model Architecture
Understanding the model architecture helps us appreciate its efficiency compared to larger models like BERT-base.


```python
# Print model architecture
print(model)
```

    DistilBertForSequenceClassification(
      (distilbert): DistilBertModel(
        (embeddings): Embeddings(
          (word_embeddings): Embedding(30522, 768, padding_idx=0)
          (position_embeddings): Embedding(512, 768)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (transformer): Transformer(
          (layer): ModuleList(
            (0-5): 6 x TransformerBlock(
              (attention): DistilBertSdpaAttention(
                (dropout): Dropout(p=0.1, inplace=False)
                (q_lin): Linear(in_features=768, out_features=768, bias=True)
                (k_lin): Linear(in_features=768, out_features=768, bias=True)
                (v_lin): Linear(in_features=768, out_features=768, bias=True)
                (out_lin): Linear(in_features=768, out_features=768, bias=True)
              )
              (sa_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (ffn): FFN(
                (dropout): Dropout(p=0.1, inplace=False)
                (lin1): Linear(in_features=768, out_features=3072, bias=True)
                (lin2): Linear(in_features=3072, out_features=768, bias=True)
                (activation): GELUActivation()
              )
              (output_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            )
          )
        )
      )
      (pre_classifier): Linear(in_features=768, out_features=768, bias=True)
      (classifier): Linear(in_features=768, out_features=2, bias=True)
      (dropout): Dropout(p=0.2, inplace=False)
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
output_dir = "./results/distilbert_welfake"
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

Now we're ready to train our DistilBERT model on the WELFake dataset. We'll track the training time to assess efficiency.

### Starting Timer
We'll measure how long the training process takes.


```python
# Start timer
start_time = time.time()
```

### Training the Model
This will fine-tune DistilBERT on our fake news detection task.


```python
# Train the model
print("Starting training...")
train_result = trainer.train()
```

    Starting training...




    <div>

      <progress value='6333' max='7825' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [6333/7825 1:34:32 < 22:16, 1.12 it/s, Epoch 4.05/5]
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
      <td>0.031500</td>
      <td>0.032046</td>
      <td>0.989472</td>
      <td>0.989592</td>
      <td>0.989472</td>
      <td>0.989473</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.013800</td>
      <td>0.025826</td>
      <td>0.992826</td>
      <td>0.992865</td>
      <td>0.992826</td>
      <td>0.992826</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.007600</td>
      <td>0.018540</td>
      <td>0.996273</td>
      <td>0.996273</td>
      <td>0.996273</td>
      <td>0.996273</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.000900</td>
      <td>0.026803</td>
      <td>0.996179</td>
      <td>0.996187</td>
      <td>0.996179</td>
      <td>0.996179</td>
    </tr>
  </tbody>
</table><p>
    <div>

      <progress value='84' max='84' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [84/84 01:24]
    </div>




    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    /tmp/ipykernel_35/919901487.py in <cell line: 0>()
          1 # Train the model
          2 print("Starting training...")
    ----> 3 train_result = trainer.train()
    

    /usr/local/lib/python3.11/dist-packages/transformers/trainer.py in train(self, resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
       2243                 hf_hub_utils.enable_progress_bars()
       2244         else:
    -> 2245             return inner_training_loop(
       2246                 args=args,
       2247                 resume_from_checkpoint=resume_from_checkpoint,


    /usr/local/lib/python3.11/dist-packages/transformers/trainer.py in _inner_training_loop(self, batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)
       2558                     )
       2559                     with context():
    -> 2560                         tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
       2561 
       2562                     if (


    /usr/local/lib/python3.11/dist-packages/transformers/trainer.py in training_step(***failed resolving arguments***)
       3780                 kwargs["scale_wrt_gas"] = False
       3781 
    -> 3782             self.accelerator.backward(loss, **kwargs)
       3783 
       3784             return loss.detach()


    /usr/local/lib/python3.11/dist-packages/accelerate/accelerator.py in backward(self, loss, **kwargs)
       2357             self.lomo_backward(loss, learning_rate)
       2358         else:
    -> 2359             loss.backward(**kwargs)
       2360 
       2361     def set_trigger(self):


    /usr/local/lib/python3.11/dist-packages/torch/_tensor.py in backward(self, gradient, retain_graph, create_graph, inputs)
        624                 inputs=inputs,
        625             )
    --> 626         torch.autograd.backward(
        627             self, gradient, retain_graph, create_graph, inputs=inputs
        628         )


    /usr/local/lib/python3.11/dist-packages/torch/autograd/__init__.py in backward(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)
        345     # some Python versions print out the first line of a multi-line function
        346     # calls in the traceback and some print out the last line
    --> 347     _engine_run_backward(
        348         tensors,
        349         grad_tensors_,


    /usr/local/lib/python3.11/dist-packages/torch/autograd/graph.py in _engine_run_backward(t_outputs, *args, **kwargs)
        821         unregister_hooks = _register_logging_hooks_on_whole_graph(t_outputs)
        822     try:
    --> 823         return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
        824             t_outputs, *args, **kwargs
        825         )  # Calls into the C++ engine to run the backward pass


    KeyboardInterrupt: 


It seems the model started overfitting at epoch 4 so I stopped the training. Now I will explicitly set the model to checkpoint at epoch 3 which gave me the best results.


```python
model = AutoModelForSequenceClassification.from_pretrained(
    "/kaggle/working/results/distilbert_welfake/checkpoint-4695"
)
```


```python
# Load it to the appropriate device
model.to(device)
```




    DistilBertForSequenceClassification(
      (distilbert): DistilBertModel(
        (embeddings): Embeddings(
          (word_embeddings): Embedding(30522, 768, padding_idx=0)
          (position_embeddings): Embedding(512, 768)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (transformer): Transformer(
          (layer): ModuleList(
            (0-5): 6 x TransformerBlock(
              (attention): DistilBertSdpaAttention(
                (dropout): Dropout(p=0.1, inplace=False)
                (q_lin): Linear(in_features=768, out_features=768, bias=True)
                (k_lin): Linear(in_features=768, out_features=768, bias=True)
                (v_lin): Linear(in_features=768, out_features=768, bias=True)
                (out_lin): Linear(in_features=768, out_features=768, bias=True)
              )
              (sa_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (ffn): FFN(
                (dropout): Dropout(p=0.1, inplace=False)
                (lin1): Linear(in_features=768, out_features=3072, bias=True)
                (lin2): Linear(in_features=3072, out_features=768, bias=True)
                (activation): GELUActivation()
              )
              (output_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            )
          )
        )
      )
      (pre_classifier): Linear(in_features=768, out_features=768, bias=True)
      (classifier): Linear(in_features=768, out_features=2, bias=True)
      (dropout): Dropout(p=0.2, inplace=False)
    )



### Calculating Training Time
Let's check how long the training took.


```python
# Calculate training time
train_time = time.time() - start_time
print(f"Training completed in {train_time/60:.2f} minutes")
```

    Training completed in 98.83 minutes


### Training Metrics Summary
We'll examine the final training metrics.


```python
# Print training metrics
print(f"Training metrics: {train_result.metrics}")
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /tmp/ipykernel_35/3970003865.py in <cell line: 0>()
          1 # Print training metrics
    ----> 2 print(f"Training metrics: {train_result.metrics}")
    

    NameError: name 'train_result' is not defined


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

    Test results: {'eval_loss': 0.02680298313498497, 'eval_accuracy': 0.9961792936352624, 'eval_precision': 0.9961872073978384, 'eval_recall': 0.9961792936352624, 'eval_f1': 0.9961794344304633}


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
plt.title('DistilBERT Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()
```


    
![png](output_85_0.png)
    


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
model_save_path = "./distilbert_welfake_model"
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




    ('./distilbert_welfake_model/tokenizer_config.json',
     './distilbert_welfake_model/special_tokens_map.json',
     './distilbert_welfake_model/vocab.txt',
     './distilbert_welfake_model/added_tokens.json',
     './distilbert_welfake_model/tokenizer.json')




```python
print(f"Model saved to {model_save_path}")
```

    Model saved to ./distilbert_welfake_model


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

    Total misclassified examples: 41


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
    Title: US Abstains From UN Vote Against Cuba Embargo
    True label: Fake
    Predicted: Real
    --------------------------------------------------------------------------------
    
    Example 3:
    Title: Hillary Campaign Manager Gets $7K a Month from Hillary Donor Linked to Economic Collapse
    True label: Fake
    Predicted: Real
    --------------------------------------------------------------------------------
    
    Example 4:
    Title: Move To Drug Test Gun Owners In Open Carry States
    True label: Fake
    Predicted: Real
    --------------------------------------------------------------------------------
    
    Example 5:
    Title: Kevin MacDonald celebrates Trump’s Amazing Victory
    True label: Fake
    Predicted: Real
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
    Parameter count: 66,955,010
    Model size: 255.41 MB


### Measuring Inference Time
Inference speed is critical for real-time applications.


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
Measuring inference time with different batch sizes helps optimize deployment.


```python
# Measure inference time for batch size 1
bs1_time = measure_inference_time(model, tokenized_test, batch_size=1)
print(f"Average inference time (batch size 1): {bs1_time:.2f} ms per sample")
```

    Average inference time (batch size 1): 4.51 ms per sample



```python
# Measure inference time for batch size 8
bs8_time = measure_inference_time(model, tokenized_test, batch_size=8)
print(f"Average inference time (batch size 8): {bs8_time:.2f} ms per sample")
```

    Average inference time (batch size 8): 0.59 ms per sample



```python
# Measure inference time for batch size 32
bs32_time = measure_inference_time(model, tokenized_test, batch_size=32)
print(f"Average inference time (batch size 32): {bs32_time:.2f} ms per sample")
```

    Average inference time (batch size 32): 0.24 ms per sample


## 14. Conclusion and Performance Analysis

Our DistilBERT model achieved excellent performance on the fake news detection task. This is particularly impressive considering the model's reduced size compared to BERT-base.

### Understanding the Results

Looking at the confusion matrix and classification report, we can analyze:

1. **True Positives (Fake News correctly identified)**
2. **True Negatives (Real News correctly identified)**
3. **False Positives (Real News misclassified as Fake)**
4. **False Negatives (Fake News misclassified as Real)**

The model's tendency to misclassify in one direction more than another is an important consideration for practical applications, as false negatives (missing fake news) might be more concerning than false positives in certain contexts.

### Efficiency Analysis

The model demonstrates efficient metrics:

1. **Model Size**: Approximately 250 MB (66 million parameters)
2. **Training Time**: To be measured during execution
3. **Inference Speed**: To be measured during execution

This efficiency is a direct result of using DistilBERT, which is approximately 40% smaller than BERT-base (which has 110M parameters) but achieves 97% of BERT's performance through knowledge distillation.

### Comparison with Traditional Models

Compared to traditional machine learning approaches (Logistic Regression and Random Forest with TF-IDF features) discussed in the baseline models document, DistilBERT is expected to show superior performance. The typical performance improvement of transformer-based models over traditional approaches demonstrates their power in capturing complex linguistic patterns and contextual relationships that traditional bag-of-words approaches miss.

## 15. Next Steps and Practical Applications

Our DistilBERT model for fake news detection shows great promise for real-world applications. Here are several directions for future work:

### Deployment Considerations

1. **API Development**: Create a simple REST API for real-time fake news detection
2. **Mobile Deployment**: Package the model for on-device inference on smartphones
3. **Browser Extension**: Develop a browser plugin that can analyze news articles as users browse

### Explainability

1. **Attention Visualization**: Analyze which parts of articles the model focuses on when making predictions
2. **Feature Importance**: Identify which words or phrases most strongly influence classification
3. **User-Friendly Explanations**: Develop ways to explain why an article was classified as fake or real

By leveraging DistilBERT's efficiency and accuracy, we can create practical tools that help combat the spread of misinformation while maintaining reasonable computational requirements.
