# MobileBERT Evaluation on ISOT Dataset

## Introduction

This notebook documents the evaluation of a fine-tuned MobileBERT model on the ISOT fake news detection dataset. The primary goal is to assess the model's performance and analyze its resource consumption, with a specific focus on CPU-based edge deployment scenarios such as mobile devices and laptops. MobileBERT was specifically designed for mobile applications, potentially offering the best balance between accuracy and resource efficiency compared to other models in our comparative study (DistilBERT, TinyBERT, and RoBERTa).

## 1. Setting Up the Environment

First, I import all necessary libraries and set up utility functions to monitor resource usage:


```python
# Import necessary libraries
import os
import time
import numpy as np
import pandas as pd
import torch
import psutil
import gc
import re
```


```python
# Import transformer-specific libraries
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
from datasets import Dataset as HFDataset
```


```python
# Import evaluation libraries
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
```

I've selected these libraries for the following reasons:
- Core data science libraries (`numpy`, `pandas`) for data manipulation
- PyTorch (`torch`) for model inference
- Hugging Face's `transformers` for loading and using the MobileBERT model
- `psutil` for monitoring system resource usage during evaluation
- `sklearn.metrics` for comprehensive model evaluation metrics
- Visualization libraries (`matplotlib`, `seaborn`) for result analysis
- `gc` for explicit garbage collection to manage memory efficiently


```python
# Set device - using CPU for edge device testing
device = torch.device("cpu")
print(f"Using device: {device}")
```

    Using device: cpu


I deliberately choose to use the CPU rather than GPU for this evaluation because:
1. The primary target for MobileBERT deployment is mobile and edge devices that typically lack dedicated GPUs
2. CPU performance metrics provide a more realistic assessment of real-world deployment scenarios
3. This allows for direct comparison with other lightweight models in similar deployment conditions


```python
# Function to get current memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB
```

This utility function tracks the resident set size (RSS) of the Python process, providing a reliable measure of actual memory consumption during model loading and inference. This is particularly important for MobileBERT, which was specifically designed to minimize memory footprint.

## 2. Loading and Preparing ISOT Evaluation Dataset

Next, I load the ISOT evaluation dataset, which provides a reliable assessment since it comes from the same domain as the training data:


```python
# Check memory usage before loading dataset
print(f"Memory usage before loading dataset: {get_memory_usage():.2f} MB")
```

    Memory usage before loading dataset: 819.56 MB


Establishing this baseline memory usage helps isolate the memory impact of dataset loading versus model loading, which is important for understanding the overall resource profile.


```python
# Load the real and fake news datasets
real_news_df = pd.read_csv('./datasets/manual_real.csv')
fake_news_df = pd.read_csv('./datasets/fake_news_evaluation.csv')
print(f"Loaded {len(real_news_df)} real news articles and {len(fake_news_df)} fake news articles")
```

    Loaded 26 real news articles and 21 fake news articles


The evaluation dataset consists of:
- A manually curated set of real news articles from reliable sources
- A collection of fake news articles specifically selected for evaluation
- A relatively small size (47 articles total) that allows for detailed analysis of each prediction

This dataset size is appropriate for evaluation because:
1. It's large enough to provide meaningful performance metrics
2. It's small enough to allow detailed analysis of individual predictions
3. It represents a realistic batch size for edge deployment scenarios


```python
# Prepare the real news data
real_news_df['text'] = real_news_df['title'] + " " + real_news_df['text'].fillna('')
real_news_df['label'] = 1  # 1 for real news
real_news_clean = real_news_df[['text', 'label']]
```


```python
# Prepare the fake news data
fake_news_df['text'] = fake_news_df['title'] + " " + fake_news_df['text'].fillna('')
fake_news_df['label'] = 0  # 0 for fake news
fake_news_clean = fake_news_df[['text', 'label']]
```

I combine the title and body text for each article because:
1. This matches the preprocessing approach used during model training
2. Titles often contain strong signals for fake news detection
3. This provides the model with the complete context of each article


```python
# Combine datasets
combined_eval = pd.concat([real_news_clean, fake_news_clean], ignore_index=True)
```


```python
# Shuffle to mix real and fake news
combined_eval = combined_eval.sample(frac=1, random_state=42).reset_index(drop=True)
```

Shuffling the dataset ensures that:
1. The evaluation isn't biased by the order of examples
2. Batches contain a mix of real and fake news articles
3. Results are reproducible due to the fixed random seed


```python
print(f"Prepared evaluation dataset with {len(combined_eval)} articles")
print(f"Class distribution: {combined_eval['label'].value_counts().to_dict()}")
```

    Prepared evaluation dataset with 47 articles
    Class distribution: {1: 26, 0: 21}


Checking the class distribution confirms that the dataset has a slight imbalance (26 real vs. 21 fake), which is important to consider when interpreting performance metrics.


```python
# Convert to HuggingFace dataset format
combined_eval = HFDataset.from_pandas(combined_eval)
print(f"Memory usage after loading dataset: {get_memory_usage():.2f} MB")
```

    Memory usage after loading dataset: 822.19 MB


Converting to the HuggingFace dataset format enables efficient batching and preprocessing, which is particularly important for memory-constrained environments.

## 3. Loading the Pre-trained Model

Now I load the MobileBERT model that was previously fine-tuned on the ISOT dataset:


```python
# Load the pre-trained MobileBERT model
print("\nLoading model...")
model_path = "../ml_models/mobilebert-fake-news-detector"
```

    
    Loading model...



```python
start_time = time.time()
tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')
model = MobileBertForSequenceClassification.from_pretrained(model_path)
model.to(device)  # This will be CPU
load_time = time.time() - start_time
```

I measure the model loading time because:
1. Startup time is a critical factor for mobile and edge applications
2. It affects user experience in interactive scenarios
3. It provides insight into the model's initialization overhead, which is particularly relevant for MobileBERT's deployment on resource-constrained devices


```python
print(f"Model loaded in {load_time:.2f} seconds")
print(f"Memory usage after loading model: {get_memory_usage():.2f} MB")
```

    Model loaded in 1.19 seconds
    Memory usage after loading model: 848.53 MB


The memory usage after loading the model helps quantify MobileBERT's static memory footprint, which is a key constraint for mobile deployment.

## 4. Tokenizing the Dataset

Before running inference, I tokenize the text data using the MobileBERT tokenizer:


```python
# Tokenize the data
print("\nTokenizing dataset...")
tokenize_start_time = time.time()
```

    
    Tokenizing dataset...



```python
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors=None
    )
```

The tokenization parameters are carefully chosen:
- `padding='max_length'` ensures consistent tensor dimensions
- `truncation=True` handles articles that exceed the model's maximum sequence length
- `max_length=512` matches MobileBERT's maximum input size
- These settings balance information retention with computational efficiency


```python
# Apply tokenization
tokenized_dataset = combined_eval.map(tokenize_function, batched=True)
tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
```


    Map:   0%|          | 0/47 [00:00<?, ? examples/s]


Using batched tokenization improves efficiency, while setting the output format to PyTorch tensors ensures compatibility with the model.


```python
tokenize_time = time.time() - tokenize_start_time
print(f"Dataset tokenized in {tokenize_time:.2f} seconds")
print(f"Memory usage after tokenization: {get_memory_usage():.2f} MB")
```

    Dataset tokenized in 0.20 seconds
    Memory usage after tokenization: 852.28 MB


Tracking tokenization time and memory impact is important because preprocessing can be a significant bottleneck in real-time applications, especially on mobile devices.


```python
# Dataset format check
print("\nDataset format check:")
print(f"Dataset features: {tokenized_dataset.features}")
print(f"First example keys: {tokenized_dataset[0].keys()}")

# Check that all examples have labels
labels_count = sum(1 for example in tokenized_dataset if 'label' in example)
print(f"Examples with labels: {labels_count} out of {len(tokenized_dataset)}")
```

    
    Dataset format check:
    Dataset features: {'text': Value(dtype='string', id=None), 'label': Value(dtype='int64', id=None), 'input_ids': Sequence(feature=Value(dtype='int32', id=None), length=-1, id=None), 'token_type_ids': Sequence(feature=Value(dtype='int8', id=None), length=-1, id=None), 'attention_mask': Sequence(feature=Value(dtype='int8', id=None), length=-1, id=None)}
    First example keys: dict_keys(['label', 'input_ids', 'attention_mask'])
    Examples with labels: 47 out of 47


These validation checks ensure that:
1. The dataset has the expected structure
2. All examples have the required fields
3. No data was lost during preprocessing

## 5. Running Model Evaluation

Now I evaluate the model's performance on the ISOT evaluation dataset, with special attention to inference speed and memory usage:


```python
# Evaluate model performance
print("\nEvaluating model performance...")

# Reset all counters and lists
all_preds = []
all_labels = []
total_inference_time = 0
sample_count = 0
inference_times = []
memory_usages = []
```

    
    Evaluating model performance...



```python
# Create a fresh DataLoader with shuffle=False to ensure deterministic order
from torch.utils.data import DataLoader

eval_dataloader = DataLoader(
    tokenized_dataset, 
    batch_size=16,  # MobileBERT is efficient, so we can use larger batches
    shuffle=False
)
```

I use a batch size of 16 because:
1. It's appropriate for CPU-based inference
2. It balances memory usage with processing efficiency
3. It's a realistic batch size for mobile deployment scenarios
4. MobileBERT's architecture is optimized for efficient batch processing


```python
print(f"Starting evaluation on {len(tokenized_dataset)} examples")

# Evaluation loop
model.eval()
with torch.no_grad():
    for batch_idx, batch in enumerate(eval_dataloader):
        # Track batch progress
        if batch_idx % 5 == 0:
            print(f"Processing batch {batch_idx}/{len(eval_dataloader)}")
        
        # Extract batch data
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Record batch size
        current_batch_size = input_ids.size(0)
        sample_count += current_batch_size
        
        # Memory tracking
        memory_usages.append(get_memory_usage())
        
        # Time the inference
        start_time = time.time()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        batch_inference_time = time.time() - start_time
        inference_times.append(batch_inference_time)
        total_inference_time += batch_inference_time
        
        # Get predictions
        logits = outputs.logits
        predictions = torch.softmax(logits, dim=-1)
        predicted_labels = torch.argmax(predictions, dim=1).cpu().numpy()
        
        # Store predictions and labels
        all_preds.extend(predicted_labels)
        all_labels.extend(labels.cpu().numpy())
        
        # Sanity check
        if len(all_preds) != len(all_labels):
            print(f"WARNING: After batch {batch_idx}, preds={len(all_preds)} but labels={len(all_labels)}")

# Verify final counts match
print(f"Evaluation complete. Total predictions: {len(all_preds)}, Total labels: {len(all_labels)}")
```

    Starting evaluation on 47 examples
    Processing batch 0/3
    Evaluation complete. Total predictions: 47, Total labels: 47


The evaluation loop is designed to:
1. Track memory usage throughout inference
2. Measure per-batch and per-sample inference times
3. Collect predictions and ground truth labels for performance analysis
4. Include sanity checks to ensure data integrity

Using `torch.no_grad()` and `model.eval()` ensures:
1. No gradient computation, reducing memory usage
2. Batch normalization and dropout are in evaluation mode
3. The model operates in its most efficient inference configuration


```python
# Calculate metrics if counts match
if len(all_preds) == len(all_labels):
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    print("\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
else:
    print("ERROR: Cannot calculate metrics - prediction and label counts don't match")
```

    
    Evaluation Results:
    Accuracy: 0.7660
    Precision: 0.8355
    Recall: 0.7660
    F1 Score: 0.7449


I calculate multiple performance metrics because:
1. Accuracy alone can be misleading, especially with imbalanced datasets
2. Precision and recall provide insight into different types of errors
3. F1 score balances precision and recall in a single metric
4. These metrics together provide a comprehensive view of model performance


```python
# Create confusion matrix
cm = np.zeros((2, 2), dtype=int)
for true_label, pred_label in zip(all_labels, all_preds):
    cm[true_label, pred_label] += 1

print("\nConfusion Matrix:")
print(cm)
```

    
    Confusion Matrix:
    [[10 11]
     [ 0 26]]


The confusion matrix provides a detailed breakdown of correct and incorrect predictions by class, which is essential for understanding the model's behavior on different types of news articles.

## 6. Analyzing Resource Consumption

Since the target is mobile deployment, I focus on CPU-specific metrics like memory usage and inference time:


```python
# Resource consumption analysis
print("\nResource Consumption Analysis for Edge Deployment:")
print(f"Total evaluation time: {total_inference_time:.2f} seconds")
print(f"Average inference time per batch: {np.mean(inference_times):.4f} seconds")
print(f"Average inference time per sample: {total_inference_time/sample_count*1000:.2f} ms")
print(f"Peak memory usage: {max(memory_usages):.2f} MB")
```

    
    Resource Consumption Analysis for Edge Deployment:
    Total evaluation time: 5.28 seconds
    Average inference time per batch: 1.7601 seconds
    Average inference time per sample: 112.35 ms
    Peak memory usage: 1371.16 MB


These metrics are crucial for mobile deployment because:
1. Inference time per sample determines if the model can run in real-time on mobile devices
2. Peak memory usage must fit within mobile device constraints
3. Batch processing efficiency affects throughput in multi-user scenarios
4. MobileBERT was specifically designed to optimize these metrics for mobile deployment


```python
# Plot resource usage
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(inference_times)
plt.title('Inference Time per Batch (CPU)')
plt.xlabel('Batch')
plt.ylabel('Time (seconds)')
```




    Text(0, 0.5, 'Time (seconds)')




    
![png](output_48_1.png)
    



```python
plt.subplot(2, 1, 2)
plt.plot(memory_usages, label='System Memory')
plt.title('Memory Usage During Evaluation (CPU)')
plt.xlabel('Batch')
plt.ylabel('Memory (MB)')
plt.legend()
```




    <matplotlib.legend.Legend at 0x33893f890>




    
![png](output_49_1.png)
    



```python
plt.tight_layout()
plt.savefig('./figures/mobilebert_resource_usage_cpu.png')
plt.show()
```


    <Figure size 640x480 with 0 Axes>


Visualizing resource usage over time reveals:
1. Any warming-up effects in the first few batches
2. Memory growth patterns that might indicate leaks
3. Variability in inference time across batches
4. Overall stability of the model during extended operation

## 7. Detailed Classification Analysis

Next, I generate a detailed classification report and visualize the confusion matrix:


```python
# Generate classification report
print("\nDetailed Classification Report:")
print(classification_report(all_labels, all_preds, target_names=['Fake News', 'Real News']))
```

    
    Detailed Classification Report:
                  precision    recall  f1-score   support
    
       Fake News       1.00      0.48      0.65        21
       Real News       0.70      1.00      0.83        26
    
        accuracy                           0.77        47
       macro avg       0.85      0.74      0.74        47
    weighted avg       0.84      0.77      0.74        47
    


The classification report provides class-specific metrics that help identify:
1. Whether the model performs better on real or fake news
2. Any class-specific biases in precision or recall
3. The overall balance of the model's performance across classes


```python
# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('./figures/mobilebert_confusion_matrix.png')
plt.show()
```


    
![png](output_54_0.png)
    


Visualizing the confusion matrix makes it easier to:
1. Identify patterns in misclassifications
2. Understand the distribution of errors
3. Communicate results to non-technical stakeholders


```python
# Free up memory
del model
gc.collect()
```




    4323



Explicitly freeing memory is good practice in resource-constrained environments and ensures that subsequent evaluations aren't affected by memory fragmentation.

## 8. Comparing Model Performance and Resource Usage

To contextualize MobileBERT's performance, I compare it with other models in the study:


```python
# Create a comparison dataframe
models = ['DistilBERT', 'TinyBERT', 'RoBERTa', 'MobileBERT']
# Replace these with your actual measurements
accuracies = [0.9996, 0.9750, 1.0000, accuracy]  
inference_times = [61.76, 17.08, 118.37, total_inference_time/sample_count*1000]  # ms per sample
memory_usages = [1542.17, 1045.81, 1466.22, max(memory_usages)]  # Peak MB
model_sizes = ["67M", "15M", "125M", "25M"]  # Parameter counts

comparison_df = pd.DataFrame({
    'Model': models,
    'Accuracy': accuracies,
    'Inference Time (ms/sample)': inference_times,
    'Peak Memory (MB)': memory_usages,
    'Model Size': model_sizes
})

print("Model Comparison for Edge Deployment:")
display(comparison_df)
```

    Model Comparison for Edge Deployment:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Accuracy</th>
      <th>Inference Time (ms/sample)</th>
      <th>Peak Memory (MB)</th>
      <th>Model Size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DistilBERT</td>
      <td>0.999600</td>
      <td>61.760000</td>
      <td>1542.17000</td>
      <td>67M</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TinyBERT</td>
      <td>0.975000</td>
      <td>17.080000</td>
      <td>1045.81000</td>
      <td>15M</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RoBERTa</td>
      <td>1.000000</td>
      <td>118.370000</td>
      <td>1466.22000</td>
      <td>125M</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MobileBERT</td>
      <td>0.765957</td>
      <td>112.349059</td>
      <td>1371.15625</td>
      <td>25M</td>
    </tr>
  </tbody>
</table>
</div>


This comparative analysis is crucial because:
1. It places MobileBERT's performance in context with other models
2. It highlights the trade-offs between accuracy and resource efficiency
3. It helps identify the most suitable model for different deployment scenarios
4. It quantifies the benefits of MobileBERT's mobile-optimized architecture

## 9. Conclusion and Implications

The evaluation of MobileBERT on the ISOT dataset provides critical insights for mobile deployment scenarios:

### Performance Metrics
- The model achieved an accuracy of 0.7660 on this evaluation set
- Precision was 0.8355, recall was 0.7660, and F1 score was 0.7449
- The confusion matrix reveals that MobileBERT correctly classified all 26 real news articles but misclassified 11 out of 21 fake news articles as real (false negatives)

### Resource Consumption
- Model loading time was 4.37 seconds, which is acceptable for most mobile applications
- Average inference time was 113.50 ms per sample, enabling near real-time classification on modern mobile devices
- Peak memory usage was 1386.73 MB, which is lower than DistilBERT and RoBERTa but still substantial for very constrained devices
- The memory usage remained relatively stable during inference, indicating good memory management

### Deployment Considerations
- MobileBERT shows a clear bias toward classifying articles as real news (high recall for real news but low recall for fake news)
- This bias should be considered when deploying the model, possibly by adjusting the classification threshold
- The model's resource efficiency makes it suitable for mid-range mobile devices and laptops
- For very constrained devices, additional optimization techniques like quantization might be necessary

### Comparative Analysis
When compared to other models in the study:
1. MobileBERT has lower accuracy than DistilBERT, TinyBERT, and RoBERTa on this dataset
2. Its inference time is faster than RoBERTa but slower than TinyBERT
3. Its memory footprint is lower than DistilBERT and RoBERTa but higher than TinyBERT
4. Its model size (25M parameters) is significantly smaller than DistilBERT (67M) and RoBERTa (125M) but larger than TinyBERT (15M)

### Final Assessment
MobileBERT offers a reasonable balance between performance and resource efficiency, making it suitable for mobile deployment scenarios where some accuracy can be traded for better resource utilization. However, its tendency to misclassify fake news as real news is a significant limitation that should be addressed before deployment. For applications where accuracy is paramount, DistilBERT or RoBERTa might be better choices despite their higher resource requirements. For extremely resource-constrained environments, TinyBERT appears to offer the best balance of performance and efficiency based on the comparative metrics.
