# TinyBERT Evaluation for Fake News Detection

## Introduction

This notebook evaluates our fine-tuned TinyBERT model for fake news detection. TinyBERT is a compressed version of BERT that maintains strong performance while using fewer parameters, making it well-suited for deployment on edge devices with limited computational resources.

We'll perform a comprehensive evaluation focusing on:

1. Performance on the WELFake test dataset
2. Generalization capabilities when tested on external datasets containing verified real news and AI-generated fake news
3. Resource efficiency metrics critical for edge device deployment:
   - Memory footprint
   - Inference speed
   - Batch processing efficiency
   - Sequence length impact

The goal is to determine if transformer-based models like TinyBERT can effectively detect fake news while meeting the practical constraints of resource-limited environments.

## Setting Up the Environment

First, we'll import the necessary libraries for our evaluation:


```python
# Import basic libraries for data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import psutil
```


```python
# Import PyTorch and transformers
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
```


```python
# Import evaluation metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
```


```python
# Suppress warnings and set visualization style
import warnings
warnings.filterwarnings('ignore')

# Set consistent visualization style
plt.style.use('ggplot')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = (10, 6)

# Force CPU usage to simulate edge device performance
device = torch.device("cpu")
print(f"Using device: {device} (simulating edge device performance)")
```

    Using device: cpu (simulating edge device performance)


## Loading Data

### WELFake Test Set

We'll first load the WELFake dataset, which contains a balanced collection of real and fake news articles. This dataset combines articles from multiple sources: PolitiFact, GossipCop, Reuters, and BuzzFeed.


```python
# Load the WELFake dataset
df = pd.read_csv('../../data/WELFake_cleaned.csv')

# Combine title and text to provide complete information to the model
df['combined_text'] = df['title'] + " " + df['text']

# Prepare features and labels
X_welfake = df['combined_text']
y_welfake = df['label']
```


```python
# Split into train and test sets with stratification to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X_welfake, y_welfake, test_size=0.2, random_state=42, stratify=y_welfake
)

print(f"WELFake test set: {len(X_test)} articles")
```

    WELFake test set: 14308 articles


### External Datasets

To evaluate the model's generalization capabilities, we'll also test it on external datasets containing news articles not seen during training. This helps assess how well the model performs on real-world content.


```python
# Load external datasets
real_df = pd.read_csv('../datasets/manual_real.csv')
fake_df = pd.read_csv('../datasets/fake_claude.csv')
```


```python
# Process real news data
if 'title' in real_df.columns and 'content' in real_df.columns:
    real_df['combined_text'] = real_df['title'] + " " + real_df['content']
elif 'text' in real_df.columns:
    real_df['combined_text'] = real_df['text']
real_df['label'] = 0  # Real news

# Process fake news data
if 'title' in fake_df.columns and 'content' in fake_df.columns:
    fake_df['combined_text'] = fake_df['title'] + " " + fake_df['content']
elif 'text' in fake_df.columns:
    fake_df['combined_text'] = fake_df['text']
fake_df['label'] = 1  # Fake news
```


```python
# Combine external datasets
external_df = pd.concat(
    [real_df[['combined_text', 'label']], fake_df[['combined_text', 'label']]],
    ignore_index=True
)
X_external = external_df['combined_text']
y_external = external_df['label']

print(f"External dataset: {len(external_df)} articles ({len(real_df)} real, {len(fake_df)} fake)")
```

    External dataset: 828 articles (399 real, 429 fake)


## Loading and Measuring TinyBERT

Now we'll load our fine-tuned TinyBERT model and measure its resource requirements, which is critical information for edge deployment scenarios.


```python
# Measure memory before model loading
memory_before = psutil.Process().memory_info().rss / (1024 * 1024)  # MB

# Load the TinyBERT model and tokenizer
model_path = '../../ml_models/tinybert_welfake_model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model = model.to(device)
```


```python
# Measure memory after model loading
memory_after = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
model_memory = memory_after - memory_before

# Calculate model size from parameters
param_size = sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 * 1024)
num_params = sum(p.numel() for p in model.parameters())

print(f"TinyBERT model loaded successfully")
print(f"Number of parameters: {num_params:,}")
print(f"Model size: {param_size:.2f} MB")
print(f"Memory increase after loading: {model_memory:.2f} MB")
```

    TinyBERT model loaded successfully
    Number of parameters: 14,350,874
    Model size: 54.74 MB
    Memory increase after loading: 414.67 MB


## Preparing Data for Evaluation

Before we can evaluate the model, we need to tokenize our text data and prepare it in the format expected by the transformer model.


```python
def prepare_data(texts, labels, tokenizer, batch_size=32):
    """
    Tokenize text data and create DataLoader for model input
    
    Args:
        texts: List or Series of text samples
        labels: List or Series of labels
        tokenizer: The tokenizer to use
        batch_size: Batch size for DataLoader
        
    Returns:
        DataLoader with tokenized inputs and labels
    """
    # Tokenize the text
    encodings = tokenizer(
        list(texts),
        truncation=True,
        padding='max_length',
        max_length=512,  # Standard for BERT models
        return_tensors='pt'
    )
    
    # Create dataset and dataloader
    dataset = TensorDataset(
        encodings['input_ids'],
        encodings['attention_mask'],
        torch.tensor(labels.values if hasattr(labels, 'values') else labels)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return dataloader
```


```python
# Prepare test sets
welfake_test_loader = prepare_data(X_test, y_test, tokenizer)
external_loader = prepare_data(X_external, y_external, tokenizer)
```

## Evaluation Function

We'll define a comprehensive evaluation function that measures both performance metrics and resource usage.


```python
def evaluate_model(model, dataloader, dataset_name):
    """
    Evaluate model and measure performance metrics and resource usage
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader with test data
        dataset_name: Name of the dataset for reporting
        
    Returns:
        Dictionary with performance metrics and resource usage
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    # Measure resources
    memory_before = psutil.Process().memory_info().rss / (1024 * 1024)
    start_time = time.time()
    
    # Make predictions
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate resource usage
    predict_time = time.time() - start_time
    memory_after = psutil.Process().memory_info().rss / (1024 * 1024)
    memory_used = memory_after - memory_before
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    # Print results
    print(f"\nTinyBERT Evaluation on {dataset_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Prediction time: {predict_time:.2f} seconds for {len(all_labels)} samples")
    print(f"Average prediction time: {predict_time/len(all_labels)*1000:.2f} ms per sample")
    print(f"Memory used during inference: {memory_used:.2f} MB")
    
    # Return results for visualization
    return {
        'y_pred': all_preds,
        'y_true': all_labels,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predict_time': predict_time,
        'samples': len(all_labels),
        'memory_used': memory_used
    }
```

## Performance on WELFake Test Set

Now we'll evaluate TinyBERT on the WELFake test set to measure its performance on data similar to what it was trained on.


```python
# Evaluate on WELFake test set
welfake_results = evaluate_model(model, welfake_test_loader, "WELFake Test Set")
```

    
    TinyBERT Evaluation on WELFake Test Set:
    Accuracy: 0.9931
    Precision: 0.9931
    Recall: 0.9931
    F1 Score: 0.9931
    Prediction time: 227.41 seconds for 14308 samples
    Average prediction time: 15.89 ms per sample
    Memory used during inference: -93.17 MB


### Confusion Matrix for WELFake

Visualizing the confusion matrix helps us understand where the model makes errors and whether there are any patterns in its mistakes.


```python
# Create and plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    """
    Create and visualize confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real News', 'Fake News'],
                yticklabels=['Real News', 'Fake News'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    # Calculate error rates
    tn, fp, fn, tp = cm.ravel()
    fpr = fp/(fp+tn)
    fnr = fn/(fn+tp)
    print(f"False Positive Rate: {fpr:.4f} ({fp} real news articles misclassified as fake)")
    print(f"False Negative Rate: {fnr:.4f} ({fn} fake news articles misclassified as real)")
```


```python
# Plot confusion matrix for WELFake
plot_confusion_matrix(
    welfake_results['y_true'], 
    welfake_results['y_pred'], 
    "TinyBERT Confusion Matrix on WELFake Test Set"
)
```


    
![png](output_24_0.png)
    


    False Positive Rate: 0.0040 (28 real news articles misclassified as fake)
    False Negative Rate: 0.0097 (71 fake news articles misclassified as real)


## Performance on External Datasets

To assess how well the model generalizes to new, unseen data, we'll evaluate it on our external datasets.


```python
# Evaluate on external datasets
external_results = evaluate_model(model, external_loader, "External Datasets")
```

    
    TinyBERT Evaluation on External Datasets:
    Accuracy: 0.8370
    Precision: 0.8731
    Recall: 0.8370
    F1 Score: 0.8340
    Prediction time: 11.94 seconds for 828 samples
    Average prediction time: 14.42 ms per sample
    Memory used during inference: -72.64 MB


### Confusion Matrix for External Data


```python
# Plot confusion matrix for External Datasets
plot_confusion_matrix(
    external_results['y_true'], 
    external_results['y_pred'], 
    "TinyBERT Confusion Matrix on External Datasets"
)
```


    
![png](output_28_0.png)
    


    False Positive Rate: 0.0100 (4 real news articles misclassified as fake)
    False Negative Rate: 0.3054 (131 fake news articles misclassified as real)


## Analyzing Misclassified Examples

Understanding specific cases where the model fails can provide insights into its limitations and potential areas for improvement.


```python
def analyze_errors(X_text, y_true, y_pred, dataset_name, n_examples=3):
    """
    Display examples of misclassified articles
    
    Args:
        X_text: Text data
        y_true: True labels
        y_pred: Predicted labels
        dataset_name: Name of the dataset
        n_examples: Number of examples to display
    """
    errors = np.where(y_true != y_pred)[0]
    
    if len(errors) == 0:
        print(f"No errors found on {dataset_name}!")
        return
    
    print(f"\nTinyBERT misclassified {len(errors)} out of {len(y_true)} articles on {dataset_name} ({len(errors)/len(y_true):.2%})")
    print(f"Showing {min(n_examples, len(errors))} examples:")
    
    # Select random errors to display
    np.random.seed(42)  # For reproducibility
    display_indices = np.random.choice(errors, size=min(n_examples, len(errors)), replace=False)
    
    for i, idx in enumerate(display_indices):
        print(f"\nExample {i+1}:")
        print(f"Text snippet: {X_text.iloc[idx][:200]}...")  # First 200 chars
        print(f"True label: {'Real' if y_true[idx] == 0 else 'Fake'}")
        print(f"Predicted: {'Real' if y_pred[idx] == 0 else 'Fake'}")
        print("-" * 80)
```


```python
# Analyze errors on WELFake
analyze_errors(
    X_test, 
    welfake_results['y_true'], 
    welfake_results['y_pred'], 
    "WELFake Test Set"
)
```

    
    TinyBERT misclassified 99 out of 14308 articles on WELFake Test Set (0.69%)
    Showing 3 examples:
    
    Example 1:
    Text snippet: Trump’s Ex-Manager: He Didn’t Even Know What A Gold Star Family Is Comments 
    Donald Trump, the draft-dodging Russophile who cloaks his xenophobia and bigotry in a veneer of false patriotism, reportedl...
    True label: Fake
    Predicted: Real
    --------------------------------------------------------------------------------
    
    Example 2:
    Text snippet: Hillary Clinton and Bernie Sanders = lecture vs. rock concert That's one way to quickly characterize the difference between a campaign stop for Hillary Clinton and Bernie Sanders.
    
    When Democrats visi...
    True label: Real
    Predicted: Fake
    --------------------------------------------------------------------------------
    
    Example 3:
    Text snippet: Round I in Iowa: Scott Walker Emerges Scott Walker 2016 begins today. After this speech to #FreedomSummit no way he doesn't become formidable presidential candidate.
    
    So, @ScottWalker is the breakout ...
    True label: Real
    Predicted: Fake
    --------------------------------------------------------------------------------



```python
# Analyze errors on External datasets
analyze_errors(
    X_external, 
    external_results['y_true'], 
    external_results['y_pred'], 
    "External Datasets"
)
```

    
    TinyBERT misclassified 135 out of 828 articles on External Datasets (16.30%)
    Showing 3 examples:
    
    Example 1:
    Text snippet: Senior banking officials speaking on condition of anonymity reveal that Federal Reserve leadership has drafted plans for a national banking 'holiday' to be implemented next month as part of final prep...
    True label: Fake
    Predicted: Real
    --------------------------------------------------------------------------------
    
    Example 2:
    Text snippet: A former senior vice president from a major pharmaceutical company has come forward with evidence allegedly proving that a remarkably effective cancer treatment was deliberately shelved after internal...
    True label: Fake
    Predicted: Real
    --------------------------------------------------------------------------------
    
    Example 3:
    Text snippet: A senior research scientist formerly employed at the Centers for Disease Control has provided extensive documentation allegedly showing systematic manipulation of vaccine safety data to artificially m...
    True label: Fake
    Predicted: Real
    --------------------------------------------------------------------------------


## Edge Device Performance Analysis

For deployment on resource-constrained edge devices, understanding how batch size affects inference efficiency is critical for optimizing throughput versus latency trade-offs.


```python
# Analyze batch processing efficiency
batch_sizes = [1, 2, 4, 8, 16, 32]
results = []

# Create sample input
sample_text = ["This is a sample news article for testing inference speed."] * 32
sample_encodings = tokenizer(
    sample_text,
    truncation=True,
    padding='max_length',
    max_length=512,
    return_tensors='pt'
)
```


```python
# Test different batch sizes
for batch_size in batch_sizes:
    # Prepare input batch
    input_ids = sample_encodings['input_ids'][:batch_size].to(device)
    attention_mask = sample_encodings['attention_mask'][:batch_size].to(device)
    
    # Warm-up
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Timed runs
    times = []
    for _ in range(5):  # 5 runs per batch size
        with torch.no_grad():
            start = time.time()
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            end = time.time()
        times.append(end - start)
    
    # Calculate statistics
    avg_time = np.mean(times)
    per_sample = avg_time / batch_size * 1000  # ms
    
    results.append({
        'Batch Size': batch_size,
        'Total Time (ms)': avg_time * 1000,
        'Time per Sample (ms)': per_sample
    })
```


```python
# Show batch efficiency results
batch_df = pd.DataFrame(results)
print("\nBatch Processing Efficiency on CPU:")
print(batch_df.round(2))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(batch_df['Batch Size'], batch_df['Time per Sample (ms)'], marker='o', linewidth=2)
plt.title('Inference Time per Sample vs Batch Size')
plt.xlabel('Batch Size')
plt.ylabel('Time per Sample (ms)')
plt.grid(True)
plt.tight_layout()
plt.show()
```

    
    Batch Processing Efficiency on CPU:
       Batch Size  Total Time (ms)  Time per Sample (ms)
    0           1            25.61                 25.61
    1           2            31.72                 15.86
    2           4            56.30                 14.08
    3           8           113.84                 14.23
    4          16           219.13                 13.70
    5          32           448.82                 14.03



    
![png](output_36_1.png)
    


## Measuring Memory Usage for Different Sequence Lengths

The memory usage of transformer models depends significantly on sequence length due to the self-attention mechanism. This analysis helps determine optimal sequence lengths for memory-constrained deployments.


```python
# Analyze memory usage for different sequence lengths
seq_lengths = [64, 128, 256, 512]
memory_results = []
```


```python
for seq_len in seq_lengths:
    # Create sample input with specific sequence length
    sample_text = ["This is a test"] * 8  # Use batch size of 8
    sample_encodings = tokenizer(
        sample_text,
        truncation=True,
        padding='max_length',
        max_length=seq_len,
        return_tensors='pt'
    )
    
    input_ids = sample_encodings['input_ids'].to(device)
    attention_mask = sample_encodings['attention_mask'].to(device)
    
    # Measure memory before inference
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    memory_before = psutil.Process().memory_info().rss / (1024 * 1024)
    
    # Run inference
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Measure memory after inference
    memory_after = psutil.Process().memory_info().rss / (1024 * 1024)
    memory_used = memory_after - memory_before
    
    memory_results.append({
        'Sequence Length': seq_len,
        'Memory Used (MB)': memory_used
    })
```


```python
# Show memory usage results
memory_df = pd.DataFrame(memory_results)
print("\nMemory Usage for Different Sequence Lengths:")
print(memory_df)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(memory_df['Sequence Length'], memory_df['Memory Used (MB)'], marker='o', linewidth=2)
plt.title('Memory Usage vs Sequence Length')
plt.xlabel('Sequence Length')
plt.ylabel('Memory Used (MB)')
plt.grid(True)
plt.tight_layout()
plt.show()
```

    
    Memory Usage for Different Sequence Lengths:
       Sequence Length  Memory Used (MB)
    0               64           0.03125
    1              128           0.00000
    2              256          10.40625
    3              512          19.50000



    
![png](output_40_1.png)
    


## Summary

Let's compile our key findings into a comprehensive summary table to better understand TinyBERT's performance and resource characteristics.


```python
# Create summary table of results
summary = pd.DataFrame({
    'Metric': [
        'Accuracy', 
        'Precision', 
        'Recall', 
        'F1 Score',
        'Inference Time (ms/sample)',
        'Memory Footprint (MB)',
        'Parameter Count'
    ],
    'WELFake Test': [
        f"{welfake_results['accuracy']:.4f}",
        f"{welfake_results['precision']:.4f}",
        f"{welfake_results['recall']:.4f}",
        f"{welfake_results['f1']:.4f}",
        f"{welfake_results['predict_time']/welfake_results['samples']*1000:.2f}",
        f"{model_memory:.2f}",
        f"{num_params:,}"
    ],
    'External Data': [
        f"{external_results['accuracy']:.4f}",
        f"{external_results['precision']:.4f}",
        f"{external_results['recall']:.4f}",
        f"{external_results['f1']:.4f}",
        f"{external_results['predict_time']/external_results['samples']*1000:.2f}",
        f"{model_memory:.2f}",
        f"{num_params:,}"
    ]
})

print("TinyBERT Performance and Resource Usage Summary:")
print(summary)
```

    TinyBERT Performance and Resource Usage Summary:
                           Metric WELFake Test External Data
    0                    Accuracy       0.9931        0.8370
    1                   Precision       0.9931        0.8731
    2                      Recall       0.9931        0.8370
    3                    F1 Score       0.9931        0.8340
    4  Inference Time (ms/sample)        15.89         14.42
    5       Memory Footprint (MB)       414.67        414.67
    6             Parameter Count   14,350,874    14,350,874


## Conclusion

This evaluation demonstrates that TinyBERT can effectively detect fake news while maintaining reasonable resource requirements for edge deployment. The model shows exceptional performance on the WELFake test set with 99.31% accuracy and generalizes reasonably well to external data with 83.70% accuracy.

Key findings:

1. **Performance**:
   - Near-perfect accuracy on the WELFake test set
   - Good but reduced performance on external datasets, with a notable tendency to misclassify fake news as real (30.5% false negative rate)

2. **Resource Efficiency**:
   - Parameter count of 14.35 million (7.5x smaller than BERT-base)
   - Memory footprint of approximately 414 MB
   - Per-sample inference time of 14 ms with optimal batch sizes

3. **Optimization Opportunities**:
   - Batch processing significantly improves efficiency (26 ms → 14 ms per sample)
   - Shorter sequence lengths dramatically reduce memory requirements
   - For extremely resource-constrained environments, using sequence lengths of 256 or less offers significant memory savings

These results suggest that transformer-based models like TinyBERT can be successfully deployed on edge devices for fake news detection, bringing advanced NLP capabilities to resource-constrained environments. However, further research is needed to improve generalization to previously unseen fake news patterns and to optimize performance for specific deployment constraints.
