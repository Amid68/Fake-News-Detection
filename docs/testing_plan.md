# Evaluation and Testing Plan

**Project Title:** Lightweight Fake News Detection System  
**Document Version:** 2.0  
**Author:** Ameed Othman   
**Date:** 27.03.2025

## 1. Introduction

### 1.1 Purpose
This document outlines the comprehensive testing and evaluation approach for the Lightweight Fake News Detection System. It establishes methodologies for assessing both technical functionality and the quality of fake news detection outputs, with special focus on resource efficiency and detection accuracy.

### 1.2 Scope
This plan covers:
- Testing strategies for all system components
- Evaluation methodologies for lightweight model detection quality
- Resource usage benchmarking
- User acceptance testing procedures
- Performance testing in resource-constrained environments
- Ongoing quality monitoring

### 1.3 References
- Project Vision Document v1.0
- Software Requirements Specification v1.0
- System Design Document v1.0

## 2. Testing Approach Overview

### 2.1 Testing Levels

#### 2.1.1 Unit Testing
- **Purpose:** Verify individual components function correctly in isolation
- **Coverage:** All modules, functions, and classes
- **Tools:** Pytest, Django test framework, Jest
- **Metrics:** Code coverage (target: >80%)

#### 2.1.2 Integration Testing
- **Purpose:** Verify components work together correctly
- **Coverage:** Component interactions, API contracts, data flow
- **Tools:** Pytest, API testing frameworks
- **Metrics:** API contract compliance, successful data flow

#### 2.1.3 System Testing
- **Purpose:** Verify complete system functionality
- **Coverage:** End-to-end workflows, error handling, edge cases
- **Tools:** Selenium, manual testing
- **Metrics:** Requirements satisfaction, workflow completion

#### 2.1.4 User Acceptance Testing
- **Purpose:** Validate system meets user needs
- **Coverage:** Key user journeys, usability, value delivery
- **Approach:** Structured sessions with representative users
- **Metrics:** User satisfaction, task completion rates

### 2.2 Testing Types

#### 2.2.1 Functional Testing
- User authentication
- Text input processing
- URL content extraction
- Fake news detection
- User preference management
- Results visualization

#### 2.2.2 Performance Testing
- Model inference time
- Memory usage during analysis
- CPU utilization
- Response time under load
- Resource usage in constrained environments

#### 2.2.3 Security Testing
- Authentication/authorization
- Input validation
- Data protection
- API security
- Common vulnerabilities (OWASP Top 10)

#### 2.2.4 Usability Testing
- UI intuitiveness
- Mobile responsiveness
- Accessibility compliance
- User journey completion

## 3. Model Evaluation

### 3.1 Fake News Detection Quality Evaluation

#### 3.1.1 Automated Metrics
- **Accuracy:** Percentage of correctly classified articles
- **Precision:** True positives / (True positives + False positives)
- **Recall:** True positives / (True positives + False negatives)
- **F1 Score:** Harmonic mean of precision and recall
- **ROC AUC:** Area under the Receiver Operating Characteristic curve
- **Target Benchmarks:**
  - Accuracy: >85%
  - F1 Score: >0.80
  - ROC AUC: >0.85

#### 3.1.2 Human Evaluation
- **Dimensions:**
  - **False Positive Rate:** How often legitimate content is flagged
  - **False Negative Rate:** How often fake content is missed
  - **Result Interpretation:** How clearly results are presented
  - **Confidence Calibration:** How well confidence scores align with accuracy
- **Methodology:**
  - Blind evaluation of 50+ articles with known ground truth
  - Minimum 3 evaluators per article
  - Comparison across different models
  - Focus on challenging edge cases
- **Success Criteria:**
  - False positive rate <15%
  - False negative rate <20%
  - >80% agreement with ground truth

#### 3.1.3 Test Dataset Creation
- Collection of 200 diverse news articles (100 real, 100 fake)
- Variety of topics, lengths, and sources
- Both obvious and subtle misinformation
- Articles not included in training data
- Stratified by difficulty level

### 3.2 Resource Efficiency Evaluation

#### 3.2.1 Memory Usage Benchmarking
- **Metrics:**
  - Peak memory usage during model loading
  - Average memory usage during inference
  - Memory leakage over multiple requests
- **Testing Methodology:**
  - Memory profiling with psutil
  - Stress testing with repeated requests
  - Testing on reference hardware platforms
- **Target Benchmarks:**
  - Peak memory: <500MB
  - Average memory: <250MB 
  - No measurable memory leaks

#### 3.2.2 Processing Speed Benchmarking
- **Metrics:**
  - Model loading time
  - Inference time per article
  - End-to-end processing time
- **Testing Methodology:**
  - Timed processing of standard article lengths
  - Distribution of processing times across test dataset
  - Comparison across different hardware profiles
- **Target Benchmarks:**
  - Model loading: <3 seconds
  - Inference time: <2 seconds per 1000 words
  - End-to-end processing: <5 seconds

#### 3.2.3 CPU Utilization
- **Metrics:**
  - Average CPU usage during inference
  - Peak CPU usage
  - CPU usage distribution over time
- **Testing Methodology:**
  - CPU profiling during standard workloads
  - Measurement on reference hardware
  - Multi-core vs. single-core performance
- **Target Benchmarks:**
  - Average CPU usage: <50% on reference hardware
  - Function on single-core processors
  - No processing timeouts

## 4. System Performance Evaluation

### 4.1 Response Time Benchmarks
- **Page Load Time:** <2 seconds for main interface
- **Text Processing Time:** <1 second for input processing
- **URL Extraction Time:** <3 seconds for content extraction
- **Model Inference Time:** <5 seconds for analysis
- **API Response Time:** <500ms for 95% of non-ML requests

### 4.2 Scalability Testing
- Concurrent user simulation (target: 10 users)
- Batch article processing (target: 20 articles/hour)
- Resource utilization monitoring
- Performance degradation patterns

### 4.3 Reliability Testing
- System uptime measurement (target: >99%)
- Error rate monitoring (target: <1% of requests)
- Recovery testing after resource exhaustion
- External API failure handling

## 5. Test Environments

### 5.1 Development Testing Environment
- Local development machines
- Docker containers for consistency
- SQLite database
- Mocked external services where appropriate

### 5.2 Integration Testing Environment
- Dedicated test server (minimal resources)
- Test database (SQLite)
- Controlled test data

### 5.3 Resource-Constrained Testing Environment
- 1GB RAM virtual machine
- Single-core CPU configuration
- Limited storage (2GB)
- Throttled network connection

## 6. Test Data Management

### 6.1 Test Data Requirements
- Representative user profiles
- Diverse news articles (real and fake)
- Various text lengths and complexities
- Edge cases and special scenarios
- Multilingual content (if supported)

### 6.2 Test Data Sources
- Manually created test cases
- Public fake news datasets:
  - LIAR dataset
  - FakeNewsNet
  - ISOT Fake News dataset
- Web-scraped recent news articles
- Synthetic data generation

### 6.3 Test Data Maintenance
- Version control for test datasets
- Automated setup and teardown
- Periodic refresh of test data
- Documentation of test data characteristics

## 7. Testing Process

### 7.1 Continuous Integration Testing
- Pre-commit hooks for linting and formatting
- Automated unit tests on commit
- Integration tests on pull request
- Daily full test suite execution

### 7.2 Release Testing
- Full regression test suite
- Performance benchmark validation
- Security scan
- Accessibility compliance check

### 7.3 User Acceptance Testing Process
1. Identify representative test users
2. Define test scenarios and success criteria
3. Conduct moderated testing sessions
4. Collect and analyze feedback
5. Prioritize and address issues

### 7.4 Bug Tracking and Resolution
- Severity classification system
- Reproduction steps documentation
- Root cause analysis
- Verification process for fixes

## 8. Model Evaluation Tools and Scripts

### 8.1 Fake News Detection Evaluation Script

```python
# Example fake news detection evaluation script
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model_name, y_true, y_pred, y_proba=None):
    """
    Evaluate a fake news detection model
    
    Args:
        model_name: Name of the model being evaluated
        y_true: Ground truth labels (0=real, 1=fake)
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional)
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Calculate classification metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # ROC AUC if probabilities provided
    auc_score = None
    if y_proba is not None:
        auc_score = roc_auc_score(y_true, y_proba)
    
    # Create confusion matrix visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f'confusion_matrix_{model_name}.png')
    
    # Compile results
    metrics = {
        'model': model_name,
        'accuracy': report['accuracy'],
        'precision_fake': report['1']['precision'],
        'recall_fake': report['1']['recall'],
        'f1_fake': report['1']['f1-score'],
        'precision_real': report['0']['precision'],
        'recall_real': report['0']['recall'],
        'f1_real': report['0']['f1-score'],
        'roc_auc': auc_score,
        'confusion_matrix': conf_matrix,
    }
    
    return metrics

def evaluate_all_models(models, test_data):
    """
    Evaluate multiple fake news detection models on test data
    
    Args:
        models: Dictionary of model objects
        test_data: Test dataset with texts and labels
    
    Returns:
        DataFrame with comparative metrics
    """
    results = []
    
    for model_name, model in models.items():
        # Get predictions
        y_true = test_data['label']
        y_pred = model.predict(test_data['text'])
        y_proba = None
        
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(test_data['text'])[:, 1]
        
        # Evaluate model
        metrics = evaluate_model(model_name, y_true, y_pred, y_proba)
        results.append(metrics)
    
    # Combine into DataFrame
    metrics_df = pd.DataFrame(results)
    
    # Create comparative visualization
    plt.figure(figsize=(12, 8))
    metrics_df[['model', 'accuracy', 'f1_fake', 'f1_real']].set_index('model').plot(kind='bar')
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.savefig('model_comparison.png')
    
    return metrics_df
```

### 8.2 Resource Usage Monitoring Script

```python
# Example resource monitoring script
import time
import psutil
import pandas as pd
import matplotlib.pyplot as plt
from functools import wraps

class ResourceMonitor:
    """Monitor CPU and memory usage during model execution"""
    
    def __init__(self, interval=0.1):
        """
        Initialize the resource monitor
        
        Args:
            interval: Sampling interval in seconds
        """
        self.interval = interval
        self.cpu_usage = []
        self.memory_usage = []
        self.timestamps = []
        self.running = False
        
    def start(self):
        """Start monitoring resources"""
        self.cpu_usage = []
        self.memory_usage = []
        self.timestamps = []
        self.start_time = time.time()
        self.running = True
        
        # Start monitoring in a separate thread
        import threading
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop monitoring resources"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        
    def _monitor(self):
        """Internal monitoring function"""
        while self.running:
            self.cpu_usage.append(psutil.cpu_percent())
            self.memory_usage.append(psutil.Process().memory_info().rss / 1024 / 1024)  # MB
            self.timestamps.append(time.time() - self.start_time)
            time.sleep(self.interval)
            
    def get_metrics(self):
        """Get the recorded metrics"""
        return {
            'timestamps': self.timestamps,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'peak_memory': max(self.memory_usage) if self.memory_usage else 0,
            'avg_memory': sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0,
            'peak_cpu': max(self.cpu_usage) if self.cpu_usage else 0,
            'avg_cpu': sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0,
        }
        
    def plot(self, save_path=None):
        """Plot the resource usage"""
        metrics = self.get_metrics()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot CPU usage
        ax1.plot(metrics['timestamps'], metrics['cpu_usage'])
        ax1.set_title('CPU Usage')
        ax1.set_ylabel('CPU %')
        ax1.grid(True)
        
        # Plot memory usage
        ax2.plot(metrics['timestamps'], metrics['memory_usage'])
        ax2.set_title('Memory Usage')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Memory (MB)')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
    def resource_usage_decorator(self, func):
        """Decorator to monitor resource usage of a function"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.start()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                self.stop()
                
        return wrapper


def benchmark_model(model, test_texts, n_runs=5):
    """
    Benchmark a model's resource usage and speed
    
    Args:
        model: The model to benchmark
        test_texts: List of texts to process
        n_runs: Number of benchmark runs
    
    Returns:
        Dictionary of benchmark results
    """
    # Initialize results
    load_times = []
    inference_times = []
    peak_memory = []
    avg_memory = []
    
    for run in range(n_runs):
        # Measure model loading
        monitor = ResourceMonitor()
        monitor.start()
        load_start = time.time()
        
        # Simulate model loading (replace with actual model loading code)
        # model.load()
        time.sleep(0.5)  # Placeholder
        
        load_end = time.time()
        monitor.stop()
        
        load_metrics = monitor.get_metrics()
        load_times.append(load_end - load_start)
        peak_memory.append(load_metrics['peak_memory'])
        
        # Measure inference
        inference_monitor = ResourceMonitor()
        all_inference_times = []
        
        for text in test_texts:
            inference_monitor.start()
            inference_start = time.time()
            
            # Run inference
            _ = model.predict([text])
            
            inference_end = time.time()
            inference_monitor.stop()
            
            all_inference_times.append(inference_end - inference_start)
            
        inference_metrics = inference_monitor.get_metrics()
        inference_times.append(sum(all_inference_times) / len(all_inference_times))
        avg_memory.append(inference_metrics['avg_memory'])
    
    # Compile results
    benchmark_results = {
        'avg_load_time': sum(load_times) / len(load_times),
        'avg_inference_time': sum(inference_times) / len(inference_times),
        'peak_memory_mb': sum(peak_memory) / len(peak_memory),
        'avg_memory_mb': sum(avg_memory) / len(avg_memory),
    }
    
    return benchmark_results
```

## 9. User Experience Testing

### 9.1 Usability Testing Methodology
- **Think-aloud Protocol:** Users verbalize thoughts while completing tasks
- **Task Completion Analysis:** Measure success rates and time
- **Post-task Questionnaires:** System Usability Scale (SUS)
- **Heatmap and Session Recording:** Visual analysis of user interaction

### 9.2 Key User Journeys for Testing
1. Registration and preference setting
2. Pasting text for analysis
3. Submitting URL for extraction and analysis
4. Interpreting detection results
5. Comparing model performance
6. Viewing analysis history

### 9.3 Accessibility Testing
- WCAG 2.1 AA compliance verification
- Screen reader compatibility
- Keyboard navigation testing
- Color contrast verification
- Font size and readability assessment

## 10. Test Reporting

### 10.1 Test Result Documentation
- Test coverage reports
- Pass/fail statistics
- Performance benchmark results
- Model evaluation metrics
- User acceptance testing findings

### 10.2 Defect Management
- Defect categorization by severity and component
- Root cause analysis for critical issues
- Resolution verification process
- Regression prevention strategy

### 10.3 Quality Metrics Dashboard
- Real-time test status visualization
- Trend analysis of key metrics
- Comparison against quality thresholds
- System health indicators

## 11. Continuous Improvement Process

### 11.1 Feedback Integration
- User feedback collection and analysis
- Bug report monitoring
- Performance issue tracking
- Detection quality assessment

### 11.2 Test Automation Enhancement
- Expanding automated test coverage
- Improving test execution efficiency
- Automating manual verification steps
- Enhancing test data generation

### 11.3 Model Evaluation Evolution
- Refining evaluation metrics
- Expanding test datasets
- Implementing new evaluation techniques
- Benchmarking against evolving state-of-the-art

## Appendices

### Appendix A: Test Case Templates

#### A.1 Functional Test Case Template
```
Test ID: FUNC-001
Test Title: User Registration
Preconditions: System is accessible, test user does not exist
Test Steps:
1. Navigate to registration page
2. Enter email and password
3. Submit registration form
4. Verify email (if required)
5. Log in with credentials
Expected Results:
- User account created successfully
- Confirmation message displayed
- User able to log in
```

#### A.2 Model Evaluation Test Case Template
```
Test ID: MODEL-001
Test Title: DistilBERT Model Accuracy Evaluation
Test Data: ISOT Fake News test dataset
Evaluation Metrics:
- Accuracy
- F1 Score
- Memory usage
- Inference time
Acceptance Criteria:
- Accuracy > 85%
- F1 Score > 0.80
- Memory usage < 500MB
- Inference time < 2s per article
```

### Appendix B: Human Evaluation Forms

#### B.1 Detection Quality Evaluation Form
```
Article ID: _________
Original Article Title: _________
Model Used: _________
Detection Result: _________
Confidence Score: _________

Please rate on a scale of 1-5 (1 = Poor, 5 = Excellent):

Detection Accuracy: How accurate was the model's assessment?
1 [ ] 2 [ ] 3 [ ] 4 [ ] 5 [ ]

Confidence Alignment: How well did the confidence score align with result quality?
1 [ ] 2 [ ] 3 [ ] 4 [ ] 5 [ ]

True Classification: In your assessment, this article is:
[ ] Genuine news  [ ] Fake news  [ ] Uncertain

Explanation: _________
```

#### B.2 Resource Efficiency Evaluation Form
```
Model ID: _________
Hardware Configuration: _________
Test Dataset: _________

Measured Metrics:
- Average Memory Usage: _________ MB
- Peak Memory Usage: _________ MB
- Average Inference Time: _________ seconds
- CPU Utilization: _________ %

Tester Assessment:
[ ] Acceptable performance for target hardware
[ ] Borderline performance - may struggle on target hardware
[ ] Unacceptable performance - exceeds resource constraints

Notes: _________
```

### Appendix C: Test Dataset Sources

1. **LIAR Dataset**
   - Source: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
   - Content: 12.8K labeled short statements
   - Usage: Model accuracy evaluation

2. **FakeNewsNet**
   - Source: https://github.com/KaiDMML/FakeNewsNet
   - Content: News content with social context
   - Usage: Comprehensive evaluation

3. **ISOT Fake News Dataset**
   - Source: https://www.uvic.ca/engineering/ece/isot/datasets/index.php
   - Content: 44,898 articles (21,417 real, 23,481 fake)
   - Usage: Large-scale testing

4. **Resource-Constrained Test Set**
   - Source: Internally developed
   - Content: 100 articles of varying length/complexity
   - Usage: Performance testing in resource-limited environments