# Evaluation and Testing Plan

**Project Title:** Automated Multilingual News Aggregation, Summarization & Bias Detection Tool  
**Document Version:** 1.0
**Author:** Ameed Othman   
**Date:** 04.03.2025

## 1. Introduction

### 1.1 Purpose
This document outlines the comprehensive testing and evaluation approach for the Automated News Aggregation, Summarization, and Bias Detection Tool. It establishes methodologies for assessing both technical functionality and the quality of AI-generated outputs, ensuring the system meets its requirements and delivers value to users.

### 1.2 Scope
This plan covers:
- Testing strategies for all system components
- Evaluation methodologies for LLM-generated content
- User acceptance testing procedures
- Performance benchmarking
- Ongoing quality monitoring

### 1.3 References
- Project Vision Document v2.0
- Software Requirements Specification v2.0
- System Design Document v2.0

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
- Authentication
- News aggregation
- Summarization
- Bias detection
- News feed generation
- User preference management

#### 2.2.2 Performance Testing
- Load testing
- Response time
- Resource utilization
- Scalability limits

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

## 3. LLM Output Evaluation

### 3.1 Summarization Quality Evaluation

#### 3.1.1 Automated Metrics
- **ROUGE Scores:** Measure overlap with reference summaries
  - ROUGE-1: Unigram overlap
  - ROUGE-2: Bigram overlap
  - ROUGE-L: Longest common subsequence
- **BERTScore:** Semantic similarity to reference
- **Target Benchmarks:**
  - ROUGE-1: >0.40
  - ROUGE-2: >0.20
  - ROUGE-L: >0.35
  - BERTScore: >0.85

#### 3.1.2 Human Evaluation
- **Dimensions:**
  - **Informativeness:** Does the summary contain key information?
  - **Coherence:** Is the summary well-structured and readable?
  - **Factual Accuracy:** Does the summary contain factual errors?
  - **Conciseness:** Is the summary appropriately brief?
- **Methodology:**
  - 5-point Likert scale for each dimension
  - Blind comparison with reference summaries
  - Minimum 50 articles evaluated
  - At least 3 evaluators per article
- **Success Criteria:**
  - Average score ≥4.0 on all dimensions
  - ≥90% of summaries without factual errors

#### 3.1.3 Test Dataset Creation
- Collection of 100 diverse news articles
- Human-written reference summaries
- Variety of topics, lengths, and sources
- Both straightforward and complex articles

### 3.2 Bias Detection Evaluation

#### 3.2.1 Benchmark Dataset
- Media Bias/Fact Check (MBFC) labeled articles
- AllSides Media Bias Ratings
- Ad Fontes Media Bias Chart
- Custom dataset with expert annotations (if feasible)

#### 3.2.2 Classification Metrics
- **Accuracy:** Overall correct classifications
- **Precision:** Correct positive predictions / all positive predictions
- **Recall:** Correct positive predictions / all actual positives
- **F1 Score:** Harmonic mean of precision and recall
- **Target benchmarks:**
  - Accuracy: >0.70
  - F1 Score: >0.65

#### 3.2.3 Human Validation
- **Methodology:**
  - Blind comparison of system vs. human expert classifications
  - Structured disagreement analysis
  - Confidence assessment
- **Success Criteria:**
  - >70% agreement with human experts
  - Confidence correctly calibrated (higher confidence correlates with accuracy)

#### 3.2.4 Error Analysis
- Systematic categorization of errors
- Identification of bias blindspots
- Documentation of challenging cases for future improvement

## 4. System Performance Evaluation

### 4.1 Response Time Benchmarks
- **Page Load Time:** <3 seconds for news feed
- **Article Processing Time:**
  - Summarization: <10 seconds per article
  - Bias Detection: <5 seconds per article
- **API Response Time:** <500ms for 95% of requests

### 4.2 Scalability Testing
- Concurrent user simulation (target: 50 users)
- Batch article processing (target: 100 articles/hour)
- Database query performance under load
- Resource utilization monitoring

### 4.3 Reliability Testing
- System uptime measurement (target: >99%)
- Error rate monitoring (target: <1% of requests)
- Failover and recovery testing
- External API failure handling

## 5. Test Environments

### 5.1 Development Testing Environment
- Local development machines
- Docker containers for consistency
- SQLite database
- Mocked external services where appropriate

### 5.2 Integration Testing Environment
- Dedicated test server
- Test database (PostgreSQL)
- Limited external service integration
- Controlled test data

### 5.3 Production-like Testing Environment
- Cloud-based staging environment
- Production-equivalent configuration
- Full external service integration
- Anonymized production-like data

## 6. Test Data Management

### 6.1 Test Data Requirements
- Representative user profiles
- Diverse news articles
- Various topic distributions
- Edge cases and special scenarios

### 6.2 Test Data Sources
- Manually created test cases
- Anonymized production data (when available)
- Public news datasets
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

## 8. LLM Evaluation Tools and Scripts

### 8.1 Summarization Evaluation Script

```python
# Example summarization evaluation script
import rouge
from bert_score import score
import pandas as pd

def evaluate_summaries(generated_summaries, reference_summaries):
    """
    Evaluate generated summaries against references using ROUGE and BERTScore
    
    Args:
        generated_summaries: List of generated summaries
        reference_summaries: List of reference summaries
    
    Returns:
        DataFrame with evaluation metrics
    """
    # Calculate ROUGE scores
    rouge_evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                                 max_n=2,
                                 limit_length=True,
                                 length_limit=100,
                                 length_limit_type='words',
                                 apply_avg=True,
                                 apply_best=False)
    
    rouge_scores = rouge_evaluator.evaluate(generated_summaries, reference_summaries)
    
    # Calculate BERTScore
    P, R, F1 = score(generated_summaries, reference_summaries, lang='en', verbose=True)
    bert_scores = F1.numpy()
    
    # Compile results
    results = {
        'rouge-1': rouge_scores['rouge-1']['f'],
        'rouge-2': rouge_scores['rouge-2']['f'],
        'rouge-l': rouge_scores['rouge-l']['f'],
        'bert_score': bert_scores.mean()
    }
    
    return pd.DataFrame([results])
```

### 8.2 Bias Detection Evaluation Script

```python
# Example bias detection evaluation script
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_bias_detection(predicted_labels, true_labels, predicted_confidences=None):
    """
    Evaluate bias detection performance
    
    Args:
        predicted_labels: Model predictions (e.g., 'left', 'center', 'right')
        true_labels: Ground truth labels
        predicted_confidences: Confidence scores for predictions (optional)
    
    Returns:
        Dictionary with evaluation metrics and plots
    """
    # Calculate classification metrics
    report = classification_report(true_labels, predicted_labels, output_dict=True)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    
    # Create confusion matrix heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(set(true_labels)),
                yticklabels=sorted(set(true_labels)))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Bias Detection Confusion Matrix')
    
    # Evaluate confidence calibration if available
    calibration_data = None
    if predicted_confidences is not None:
        # Convert to numpy arrays
        confidences = np.array(predicted_confidences)
        correct = (np.array(predicted_labels) == np.array(true_labels)).astype(int)
        
        # Create calibration curve
        confidence_bins = np.linspace(0, 1, 11)
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(len(confidence_bins) - 1):
            bin_mask = (confidences >= confidence_bins[i]) & (confidences < confidence_bins[i+1])
            if np.sum(bin_mask) > 0:
                bin_accuracies.append(np.mean(correct[bin_mask]))
                bin_confidences.append(np.mean(confidences[bin_mask]))
                bin_counts.append(np.sum(bin_mask))
        
        calibration_data = pd.DataFrame({
            'bin_accuracy': bin_accuracies,
            'bin_confidence': bin_confidences,
            'bin_count': bin_counts
        })
    
    return {
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'calibration_data': calibration_data
    }
```

## 9. User Experience Testing

### 9.1 Usability Testing Methodology
- **Think-aloud Protocol:** Users verbalize thoughts while completing tasks
- **Task Completion Analysis:** Measure success rates and time
- **Post-task Questionnaires:** System Usability Scale (SUS)
- **Heatmap and Session Recording:** Visual analysis of user interaction

### 9.2 Key User Journeys for Testing
1. Registration and preference setting
2. Browsing personalized news feed
3. Reading article summaries and bias assessments
4. Adjusting preferences and observing feed changes
5. Searching for specific topics
6. Saving and retrieving articles

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
- LLM evaluation metrics
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
- LLM output quality assessment

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

#### A.2 LLM Evaluation Test Case Template
```
Test ID: LLM-001
Test Title: News Article Summarization Quality
Test Data: Article ID A123 from test dataset
Evaluation Metrics:
- ROUGE-1, ROUGE-2, ROUGE-L scores
- BERTScore
- Human evaluation (informativeness, coherence, accuracy)
Acceptance Criteria:
- ROUGE-L score > 0.35
- No factual errors
- Human rating average > 4.0
```

### Appendix B: Human Evaluation Forms

#### B.1 Summarization Quality Evaluation Form
```
Article ID: _________
Original Article Title: _________
Generated Summary: _________

Please rate on a scale of 1-5 (1 = Poor, 5 = Excellent):

Informativeness: How well does the summary capture key information?
1 [ ] 2 [ ] 3 [ ] 4 [ ] 5 [ ]

Coherence: How well-structured and readable is the summary?
1 [ ] 2 [ ] 3 [ ] 4 [ ] 5 [ ]

Factual Accuracy: Does the summary contain factual errors?
No errors [ ] Minor errors [ ] Major errors [ ]

Conciseness: Is the summary appropriately brief?
1 [ ] 2 [ ] 3 [ ] 4 [ ] 5 [ ]

Additional Comments: _________
```

#### B.2 Bias Detection Evaluation Form
```
Article ID: _________
Article Title: _________
System's Bias Rating: _________
Confidence Score: _________

Your Assessment:
[ ] Strong Left   [ ] Moderate Left   [ ] Neutral   [ ] Moderate Right   [ ] Strong Right

Agreement with System:
[ ] Strongly Disagree   [ ] Disagree   [ ] Neutral   [ ] Agree   [ ] Strongly Agree

Confidence in Your Assessment:
[ ] Low   [ ] Medium   [ ] High

Explanation for Disagreement (if any): _________
```

### Appendix C: Test Dataset Sources

1. **AllSides Balanced News Dataset**
   - Source: https://www.allsides.com/
   - Content: News articles with bias ratings
   - Usage: Bias detection evaluation

2. **CNN/Daily Mail Dataset**
   - Source: https://github.com/abisee/cnn-dailymail
   - Content: News articles with human-written summaries
   - Usage: Summarization evaluation

3. **Media Bias/Fact Check (MBFC)**
   - Source: https://mediabiasfactcheck.com/
   - Content: Source-level bias ratings
   - Usage: Bias detection calibration

4. **NELA-GT-2020**
   - Source: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ZCXSKG
   - Content: News articles with source-level annotations
   - Usage: News aggregation and classification testing