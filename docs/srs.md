# Software Requirements Specification

**Project Title:** Lightweight Fake News Detection System  
**Document Version:** 4.0  
**Author:** Ameed Othman
**Date:** 29.03.2025

## 1. Introduction

### 1.1 Purpose
This document defines the functional and non-functional requirements for the Lightweight Fake News Detection System. It serves as a comprehensive guide for implementation, focusing on efficient misinformation detection in resource-constrained environments.

### 1.2 Scope
This specification covers the Minimum Viable Product (MVP) scope, focusing on:
- User authentication and preference management
- Text input and URL processing for news articles
- Fake news detection using lightweight pre-trained models
- Model comparison and performance metrics
- Resource usage monitoring and optimization
- Web-based user interface

### 1.3 Definitions, Acronyms, and Abbreviations
- **API:** Application Programming Interface
- **FND:** Fake News Detection
- **JWT:** JSON Web Token
- **LLM:** Large Language Model
- **MVP:** Minimum Viable Product
- **NLP:** Natural Language Processing
- **PLM:** Pre-trained Language Model
- **UI/UX:** User Interface/User Experience

### 1.4 References
- Project Vision Document v1.0
- System Design Document v1.0
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [FakeNewsNet Dataset](https://github.com/KaiDMML/FakeNewsNet)
- [LIAR Dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)

## 2. Functional Requirements

### 2.1 User Authentication and Profile Management

#### 2.1.1 User Registration
- **FR-1.1:** The system shall allow users to register with email and password.
- **FR-1.2:** The system shall validate email format and uniqueness.
- **FR-1.3:** The system shall enforce password complexity requirements (min. 8 characters, including numbers/symbols).
- **FR-1.4:** The system shall store passwords using secure hashing (bcrypt).
- **FR-1.5:** The system shall confirm successful registration to the user.

#### 2.1.2 User Authentication
- **FR-2.1:** The system shall authenticate users via email/password.
- **FR-2.2:** The system shall provide secure session management using JWT.
- **FR-2.3:** The system shall implement password reset functionality.
- **FR-2.4:** The system shall log failed login attempts.
- **FR-2.5:** The system shall allow users to log out from any page.

#### 2.1.3 User Preferences
- **FR-3.1:** The system shall allow users to select a default detection model.
- **FR-3.2:** The system shall allow users to toggle detailed metrics display.
- **FR-3.3:** The system shall store user preferences securely.
- **FR-3.4:** The system shall apply user preferences consistently across sessions.

### 2.2 Content Input and Processing

#### 2.2.1 Text Input
- **FR-4.1:** The system shall provide a text input area for pasting news content.
- **FR-4.2:** The system shall accept text input of at least 5,000 characters.
- **FR-4.3:** The system shall validate and sanitize text input.
- **FR-4.4:** The system shall provide a clear button to reset the input area.
- **FR-4.5:** The system shall preserve input text if analysis fails.

#### 2.2.2 URL Processing
- **FR-5.1:** The system shall provide a URL input field for news article links.
- **FR-5.2:** The system shall validate URL format.
- **FR-5.3:** The system shall extract text content from provided URLs.
- **FR-5.4:** The system shall handle common news site formats.
- **FR-5.5:** The system shall display extracted text before analysis.
- **FR-5.6:** The system shall handle and report URL access errors gracefully.

### 2.3 Fake News Detection

#### 2.3.1 Model Integration
- **FR-6.1:** The system shall integrate at least 3 lightweight pre-trained models for FND.
- **FR-6.2:** The system shall include the following models:
  - DistilBERT-based fake news detector
  - TinyBERT-based fake news detector
  - ALBERT-based fake news detector
  - Optional: Non-transformer alternatives (e.g., FastText)
- **FR-6.3:** The system shall implement text preprocessing appropriate for each model.
- **FR-6.4:** The system shall optimize model loading for minimal resource usage.
- **FR-6.5:** The system shall maintain consistent inference interfaces across models.

#### 2.3.2 Detection Processing
- **FR-7.1:** The system shall analyze input text for indicators of fake news.
- **FR-7.2:** The system shall generate a credibility score (0-100%) for analyzed content.
- **FR-7.3:** The system shall categorize results (e.g., "Likely Credible," "Possibly Fake," "Likely Fake").
- **FR-7.4:** The system shall provide confidence metrics for its assessment.
- **FR-7.5:** The system shall complete analysis within 10 seconds on target hardware.
- **FR-7.6:** The system shall allow re-analysis of the same content with different models.

#### 2.3.3 Result Management
- **FR-8.1:** The system shall store detection results with timestamps.
- **FR-8.2:** The system shall associate results with user accounts when available.
- **FR-8.3:** The system shall provide access to historical analysis results.
- **FR-8.4:** The system shall allow users to delete their analysis history.
- **FR-8.5:** The system shall implement a data retention policy for analysis results.

### 2.4 Model Comparison and Metrics

#### 2.4.1 Performance Metrics
- **FR-9.1:** The system shall track accuracy metrics for each model.
- **FR-9.2:** The system shall measure processing time for each analysis.
- **FR-9.3:** The system shall monitor memory usage during model inference.
- **FR-9.4:** The system shall track CPU utilization during analysis.
- **FR-9.5:** The system shall aggregate metrics for comparison purposes.

#### 2.4.2 Comparison Dashboard
- **FR-10.1:** The system shall provide a visual comparison of model accuracy.
- **FR-10.2:** The system shall display processing time comparisons across models.
- **FR-10.3:** The system shall visualize memory usage across models.
- **FR-10.4:** The system shall present tradeoffs between accuracy and resource usage.
- **FR-10.5:** The system shall update comparison metrics based on ongoing analyses.

### 2.5 Resource Monitoring and Optimization

#### 2.5.1 Resource Tracking
- **FR-11.1:** The system shall measure and record memory usage during model loading.
- **FR-11.2:** The system shall measure and record memory usage during inference.
- **FR-11.3:** The system shall track CPU utilization throughout the analysis process.
- **FR-11.4:** The system shall monitor response times for all API endpoints.
- **FR-11.5:** The system shall log resource usage anomalies.

#### 2.5.2 Optimization Controls
- **FR-12.1:** The system shall provide mechanisms to unload unused models.
- **FR-12.2:** The system shall implement configurable resource limits.
- **FR-12.3:** The system shall offer degraded functionality when resource limits are reached.
- **FR-12.4:** The system shall provide recommendations for optimal model selection based on available resources.

### 2.6 User Interface

#### 2.6.1 General UI Requirements
- **FR-13.1:** The system shall provide an intuitive, responsive web interface.
- **FR-13.2:** The system shall implement a clean, minimalist design for resource efficiency.
- **FR-13.3:** The system shall support both light and dark mode.
- **FR-13.4:** The system shall provide consistent navigation across all pages.
- **FR-13.5:** The system shall display appropriate loading indicators during analysis.

#### 2.6.2 Specific UI Views
- **FR-14.1:** The system shall include a login/registration page.
- **FR-14.2:** The system shall include a text/URL input page for analysis.
- **FR-14.3:** The system shall include a results display page with credibility assessment.
- **FR-14.4:** The system shall include a model comparison dashboard page.
- **FR-14.5:** The system shall include a user history page.
- **FR-14.6:** The system shall include a user preferences management page.

## 3. Non-Functional Requirements

### 3.1 Performance Requirements
- **NFR-1.1:** The system shall load the main interface within 2 seconds under normal conditions.
- **NFR-1.2:** The system shall complete text analysis within 5 seconds on target hardware.
- **NFR-1.3:** The system shall use no more than 500MB of RAM during analysis.
- **NFR-1.4:** The system shall render visualizations within 1 second.
- **NFR-1.5:** The system shall support at least a 3.0 requests per minute rate for analysis on target hardware.

### 3.2 Resource Efficiency Requirements
- **NFR-2.1:** The system shall operate on devices with at least 1GB available RAM.
- **NFR-2.2:** The system shall function on single-core processors with at least 1.2GHz clock speed.
- **NFR-2.3:** The system shall require no more than 1GB of storage for models and application.
- **NFR-2.4:** The system shall optimize network usage for low-bandwidth environments.
- **NFR-2.5:** The system shall implement lazy loading for non-critical components.

### 3.3 Security Requirements
- **NFR-3.1:** The system shall implement HTTPS for all communications.
- **NFR-3.2:** The system shall securely store user credentials and preferences.
- **NFR-3.3:** The system shall implement protection against common web vulnerabilities (OWASP Top 10).
- **NFR-3.4:** The system shall implement rate limiting for authentication attempts.
- **NFR-3.5:** The system shall implement proper input validation for all user inputs.

### 3.4 Usability Requirements
- **NFR-4.1:** The system shall be usable on desktop and mobile devices.
- **NFR-4.2:** The system shall provide clear error messages to users.
- **NFR-4.3:** The system shall include help text for interpreting results.
- **NFR-4.4:** The system shall follow web accessibility guidelines (WCAG 2.1 AA).
- **NFR-4.5:** The system shall support keyboard navigation.

### 3.5 Reliability Requirements
- **NFR-5.1:** The system shall handle errors gracefully without crashing.
- **NFR-5.2:** The system shall implement fallback detection methods if primary models fail.
- **NFR-5.3:** The system shall log all errors for troubleshooting.
- **NFR-5.4:** The system shall maintain consistent functionality across supported browsers.
- **NFR-5.5:** The system shall recover from unexpected input without requiring restart.

### 3.6 Compatibility Requirements
- **NFR-6.1:** The system shall function on current versions of major browsers (Chrome, Firefox, Safari, Edge).
- **NFR-6.2:** The system shall degrade gracefully on older browsers.
- **NFR-6.3:** The system shall be responsive on screen sizes from 320px to 1920px width.
- **NFR-6.4:** The system shall support recent versions of Python (3.9+) for backend components.

### 3.7 Legal and Ethical Requirements
- **NFR-7.1:** The system shall include disclaimers about detection accuracy.
- **NFR-7.2:** The system shall clearly communicate model limitations to users.
- **NFR-7.3:** The system shall implement appropriate data retention policies.
- **NFR-7.4:** The system shall provide terms of service and privacy policy.
- **NFR-7.5:** The system shall obtain user consent for data collection.

## 4. System Models

### 4.1 Use Cases

#### 4.1.1 User Registration and Login
1. User navigates to the application
2. User selects "Register" option
3. User provides email and password
4. System validates input and creates account
5. User logs in with credentials
6. System authenticates user and provides access

#### 4.1.2 News Text Analysis
1. User logs into the application
2. User navigates to analysis page
3. User pastes news text or enters URL
4. User selects detection model
5. System processes text and generates credibility assessment
6. User views detailed results
7. User optionally saves results

#### 4.1.3 Model Comparison
1. User navigates to comparison dashboard
2. System displays performance metrics for all models
3. User explores accuracy, speed, and resource visualizations
4. User selects preferred default model
5. System saves user preference

### 4.2 Data Dictionary

| Term | Definition | Format | Validation Rules |
|------|------------|--------|------------------|
| User | Registered system user | Object | Valid email required |
| AnalysisRequest | Text submitted for detection | Object | Text or URL required |
| DetectionResult | Outcome of analysis | Object | Must have credibility score |
| Model | Fake news detection model | Object | Must be configured properly |
| ModelMetrics | Performance statistics | Object | Must have accuracy metrics |
| ResourceUsage | Computational resource data | Object | Must include memory metrics |

## 5. Appendices

### 5.1 Supported Models (Initial)
- DistilBERT-based fake news detector
- TinyBERT-based fake news detector
- ALBERT-based fake news detector
- FastText-based classifier (optional)

### 5.2 Target Hardware Specifications
- Minimum: 1GB RAM, 1.2GHz single-core processor
- Recommended: 2GB RAM, 1.5GHz dual-core processor
- Storage: 1GB available space

### 5.3 Performance Metrics
- Accuracy: Correctly classified articles / Total articles
- Precision: True positives / (True positives + False positives)
- Recall: True positives / (True positives + False negatives)
- F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
- Processing Time: Seconds from submission to result
- Memory Usage: Peak RAM used during analysis

### 5.4 References and Datasets
- LIAR: Dataset containing 12.8K human-labeled short statements
- FakeNewsNet: Comprehensive repository with news content and social context
- ISOT Fake News Dataset: 44,898 articles with approximately 21,417 real and 23,481 fake news articles