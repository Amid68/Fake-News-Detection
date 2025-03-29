# System Design Document

**Project Title:** Lightweight Fake News Detection System  
**Document Version:** 4.0  
**Author:** Ameed Othman  
**Date:** 29.03.2025

## 1. Introduction

### 1.1 Purpose
This System Design Document (SDD) provides a technical blueprint for implementing the Lightweight Fake News Detection System. It transforms the requirements into an actionable architecture optimized for resource efficiency.

### 1.2 Scope
This document covers the system architecture, data design, interfaces, and implementation considerations for the MVP as specified in the updated Project Vision Document.

### 1.3 Definitions, Acronyms, and Abbreviations
- **API:** Application Programming Interface
- **CRUD:** Create, Read, Update, Delete
- **FND:** Fake News Detection
- **JWT:** JSON Web Token
- **LLM:** Large Language Model
- **MVC:** Model-View-Controller
- **NLP:** Natural Language Processing
- **PLM:** Pre-trained Language Model
- **REST:** Representational State Transfer

### 1.4 References
- Project Vision Document v1.0
- [Django Documentation](https://docs.djangoproject.com/)
- [React Documentation](https://reactjs.org/docs/getting-started.html)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)

## 2. System Architecture

### 2.1 Architectural Overview

The system follows a lightweight web application architecture with the following key components:

#### 2.1.1 High-Level Architecture
- **Frontend Layer:** React-based single-page application with minimal dependencies
- **Backend Layer:** Django REST API with optimized endpoints
- **Database Layer:** SQLite for resource efficiency
- **ML Layer:** Lightweight pre-trained models for fake news detection

#### 2.1.2 Architecture Diagram

```
┌─────────────────┐    ┌──────────────────────────────────────┐
│                 │    │                                      │
│   Web Browser   │◄───┤        Django Application            │
│   (React SPA)   │    │        (REST API Backend)            │
│                 │    │                                      │
└────────┬────────┘    └──────────────────┬───────────────────┘
         │                                │
         │                                │
         │                                ▼
         │              ┌──────────────────────────────────────┐
         │              │                                      │
         └─────────────►│     Fake News Detection Models       │
                        │     (Lightweight Pre-trained)        │
                        │                                      │
                        └──────────────────┬───────────────────┘
                                          │
                                          │
                                          ▼
                         ┌────────────────────────────────────┐
                         │                                    │
                         │           SQLite Database          │
                         │                                    │
                         └────────────────────────────────────┘
```

### 2.2 Component Design

#### 2.2.1 Frontend Components

**Core Components:**
- **Authentication Module:** Handles user registration, login, and session management
- **Text Input Component:** Captures news content for analysis
- **URL Fetcher Component:** Extracts text from news URLs
- **Analysis Results Component:** Displays detection results with confidence scores
- **Model Comparison Dashboard:** Shows performance metrics across models
- **Resource Usage Display:** Visualizes computational resource consumption
- **User History Component:** Displays previously analyzed content

**Technical Implementation:**
- React function components with hooks
- Minimal Redux for essential state management
- React Router for navigation
- Axios for API communication
- Recharts for lightweight data visualization
- Tailwind CSS for styling with minimal footprint

#### 2.2.2 Backend Components

**Core Components:**
- **User Management:** Authentication, authorization, and profile management
- **Text Preprocessing Service:** Cleans and prepares text for analysis
- **Model Management Service:** Loads and manages lightweight models
- **Detection Service:** Coordinates analysis across multiple models
- **Performance Metrics Service:** Measures and logs model performance
- **Resource Monitoring Service:** Tracks computational resource usage
- **API Gateway:** Provides REST endpoints for frontend communication

**Technical Implementation:**
- Django REST Framework for API development (minimal middleware)
- SQLite with Django ORM for database interactions
- JWT for authentication
- Hugging Face Transformers (optimized loading) for model integration
- Memory and CPU usage monitoring

#### 2.2.3 ML Components

**Model Pipeline:**
- **Text Preprocessing:** Tokenization, cleaning, feature extraction
- **Model Selection:** Dynamic loading of appropriate lightweight model
- **Inference Engine:** Optimized for memory efficiency
- **Result Aggregation:** When using multiple models
- **Performance Tracking:** Accuracy, speed, and resource usage metrics

**Candidate Models:**
- DistilBERT (66M parameters)
- TinyBERT (14.5M parameters)
- MobileBERT (25.3M parameters)
- ALBERT (12M parameters)
- FastText-based models (non-transformer alternatives)

### 2.3 Interaction and Communication

#### 2.3.1 Internal Communication
- **Frontend to Backend:** REST API calls (optimized payload size)
- **Backend to Models:** Direct Python function calls
- **Backend to Database:** ORM queries with optimization

#### 2.3.2 Authentication Flow
1. User submits credentials via frontend
2. Backend validates and issues JWT
3. JWT included in subsequent API requests
4. Token refresh mechanism for extended sessions

## 3. Data Design

### 3.1 Database Selection

SQLite was selected for the following reasons:
- Minimal resource footprint
- Zero-configuration database
- File-based storage suitable for lightweight applications
- Excellent support for Django ORM
- Sufficient performance for expected user load
- Easy backup and migration

### 3.2 Database Schema

#### 3.2.1 User-Related Tables

**Users Table**
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    date_joined TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP NULL
);
```

**UserPreferences Table**
```sql
CREATE TABLE user_preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    default_model VARCHAR(100) DEFAULT 'distilbert',
    show_detailed_metrics BOOLEAN DEFAULT FALSE,
    UNIQUE (user_id)
);
```

#### 3.2.2 Content-Related Tables

**AnalysisRequests Table**
```sql
CREATE TABLE analysis_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    content_text TEXT NOT NULL,
    content_url VARCHAR(512) NULL,
    content_title VARCHAR(255) NULL,
    requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address VARCHAR(45) NULL
);
```

**DetectionResults Table**
```sql
CREATE TABLE detection_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id INTEGER REFERENCES analysis_requests(id) ON DELETE CASCADE,
    model_name VARCHAR(100) NOT NULL,
    fake_probability FLOAT NOT NULL,
    real_probability FLOAT NOT NULL,
    confidence_score FLOAT NOT NULL,
    processing_time FLOAT NOT NULL,
    memory_usage FLOAT NULL,
    cpu_usage FLOAT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**ModelMetrics Table**
```sql
CREATE TABLE model_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name VARCHAR(100) NOT NULL,
    accuracy FLOAT NOT NULL,
    precision_score FLOAT NOT NULL,
    recall_score FLOAT NOT NULL,
    f1_score FLOAT NOT NULL,
    avg_processing_time FLOAT NOT NULL,
    avg_memory_usage FLOAT NOT NULL,
    parameter_count INTEGER NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (model_name)
);
```

### 3.3 Data Flow

#### 3.3.1 Analysis Request Flow
1. User submits text or URL for analysis
2. System preprocesses and cleans the text
3. Text is sent to one or more detection models
4. Models generate credibility scores
5. Results are stored in database
6. Performance metrics are updated
7. Results are returned to user interface

#### 3.3.2 Model Comparison Flow
1. User requests model comparison
2. System retrieves metrics for all models
3. Comparison data is formatted for visualization
4. Dashboard displays performance tradeoffs
5. User can select preferred model for future analysis

## 4. Interface Design

### 4.1 API Endpoints

#### 4.1.1 Authentication APIs
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `POST /api/auth/logout` - User logout
- `POST /api/auth/password/reset` - Password reset request
- `POST /api/auth/password/reset/confirm` - Password reset confirmation

#### 4.1.2 User APIs
- `GET /api/user/profile` - Get user profile
- `PUT /api/user/profile` - Update user profile
- `GET /api/user/preferences` - Get user preferences
- `PUT /api/user/preferences` - Update user preferences
- `GET /api/user/history` - Get analysis history

#### 4.1.3 Analysis APIs
- `POST /api/analyze/text` - Analyze text content
- `POST /api/analyze/url` - Analyze content from URL
- `GET /api/analyze/result/{id}` - Get specific analysis result
- `GET /api/models` - Get available models info
- `GET /api/models/comparison` - Get model comparison metrics
- `GET /api/models/resources` - Get resource usage metrics

### 4.2 User Interface Design

#### 4.2.1 Wireframes

##### Main Analysis Interface
```
┌─────────────────────────────────────────────────────────────┐
│ ┌─────┐ FakeNews Detector                    [User ▼] [🔍]   │
│ │ Logo│                                                     │
│ └─────┘                                                     │
├─────────────────────────────────────────────────────────────┤
│ [Text Analysis] [URL Analysis]                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                     │   │
│  │  Paste news text here or enter URL                  │   │
│  │  [                                               ]  │   │
│  │  [                                               ]  │   │
│  │  [                                               ]  │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Select model: [DistilBERT ▼]                              │
│                                                             │
│  [Analyze]    [Clear]                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

##### Results Display
```
┌─────────────────────────────────────────────────────────────┐
│ ┌─────┐ FakeNews Detector                    [User ▼] [🔍]   │
│ │ Logo│                                                     │
│ └─────┘                                                     │
├─────────────────────────────────────────────────────────────┤
│ < Back to Analysis                                          │
├─────────────────────────────────────────────────────────────┤
│ Analysis Results                                            │
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Credibility Assessment                                   │ │
│ │                                                         │ │
│ │ ┌───────────────────────┐                              │ │
│ │ │     78%               │  LIKELY CREDIBLE             │ │
│ │ │  █████████████████░░░ │                              │ │
│ │ └───────────────────────┘                              │ │
│ │                                                         │ │
│ │ Model used: DistilBERT                                  │ │
│ │ Processing time: 1.2 seconds                            │ │
│ │ Memory used: 450 MB                                     │ │
│ │                                                         │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ ▶ View Results from Other Models                        │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ ▶ View Processed Text                                   │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ [Save Result]  [Share]  [Analyze New Content]               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

##### Model Comparison Dashboard
```
┌─────────────────────────────────────────────────────────────┐
│ ┌─────┐ FakeNews Detector                    [User ▼] [🔍]   │
│ │ Logo│                                                     │
│ └─────┘                                                     │
├─────────────────────────────────────────────────────────────┤
│ Model Comparison Dashboard                                  │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────┐  ┌─────────────────────────┐   │
│ │                         │  │                         │   │
│ │    Accuracy             │  │    Processing Time      │   │
│ │    [Bar Chart]          │  │    [Bar Chart]          │   │
│ │                         │  │                         │   │
│ └─────────────────────────┘  └─────────────────────────┘   │
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │                                                         │ │
│ │                 Memory Usage                            │ │
│ │                 [Bar Chart]                             │ │
│ │                                                         │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │                                                         │ │
│ │                 Performance vs Resources                │ │
│ │                 [Scatter Plot]                          │ │
│ │                                                         │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ Set as default: [DistilBERT ▼]  [Save Preference]           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 5. Technology Selection

### 5.1 Development Stack

#### 5.1.1 Frontend Stack
- **Core Framework:** React 18 (minimal configuration)
- **State Management:** React Context API (avoiding Redux where possible)
- **Routing:** React Router 6
- **UI Framework:** Tailwind CSS (purged for minimal footprint)
- **Visualization:** Recharts (lightweight charts)
- **HTTP Client:** Axios
- **Testing:** Jest, React Testing Library

#### 5.1.2 Backend Stack
- **Core Framework:** Django 4.2 with Django REST Framework
- **Authentication:** SimpleJWT (lightweight token auth)
- **ORM:** Django ORM with query optimization
- **Testing:** Pytest, Django Test Client

#### 5.1.3 Database
- **RDBMS:** SQLite 3
- **Migration Tool:** Django Migrations

### 5.2 ML Stack

#### 5.2.1 Core ML Libraries
- **Transformers:** Hugging Face Transformers (optimized loading)
- **Optimization:** ONNX Runtime for inference acceleration
- **Text Processing:** NLTK (core components only)
- **Metrics:** scikit-learn (for evaluation metrics)

#### 5.2.2 Model Optimization Techniques
- **Model Pruning:** Removing unnecessary weights
- **Knowledge Distillation:** Using smaller student models
- **Quantization:** Reducing precision of model weights
- **Lazy Loading:** Loading models only when needed
- **Caching:** Reusing model instances across requests

### 5.3 Monitoring and Logging
- **Application Logging:** Python logging with rotating file handler
- **Resource Monitoring:** psutil for memory and CPU tracking
- **Performance Tracking:** Custom middleware for request timing

## 6. Implementation Strategy

### 6.1 Development Approach
- Iterative development with 2-week cycles
- Feature branches with pull requests
- Focus on resource efficiency in all components
- Regular performance benchmarking

### 6.2 Implementation Order
1. Setup development environment with resource monitoring
2. Implement database schema and core models
3. Integrate lightweight FND models
4. Develop text preprocessing pipeline
5. Create model comparison framework
6. Build analysis API endpoints
7. Implement authentication system
8. Develop frontend components
9. Create visualization dashboards
10. Add resource usage monitoring
11. Test, optimize, and refine

### 6.3 Deployment Strategy

#### 6.3.1 Development Environment
- Local development with resource constraints simulation
- SQLite database
- Local model testing with sample datasets

#### 6.3.2 Production Environment
- Lightweight VPS or edge computing device
- Containerized deployment with Alpine-based Docker images
- SQLite database with regular backups
- HTTPS with Let's Encrypt
- Static files served directly from nginx

## 7. Testing Strategy

### 7.1 Testing Levels
- **Unit Testing:** Individual functions and classes
- **Integration Testing:** Component interactions
- **Model Testing:** FND model accuracy and resource usage
- **UI Testing:** Frontend component testing
- **Resource Testing:** Memory and CPU usage monitoring

### 7.2 Testing Tools
- **Backend:** Pytest, Django Test Client
- **Frontend:** Jest, React Testing Library
- **Models:** Standard NLP metrics (accuracy, F1, etc.)
- **Resource Usage:** psutil for tracking memory and CPU

## 8. Security Considerations

### 8.1 Authentication and Authorization
- JWT-based authentication with short expiry
- Secure password storage with bcrypt
- Role-based access control
- Rate limiting for sensitive endpoints

### 8.2 Data Protection
- HTTPS for all communications
- Minimal data collection and storage
- Input validation and sanitization
- Protection against common web vulnerabilities

### 8.3 API Security
- Rate limiting to prevent abuse
- CORS configuration
- Input validation
- Authentication for all non-public endpoints

## 9. Performance Considerations

### 9.1 Resource Optimization Techniques
- **Model Loading:** Lazy loading and model sharing
- **Memory Management:** Careful garbage collection
- **Database:** Optimized queries and proper indexing
- **Caching:** Response caching for common requests
- **Frontend:** Code splitting and lazy component loading

### 9.2 Performance Targets
- **Model Inference:** <5 seconds per article
- **Memory Usage:** <500MB per instance
- **Page Load Time:** <2 seconds initial load
- **API Response Time:** <1 second for non-ML endpoints

## 10. Maintenance and Support

### 10.1 Monitoring
- Resource usage tracking
- Model performance metrics
- Error logging and alerting
- Regular performance benchmarking

### 10.2 Update Process
- Regular model evaluation with new datasets
- Security patches for dependencies
- Documentation updates for deployment optimizations

## 11. Risk Mitigation

### 11.1 Technical Risks and Mitigations
- **Accuracy Degradation:** Regular benchmarking against latest FND datasets
- **Resource Spikes:** Implement resource limits and graceful degradation
- **Model Obsolescence:** Framework for easy model updates
- **Data Drift:** Monitoring detection accuracy over time
- **False Positives:** User feedback mechanism for improvement

### 11.2 Operational Risks and Mitigations
- **Resource Constraints:** Implement adaptive model selection based on available resources
- **High Request Volume:** Queuing system for peak periods
- **Model Loading Failures:** Fallback to simpler models
- **Data Storage Growth:** Implement retention policies for analysis history