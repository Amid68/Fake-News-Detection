# System Design Document

**Project Title:** Automated News Aggregation, Summarization & Topic Analysis Tool  
**Document Version:** 3.0  
**Author:** Ameed Othman
**Date:** 10.03.2025

## 1. Introduction

### 1.1 Purpose
This System Design Document (SDD) provides a technical blueprint for implementing the News Aggregation, Summarization & Topic Analysis Tool. It transforms the requirements into an actionable architecture and design.

### 1.2 Scope
This document covers the system architecture, data design, interfaces, and implementation considerations for the MVP as specified in the Software Requirements Specification (SRS).

### 1.3 Definitions, Acronyms, and Abbreviations
- **API:** Application Programming Interface
- **CRUD:** Create, Read, Update, Delete
- **JWT:** JSON Web Token
- **LLM:** Large Language Model
- **MVC:** Model-View-Controller
- **NLP:** Natural Language Processing
- **ORM:** Object-Relational Mapping
- **REST:** Representational State Transfer

### 1.4 References
- Project Vision Document v3.0
- Software Requirements Specification v3.0
- Evaluation and Testing Plan
- [Django Documentation](https://docs.djangoproject.com/)
- [React Documentation](https://reactjs.org/docs/getting-started.html)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)

## 2. System Architecture

### 2.1 Architectural Overview

The system follows a modern web application architecture with the following key components:

#### 2.1.1 High-Level Architecture
- **Frontend Layer:** React-based single-page application with visualization components
- **Backend Layer:** Django REST API
- **Database Layer:** PostgreSQL relational database
- **Processing Layer:** Asynchronous task processing with Celery and Redis
- **External Services:** News APIs, LLM integration

#### 2.1.2 Architecture Diagram

```
┌─────────────────┐    ┌──────────────────────────────────────┐    ┌────────────────┐
│                 │    │                                      │    │                │
│   Web Browser   │◄───┤             Nginx Server             │◄───┤  News APIs     │
│   (React SPA)   │    │             (Load Balancer)          │    │                │
│                 │    │                                      │    └────────────────┘
└────────┬────────┘    └──────────────────┬───────────────────┘
         │                                │                         ┌────────────────┐
         │                                │                         │                │
         │                                │                         │  Hugging Face  │
         │                                ▼                         │  Models        │
         │              ┌──────────────────────────────────────┐   │                │
         │              │                                      │   └────────┬───────┘
         └─────────────►│         Django Application           │◄───────────┘
                        │         (REST API Backend)           │
                        │                                      │
                        └──────────────────┬───────────────────┘
                                          │
                                          │
         ┌─────────────────┐              │              ┌────────────────┐
         │                 │              │              │                │
         │  Redis Cache    │◄─────────────┴─────────────►│  PostgreSQL DB │
         │  & Task Queue   │                             │                │
         │                 │              ┌──────────────┴───────────────┐
         └─────────┬───────┘              │                              │
                   │                      │       Celery Workers         │
                   └─────────────────────►│ (Summarization, Topic Analysis) │
                                         │                              │
                                         └──────────────────────────────┘
```

### 2.2 Component Design

#### 2.2.1 Frontend Components

**Core Components:**
- **Authentication Module:** Handles user registration, login, and session management
- **News Feed Component:** Displays personalized article list with summaries and topic indicators
- **Article Detail Component:** Shows full article information with enhanced metadata
- **Topic Analysis Component:** Displays topic classification and entity extraction results
- **Visualization Dashboard:** Interactive data visualizations for content analysis
- **User Preferences Component:** Manages topic selections and other user settings

**Technical Implementation:**
- React function components with hooks
- Redux for state management
- React Router for navigation
- Axios for API communication
- Recharts for data visualization
- Tailwind CSS for styling

#### 2.2.2 Backend Components

**Core Components:**
- **User Management:** Authentication, authorization, and profile management
- **News Aggregation Service:** Fetches and processes articles from sources
- **Summarization Service:** Generates article summaries using LLMs
- **Topic Analysis Service:** Analyzes and classifies article content
- **Feed Management:** Generates personalized feeds based on preferences
- **Visualization Data Service:** Aggregates and processes data for visualizations
- **API Gateway:** Provides REST endpoints for frontend communication

**Technical Implementation:**
- Django REST Framework for API development
- Django ORM for database interactions
- Celery for asynchronous task processing
- JWT for authentication
- Hugging Face Transformers for LLM integration
- spaCy for natural language processing

#### 2.2.3 Database Components

**Core Entities:**
- Users and authentication
- User preferences
- News sources
- Articles
- Summaries
- Topic analysis results
- Entity recognition results
- System logs

### 2.3 Interaction and Communication

#### 2.3.1 Internal Communication
- **Frontend to Backend:** REST API calls
- **Backend to Database:** ORM queries
- **Backend to Processing:** Task queue messages

#### 2.3.2 External Communication
- **News Source Integration:** HTTP requests to APIs
- **LLM Integration:** Hugging Face Transformers API

#### 2.3.3 Authentication Flow
1. User submits credentials via frontend
2. Backend validates and issues JWT
3. JWT included in subsequent API requests
4. Token refresh mechanism for extended sessions

## 3. Data Design

### 3.1 Database Selection

PostgreSQL was selected for the following reasons:
- Full SQL compliance and advanced features
- Excellent support for JSON/JSONB fields (useful for topic data)
- Strong support for Django ORM
- Free and open-source
- Robust backup and recovery options
- Scalability for future growth

### 3.2 Database Schema

#### 3.2.1 User-Related Tables

**Users Table**
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    date_joined TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE NULL
);
```

**UserPreferences Table**
```sql
CREATE TABLE user_preferences (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    topic VARCHAR(100) NOT NULL,
    UNIQUE (user_id, topic)
);
```

**UserActivity Table**
```sql
CREATE TABLE user_activity (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    action_type VARCHAR(50) NOT NULL,
    action_details JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ip_address VARCHAR(45) NULL
);
```

#### 3.2.2 Content-Related Tables

**NewsSources Table**
```sql
CREATE TABLE news_sources (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    url VARCHAR(255) NOT NULL,
    api_details JSONB NULL,
    reliability_score FLOAT NULL,
    is_active BOOLEAN DEFAULT TRUE
);
```

**Articles Table**
```sql
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES news_sources(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    url VARCHAR(512) UNIQUE NOT NULL,
    author VARCHAR(255) NULL,
    published_at TIMESTAMP WITH TIME ZONE NOT NULL,
    fetched_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    image_url VARCHAR(512) NULL
);
```

**ArticleTopics Table**
```sql
CREATE TABLE article_topics (
    id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
    topic VARCHAR(100) NOT NULL,
    confidence FLOAT NULL,
    UNIQUE (article_id, topic)
);
```

**Summaries Table**
```sql
CREATE TABLE summaries (
    id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
    headline_summary TEXT NULL,
    standard_summary TEXT NOT NULL,
    detailed_summary TEXT NULL,
    model_used VARCHAR(100) NOT NULL,
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    quality_metrics JSONB NULL,
    UNIQUE (article_id)
);
```

**TopicAnalysisResults Table**
```sql
CREATE TABLE topic_analysis_results (
    id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
    primary_topic VARCHAR(100) NOT NULL,
    confidence FLOAT NOT NULL,
    secondary_topics JSONB NULL,
    keywords JSONB NULL,
    entities JSONB NULL,
    sentiment_score FLOAT NULL,
    model_used VARCHAR(100) NOT NULL,
    processing_time FLOAT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE (article_id)
);
```

**UserSavedArticles Table**
```sql
CREATE TABLE user_saved_articles (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
    saved_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE (user_id, article_id)
);
```

### 3.3 Data Flow

#### 3.3.1 News Aggregation Flow
1. Scheduled task triggers article collection
2. System fetches articles from configured sources
3. Articles processed, deduplicated, and stored
4. Tasks queued for summarization and topic analysis
5. Processed articles become available in feeds

#### 3.3.2 Topic Analysis Flow
1. New articles are queued for topic analysis
2. Topic analysis processor extracts primary and secondary topics
3. Entity recognition identifies people, organizations, and locations
4. Keywords are extracted and ranked by relevance
5. Simple sentiment analysis assigns a sentiment score
6. Results stored in TopicAnalysisResults table

#### 3.3.3 Visualization Data Flow
1. Frontend requests visualization data
2. Backend aggregates article metadata
3. Topic distribution calculated from analysis results
4. Time-based trends extracted from publication dates
5. Data formatted for visualization components
6. Cached responses used for improved performance

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
- `GET /api/user/saved-articles` - Get user saved articles
- `POST /api/user/saved-articles` - Save article
- `DELETE /api/user/saved-articles/{id}` - Remove saved article

#### 4.1.3 Content APIs
- `GET /api/articles` - Get articles with filtering
- `GET /api/articles/{id}` - Get article details
- `GET /api/articles/{id}/insights` - Get article topic insights
- `GET /api/topics` - Get available topics
- `GET /api/sources` - Get available sources

#### 4.1.4 Visualization APIs
- `GET /api/dashboard/stats` - Get dashboard statistics
- `GET /api/dashboard/topic-trends` - Get topic trend data
- `GET /api/dashboard/source-distribution` - Get source distribution data
- `GET /api/dashboard/sentiment-analysis` - Get sentiment analysis data

### 4.2 User Interface Design

#### 4.2.1 Wireframes

##### Analytics Dashboard
```
┌─────────────────────────────────────────────────────────────┐
│ ┌─────┐ News Analyzer                       [User ▼] [🔍]   │
│ │ Logo│                                                     │
│ └─────┘                                                     │
├─────────────────────────────────────────────────────────────┤
│ [Week ▼] [Month ▼] [All Time ▼] [Refresh]                   │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────┐  ┌─────────────────────────┐   │
│ │                         │  │                         │   │
│ │    Topic Distribution   │  │    Sentiment Analysis   │   │
│ │    [Bar Chart]          │  │    [Pie Chart]          │   │
│ │                         │  │                         │   │
│ └─────────────────────────┘  └─────────────────────────┘   │
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │                                                         │ │
│ │                 Topic Trends Over Time                  │ │
│ │                 [Line Chart]                            │ │
│ │                                                         │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ ┌─────────────────────────┐  ┌─────────────────────────┐   │
│ │                         │  │                         │   │
│ │    Source Distribution  │  │    Keyword Cloud        │   │
│ │    [Bar Chart]          │  │    [Tag Cloud]          │   │
│ │                         │  │                         │   │
│ └─────────────────────────┘  └─────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

##### Article Detail with Topic Analysis
```
┌─────────────────────────────────────────────────────────────┐
│ ┌─────┐ News Analyzer                       [User ▼] [🔍]   │
│ │ Logo│                                                     │
│ └─────┘                                                     │
├─────────────────────────────────────────────────────────────┤
│ < Back to Feed                                              │
├─────────────────────────────────────────────────────────────┤
│ Article Title                                               │
│ Source: [Source Name] | Published: [Date] | Author: [Name]  │
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ [Summary] [Topic Analysis] [Original Article]           │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │                                                         │ │
│ │ Primary Topic: Technology (85% confidence)              │ │
│ │                                                         │ │
│ │ Secondary Topics:                                       │ │
│ │ • Business (45%)                                        │ │
│ │ • Science (30%)                                         │ │
│ │                                                         │ │
│ │ Key Terms:                                              │ │
│ │ [artificial intelligence] [machine learning] [data]     │ │
│ │                                                         │ │
│ │ Named Entities:                                         │ │
│ │ • Organizations: Google, OpenAI, Microsoft              │ │
│ │ • People: Sam Altman, Sundar Pichai                     │ │
│ │ • Locations: San Francisco, Seattle                     │ │
│ │                                                         │ │
│ │ Sentiment: Slightly Positive (0.32)                     │ │
│ │                                                         │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ [Read Original Article]  [Save]  [Share]                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 5. Technology Selection

### 5.1 Development Stack

#### 5.1.1 Frontend Stack
- **Core Framework:** React 18
- **State Management:** Redux Toolkit
- **Routing:** React Router 6
- **UI Framework:** Tailwind CSS
- **Visualization:** Recharts, D3.js
- **HTTP Client:** Axios
- **Testing:** Jest, React Testing Library

#### 5.1.2 Backend Stack
- **Core Framework:** Django 4.2 with Django REST Framework
- **Authentication:** Django REST Knox or SimpleJWT
- **Task Queue:** Celery with Redis broker
- **ORM:** Django ORM
- **Testing:** Pytest, Django Test Client

#### 5.1.3 Database
- **RDBMS:** PostgreSQL 15
- **Migration Tool:** Django Migrations

### 5.2 NLP Stack

#### 5.2.1 Core NLP Libraries
- **spaCy:** General NLP tasks, entity recognition
- **Hugging Face Transformers:** For LLM-based tasks
- **NLTK:** Text processing utilities
- **TextBlob:** Simple sentiment analysis
- **scikit-learn:** Optional for machine learning components

#### 5.2.2 NLP Models
- **Summarization:** BART-large-cnn or T5
- **Topic Classification:** Zero-shot classification with transformer models
- **Entity Recognition:** spaCy English model (en_core_web_md)
- **Sentiment Analysis:** TextBlob or fine-tuned classifier

### 5.3 External Services and Libraries

#### 5.3.1 News Sources
- Primary: NewsAPI (https://newsapi.org/)
- Alternatives:
  - The Guardian API
  - New York Times API
  - Custom web scrapers (with proper attribution)

#### 5.3.2 Monitoring and Logging
- **Application Logging:** Python logging with rotating file handler
- **Error Tracking:** Sentry
- **Metrics:** Prometheus (optional)
- **Visualization:** Grafana (optional)

## 6. Implementation Strategy

### 6.1 Development Approach
- Iterative development with 2-week cycles
- Feature branches with pull requests
- CI/CD pipeline with automated testing
- Regular code review with supervisor

### 6.2 Implementation Order
1. Setup development environment and project structure
2. Implement database schema and core models
3. Develop authentication system
4. Build news aggregation service
5. Implement LLM integration for summarization
6. Develop topic analysis system
7. Create visualization data processing
8. Build frontend dashboard components
9. Implement enhanced article detail view
10. Connect frontend to backend APIs
11. Test, optimize, and refine

### 6.3 Deployment Strategy

#### 6.3.1 Development Environment
- Local Docker Compose setup
- SQLite for development database
- Local LLM models for offline development

#### 6.3.2 Production Environment
- AWS Elastic Beanstalk or EC2
- Containerized deployment with Docker
- PostgreSQL RDS database
- Redis for caching and task queue
- HTTPS with Let's Encrypt
- Static files on S3 or CloudFront

## 7. Testing Strategy

### 7.1 Testing Levels
- **Unit Testing:** Individual components and functions
- **Integration Testing:** Component interactions
- **API Testing:** Endpoint functionality
- **UI Testing:** Frontend component testing
- **Performance Testing:** Visualization and data processing performance

### 7.2 Testing Tools
- **Backend:** Pytest, Django Test Client
- **Frontend:** Jest, React Testing Library
- **API:** Postman, Newman
- **Performance:** Locust (basic load testing)

## 8. Security Considerations

### 8.1 Authentication and Authorization
- JWT-based authentication
- Password storage with bcrypt
- Role-based access control
- Rate limiting for sensitive endpoints

### 8.2 Data Protection
- HTTPS for all communications
- Database encryption at rest
- Input validation and sanitization
- Protection against common web vulnerabilities

### 8.3 API Security
- Rate limiting
- CORS configuration
- Input validation
- Authentication for all non-public endpoints

## 9. Performance Considerations

### 9.1 Optimization Techniques
- Database query optimization
- Efficient ORM usage
- Redis caching for visualization data
- Asynchronous processing for resource-intensive tasks
- Frontend optimizations (code splitting, lazy loading)

### 9.2 Visualization Performance
- Data aggregation performed on backend
- Caching of visualization datasets
- Progressive loading for large datasets
- Time-based data partitioning
- Client-side data transformations minimized

## 10. Maintenance and Support

### 10.1 Monitoring
- Application logs
- Error tracking
- Performance metrics
- Regular security scanning

### 10.2 Backup Strategy
- Daily database backups
- Backup retention policy
- Periodic recovery testing
- Code repository backups

### 10.3 Update Process
- Regular dependency updates
- Security patch procedure
- Feature enhancement process
- Documentation updates

## 11. Risk Mitigation

### 11.1 Technical Risks and Mitigations
- **LLM Performance:** Fall back to simpler models if needed
- **API Rate Limits:** Implement caching and retries
- **Visualization Performance:** Implement data pagination and aggregation
- **Resource Constraints:** Optimize and prioritize features
- **Data Quality:** Implement robust validation and error handling

### 11.2 Operational Risks and Mitigations
- **Service Availability:** Implement monitoring and redundancy
- **Data Loss:** Regular backups and recovery testing
- **Security Breaches:** Regular security reviews and updates
- **Performance Issues:** Monitoring and optimization plan