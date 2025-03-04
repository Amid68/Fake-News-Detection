# System Design Document

**Project Title:** Automated Multilingual News Aggregation, Summarization & Bias Detection Tool  
**Document Version:** 2.0 (Revised)  
**Author:** Ameed Othman  
**Supervisor Review:** Completed  
**Date:** 04.03.2025

## 1. Introduction

### 1.1 Purpose
This System Design Document (SDD) provides a comprehensive technical blueprint for implementing the Automated News Aggregation, Summarization & Bias Detection Tool. It transforms the requirements into an actionable architecture and technical design.

### 1.2 Scope
This document covers the complete system architecture, data design, interfaces, and implementation considerations for the MVP as specified in the Software Requirements Specification (SRS).

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
- Project Vision Document v2.0
- Software Requirements Specification v2.0
- Evaluation and Testing Plan
- [Django Documentation](https://docs.djangoproject.com/)
- [React Documentation](https://reactjs.org/docs/getting-started.html)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)

## 2. System Architecture

### 2.1 Architectural Overview

The system follows a modern web application architecture with the following key components:

#### 2.1.1 High-Level Architecture
- **Frontend Layer:** React-based single-page application
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
                   └─────────────────────►│  (Summarization, Bias)       │
                                         │                              │
                                         └──────────────────────────────┘
```

### 2.2 Component Design

#### 2.2.1 Frontend Components

**Core Components:**
- **Authentication Module:** Handles user registration, login, and session management
- **News Feed Component:** Displays personalized article list with summaries and bias indicators
- **Article Detail Component:** Shows full article information with enhanced metadata
- **User Preferences Component:** Manages topic selections and other user settings
- **Search Component:** Provides content discovery functionality

**Technical Implementation:**
- React function components with hooks
- Redux for state management
- React Router for navigation
- Axios for API communication
- Tailwind CSS for styling

#### 2.2.2 Backend Components

**Core Components:**
- **User Management:** Authentication, authorization, and profile management
- **News Aggregation Service:** Fetches and processes articles from sources
- **Summarization Service:** Generates article summaries using LLMs
- **Bias Detection Service:** Analyzes and classifies article bias
- **Feed Management:** Generates personalized feeds based on preferences
- **API Gateway:** Provides REST endpoints for frontend communication

**Technical Implementation:**
- Django REST Framework for API development
- Django ORM for database interactions
- Celery for asynchronous task processing
- JWT for authentication
- Hugging Face Transformers for LLM integration

#### 2.2.3 Database Components

**Core Entities:**
- Users and authentication
- User preferences
- News sources
- Articles
- Summaries
- Bias assessments
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
- Excellent performance for concurrent operations
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
    summary_text TEXT NOT NULL,
    model_used VARCHAR(100) NOT NULL,
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE (article_id)
);
```

**BiasAssessments Table**
```sql
CREATE TABLE bias_assessments (
    id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
    bias_score FLOAT NOT NULL, -- Range from -1.0 (left) to 1.0 (right)
    confidence FLOAT NOT NULL, -- Range from 0.0 to 1.0
    model_used VARCHAR(100) NOT NULL,
    assessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
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
4. Tasks queued for summarization and bias detection
5. Processed articles become available in feeds

#### 3.3.2 User Interaction Flow
1. User authenticates and loads personalized feed
2. System queries articles matching user preferences
3. Article data enriched with summaries and bias scores
4. Paginated results returned to frontend
5. User interactions tracked for potential future enhancements

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
- `GET /api/topics` - Get available topics
- `GET /api/sources` - Get available sources

### 4.2 User Interface Design

#### 4.2.1 Wireframes

##### Home/Feed Page
```
┌─────────────────────────────────────────────────────────────┐
│ ┌─────┐ News Analyzer                       [User ▼] [🔍]   │
│ │ Logo│                                                     │
│ └─────┘                                                     │
├─────────────────────────────────────────────────────────────┤
│ [Politics▼] [Business▼] [Technology▼] [More▼]  [Sort by▼]   │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Article Title 1                                   [src] │ │
│ │ [Summary of the article goes here...]                   │ │
│ │                                                         │ │
│ │ [Bias Indicator]            [Read More] [Save] [Share]  │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Article Title 2                                   [src] │ │
│ │ [Summary of the article goes here...]                   │ │
│ │                                                         │ │
│ │ [Bias Indicator]            [Read More] [Save] [Share]  │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Article Title 3                                   [src] │ │
│ │ [Summary of the article goes here...]                   │ │
│ │                                                         │ │
│ │ [Bias Indicator]            [Read More] [Save] [Share]  │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ [Load More]                                                 │
└─────────────────────────────────────────────────────────────┘
```

##### Article Detail Page
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
│ │                                                         │ │
│ │ [Summary Box]                                           │ │
│ │                                                         │ │
│ │ AI-Generated Summary:                                   │ │
│ │ [Detailed summary text...]                              │ │
│ │                                                         │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │                                                         │ │
│ │ [Bias Analysis Box]                                     │ │
│ │                                                         │ │
│ │ Political Leaning Assessment:                           │ │
│ │ [Visual spectrum indicator]                             │ │
│ │                                                         │ │
│ │ Confidence: [Score]                                     │ │
│ │                                                         │ │
│ │ [Was this assessment helpful? [👍] [👎]]                │ │
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

### 5.2 External Services and Libraries

#### 5.2.1 News Sources
- Primary: NewsAPI (https://newsapi.org/)
- Alternatives:
  - The Guardian API
  - New York Times API
  - Custom web scrapers (with proper attribution)

#### 5.2.2 NLP Components
- **Summarization:** Hugging Face Transformers
  - Models considered: BART, T5, PEGASUS
  - Initial selection: BART-large-cnn (fine-tuned for news summarization)
  
- **Bias Detection:**
  - Base: BERT-base or RoBERTa-base
  - Fine-tuning required on labeled dataset
  - Fallback: Rule-based approach for MVP

#### 5.2.3 Monitoring and Logging
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
6. Develop bias detection system
7. Create personalized feed system
8. Build frontend components
9. Connect frontend to backend APIs
10. Test, optimize, and refine

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
- **Performance Testing:** Load and response time testing

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
- Redis caching for frequent queries
- Asynchronous processing for resource-intensive tasks
- Frontend optimizations (code splitting, lazy loading)

### 9.2 Scalability Approach
- Horizontal scaling for web tier
- Vertical scaling for database tier initially
- Efficient caching strategy
- Optimized database indexes
- Background workers for processing tasks

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
- **Resource Constraints:** Optimize and prioritize features
- **Data Quality:** Implement robust validation and error handling

### 11.2 Operational Risks and Mitigations
- **Service Availability:** Implement monitoring and redundancy
- **Data Loss:** Regular backups and recovery testing
- **Security Breaches:** Regular security reviews and updates
- **Performance Issues:** Monitoring and optimization plan

## 12. Glossary
- **API:** Application Programming Interface
- **JWT:** JSON Web Token
- **LLM:** Large Language Model
- **MVP:** Minimum Viable Product
- **NLP:** Natural Language Processing
- **ORM:** Object-Relational Mapping
- **SPA:** Single Page Application