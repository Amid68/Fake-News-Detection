# System Design Document

**Project Title:** Automated Multilingual News Aggregation, Summarization & Bias Detection Tool  
**Document Version:** 1.0 (Initial Draft)  
**Author**: Ameed Othman    
**Date:** 03.03.2025

---

## 1. Introduction

### 1.1 Purpose
This System Design Document (SDD) provides a comprehensive technical blueprint for the Stage 1 web application of the *Automated Multilingual News Aggregation, Summarization & Bias Detection Tool*. It details the system architecture, components, data flow, and technology stack based on the Software Requirements Specification (SRS), ensuring a robust, scalable, and maintainable solution for development and deployment.

### 1.2 Scope
This document focuses on the Stage 1 web application, covering:
- User authentication and profile management.
- News aggregation from reliable English sources.
- Automated summarization and bias detection using open-source LLMs.
- Personalized news feed generation.
- Responsive user interface design.
Future stages (e.g., multilingual support, browser extension) are considered for extensibility but not detailed here.

### 1.3 Intended Audience
This SDD is intended for the project developer, potential collaborators, and stakeholders requiring a technical understanding of the system’s design and implementation.

### 1.4 References
- Project Vision Document: [vision.md](vision.md)
- Software Requirements Specification (SRS): [srs.md](srs.md)

---

## 2. Architectural Design

### 2.1 High-Level Architecture Diagram
The system employs a **three-tier architecture**:
- **Presentation Layer:** Web browser with HTML, CSS, JavaScript (optional React/Vue.js).
- **Application Layer:** Django backend managing logic and integrations.
- **Data Layer:** SQLite (upgradable to PostgreSQL).

**External Integrations:**
- News APIs (e.g., NewsAPI) or scraping for article fetching.
- Open-source LLMs via Hugging Face Transformers.

**Diagram:**
```
+-------------------+       +-----------------------+       +-------------------+       +--------------------+
| Web Browser       | <---> | Django Web Application | <---> | Database (SQLite)  |       | Open-Source LLM   |
| (Client - Front-End)|       | (Server - Back-End)   |       |                   |       | (e.g., Hugging Face)|
+-------------------+       +-----------------------+       +-------------------+       +--------------------+
        ^                             ^                          ^
        |                             |                          |
User Interaction        Data/Request Flow        Data Storage/Retrieval     NLP Tasks (Summarization, Bias Detection)

                         ^
                         |
                +-------------------+
                | News APIs/Websites|
                | (External Sources)|
                +-------------------+
                News Article Fetching
```

### 2.2 Component Descriptions
- **Web Application Server (Django Backend):**
  - **Responsibilities:** Manage user requests, business logic, API endpoints, and orchestrate news processing.
  - **Interfaces:** REST API for front-end, Django ORM for database, API/library calls for LLM and news services.

- **Database (SQLite):**
  - **Responsibilities:** Store user data, articles, sources, and logs.
  - **Interfaces:** Django ORM.

- **News Aggregation Module:**
  - **Responsibilities:** Fetch and parse articles from sources like BBC, Reuters, AP.
  - **Interfaces:** Backend requests, outputs structured article data.

- **Summarization Module:**
  - **Responsibilities:** Generate concise summaries (~50 words) using LLMs.
  - **Interfaces:** Local LLM integration via Hugging Face.

- **Bias Detection Module:**
  - **Responsibilities:** Assign bias scores (e.g., neutral, left, right) using LLM classification.
  - **Interfaces:** Local LLM integration.

- **Personalized Feed Module:**
  - **Responsibilities:** Filter articles by user preferences.
  - **Interfaces:** Database queries, front-end API.

- **User Interface (Front-End):**
  - **Responsibilities:** Display login, news feed, article details, and settings.
  - **Interfaces:** REST API calls.

- **Administrative & Monitoring Tools:**
  - **Responsibilities:** Track system performance and user activity via analytics and logs.
  - **Interfaces:** Backend integration, dashboard display.

---

## 3. Data Design

### 3.1 Database Schema
- **Users Table:**
  - `user_id` (INTEGER, PRIMARY KEY, AUTOINCREMENT)
  - `email` (VARCHAR(255), UNIQUE, NOT NULL)
  - `password_hash` (VARCHAR(255), NOT NULL)
  - `registration_date` (DATETIME)

- **UserPreferences Table:**
  - `preference_id` (INTEGER, PRIMARY KEY, AUTOINCREMENT)
  - `user_id` (INTEGER, FOREIGN KEY → Users.user_id)
  - `topic_keyword` (VARCHAR(255), NOT NULL)
  - UNIQUE (user_id, topic_keyword)

- **Articles Table:**
  - `article_id` (INTEGER, PRIMARY KEY, AUTOINCREMENT)
  - `title` (TEXT, NOT NULL)
  - `content` (TEXT)
  - `source` (VARCHAR(255))
  - `summary` (TEXT)
  - `bias_score` (FLOAT)
  - `publication_date` (DATETIME)
  - `source_article_url` (TEXT, UNIQUE)

- **Sources Table:**
  - `source_id` (INTEGER, PRIMARY KEY, AUTOINCREMENT)
  - `name` (VARCHAR(255), UNIQUE, NOT NULL)
  - `base_url` (TEXT, NOT NULL)
  - `api_endpoint` (TEXT, optional)

- **ActivityLogs Table:**
  - `log_id` (INTEGER, PRIMARY KEY, AUTOINCREMENT)
  - `user_id` (INTEGER, FOREIGN KEY → Users.user_id)
  - `action` (VARCHAR(255))
  - `timestamp` (DATETIME)

### 3.2 Data Flow
1. **User Interaction:** Registration/login → Preferences set → Feed accessed.
2. **News Aggregation:** Periodic fetches → Articles stored.
3. **Processing:** Articles summarized and analyzed for bias → Results saved.
4. **Feed Generation:** Articles filtered by preferences → Displayed with summaries and bias scores.

---

## 4. Technology Stack

- **Backend:** Python, Django
- **Frontend:** HTML, CSS, JavaScript (optional React/Vue.js)
- **Database:** SQLite (Stage 1), upgradable to PostgreSQL
- **LLM Models:**
  - **Summarization:** BART (chosen for strong news summarization performance, open-source availability).
  - **Bias Detection:** Fine-tuned BERT (selected for classification, requires validation).
- **News Aggregation:** NewsAPI, BeautifulSoup/Scrapy for scraping.
- **Task Scheduling:** Celery with Redis.

---

## 5. API Design

- **POST /api/register:** Register user (email, password) → JSON {user_id, status}.
- **POST /api/login:** Authenticate user → JSON {token, status}.
- **GET /api/newsfeed:** Fetch personalized feed (auth token) → JSON {articles}.
- **GET /api/article/{article_id}:** Article details → JSON {title, summary, bias_score}.
- **PUT /api/user/preferences:** Update preferences (auth token, topics) → JSON {status}.

**Format:** JSON  
**Auth:** Session-based (Django).

---

## 6. Security and Performance

### 6.1 Security Measures
- **Authentication:** Django session-based with bcrypt hashing.
- **Encryption:** HTTPS, sensitive data encrypted at rest.
- **Validation:** ORM prevents SQL injection, sanitization prevents XSS.
- **LLM Security:** Local models with updated libraries, API keys secured if used.

### 6.2 Performance Optimization
- **Caching:** Redis for frequent queries (e.g., feed generation).
- **Asynchronous Processing:** Celery for aggregation and NLP tasks.
- **Database:** Indexed queries for fast retrieval.
- **Scalability:** Load balancing planned for future growth.

---

## 7. Deployment Architecture

- **Development:** Local Django server.
- **Production:** Heroku/AWS with Gunicorn, Nginx reverse proxy, Celery/Redis task queue.
- **Steps:**
  1. Set up server (Linux).
  2. Install dependencies.
  3. Configure SQLite database.
  4. Deploy Django app.
  5. Set up HTTPS via Nginx.
- **Monitoring:** Logs and analytics via Administrative Tools.
- **Backup:** Regular database snapshots.

---

## 8. Future Considerations
- **Modularity:** API and components reusable for mobile/browser extensions.
- **Multilingual:** Language-specific modules pluggable.
- **Scalability:** Load balancing and database upgrades supported.
