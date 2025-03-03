# Software Requirements Specification Document

**Project Title:** Automated Multilingual News Aggregation, Summarization & Bias Detection Tool  
**Document Version:** 1.0 (Initial Draft)  
**Author**: Ameed Othman    
**Date:** 03.03.2025

---

## 1. Introduction

### 1.1 Purpose
This document defines the functional and non-functional requirements for the *Automated Multilingual News Aggregation, Summarization, and Bias Detection Tool*. It serves as a comprehensive guide for developing a web application that meets the project’s Stage 1 objectives, ensuring a structured and professional execution.

### 1.2 Scope
The initial scope focuses on building a web application with the following core functionalities:
- User authentication for secure access and preference management.
- News aggregation from reliable English language sources.
- Automated summarization of articles using large language models (LLMs).
- Bias detection to identify potential political bias in articles.
- A personalized news feed based on user-selected topics.

Future phases will expand to include multilingual support (e.g., German, Arabic), additional bias types, recommendation systems, and cross-platform accessibility (browser extension, mobile app).

### 1.3 Intended Audience
This document is intended for the project developer, future collaborators, and stakeholders interested in understanding the technical requirements of the web application.

### 1.4 References
- Project Vision Document: [Link to vision.md]

---

## 2. Functional Requirements
These specify what the system must do, detailing inputs, processes, and outputs for each feature.

### 2.1 User Authentication
- **Requirement:** Users must register with an email and password.
- **Process:** Validate email uniqueness and store passwords securely (hashed).
- **Output:** Successful registration creates a user profile.
- **Requirement:** Users must log in and log out securely.
- **Process:** Authenticate credentials and manage session state.
- **Output:** Access to personalized features upon login; session termination on logout.
- **Requirement:** Users must reset forgotten passwords.
- **Process:** Send a reset link to the registered email.
- **Output:** User regains access after resetting their password.

### 2.2 News Aggregation
- **Requirement:** Aggregate articles from at least three reliable English sources (e.g., BBC, Reuters, AP).
- **Process:** Fetch articles via APIs (e.g., NewsAPI) or custom scripts based on user topics/keywords.
- **Output:** A collection of articles stored for processing.
- **Requirement:** Support periodic updates.
- **Process:** Schedule aggregation (e.g., hourly) to retrieve new articles.
- **Output:** Updated article database.

### 2.3 Automated Summarization
- **Requirement:** Generate concise summaries for each article using an open-source LLM (e.g., BART, T5).
- **Process:** Process article content through the LLM, limiting summaries to 2-3 sentences (~50 words).
- **Output:** A summary appended to each article.
- **Requirement:** Trigger summarization automatically.
- **Process:** Execute upon article aggregation.
- **Output:** Immediate availability of summaries.

### 2.4 Bias Detection
- **Requirement:** Analyze articles for political bias using an LLM.
- **Process:** Classify articles as neutral, left-leaning, or right-leaning (or similar scale).
- **Output:** Bias classification stored and displayed with each article.
- **Requirement:** Present bias results clearly.
- **Process:** Integrate bias score with article summary.
- **Output:** User-friendly bias indicator.

### 2.5 Personalized News Feed
- **Requirement:** Allow topic selection during registration or via settings.
- **Process:** Save user preferences in the database.
- **Output:** Persistent topic list per user.
- **Requirement:** Display a tailored news feed.
- **Process:** Filter aggregated articles by user topics.
- **Output:** Dynamic feed of relevant summaries and bias analyses.

### 2.6 User Interface
- **Requirement:** Provide an intuitive web interface.
- **Process:** Design pages for:
  - Registration/login.
  - Topic selection/management.
  - News feed browsing.
  - Article detail viewing (summary + bias).
- **Output:** Responsive, user-friendly interface accessible on desktop and mobile browsers.

---

## 3. Non-Functional Requirements
These define how the system should perform.

### 3.1 Performance
- News feed loads within 3 seconds under typical conditions.
- Summarization and bias detection complete within 5 seconds per article.

### 3.2 Security
- User data (email, hashed passwords, preferences) encrypted at rest.
- HTTPS used for all communications.
- Protection against common vulnerabilities (e.g., SQL injection, XSS).

### 3.3 Usability
- Interface navigable with clear instructions; core features accessible within three clicks.
- Responsive design for cross-device compatibility.

### 3.4 Scalability
- Initial capacity for 100 concurrent users.
- Architecture supports future growth (e.g., more users, languages).

---

## 4. System Architecture
- **Overview:** A modular web application built with Django.
- **Components:**
  - **Front-End:** HTML, CSS, JavaScript (optional React/Vue.js for interactivity).
  - **Back-End:** Django for request handling, authentication, and logic.
  - **Database:** SQLite (initially), upgradable to PostgreSQL.
  - **LLM Integration:** Open-source models via Transformers (Hugging Face).
  - **News Aggregation:** APIs (e.g., NewsAPI) or custom scraping scripts.

---

## 5. Data Requirements
- **Sources:** Articles from at least three English news outlets.
- **Storage:**
  - User data: email, hashed password, topic preferences.
  - Article data: title, content, source, summary, bias score.
- **Processing:**
  - Articles processed for summarization and bias upon aggregation.
  - Preferences filter articles for the feed.

### 5.1 Database and Data Model Design
- **Users Table:**
  - Fields: `user_id` (int), `email` (string), `password_hash` (string), `preferences` (JSON or related table).
- **Articles Table:**
  - Fields: `article_id` (int), `title` (string), `content` (text), `source` (string), `summary` (text), `bias_score` (float), `publication_date` (datetime).
- **Sources Table:**
  - Fields: `source_id` (int), `name` (string), `url` (string), `reliability_score` (float, optional).
- **Activity Logs:**
  - Fields: `log_id` (int), `user_id` (int), `action` (string), `timestamp` (datetime).

---

## 6. User Interface Design
- **Preliminary Design:** (To be sketched)
  - Login/registration page.
  - Dashboard with news feed.
  - Article detail page (summary + bias).
- **Principles:** Minimalistic, readable, intuitive navigation.

---

## 7. Use Cases and User Stories
These examples illustrate how users will interact with the system.

### 7.1 Use Case: User Registration
- **User Action:** A new user provides email, password, and selects topics of interest.
- **System Response:** The system verifies the email, securely stores the password, saves preferences, and creates a personalized news feed.

### 7.2 User Story: Viewing a Personalized News Feed
- **As a** registered user,
- **I want to** see a curated news feed based on my selected topics,
- **So that** I can quickly access relevant articles and summaries without information overload.

---

## 8. Quality Assurance and Testing
- **Testing Strategies:**
  - Unit testing for individual modules (e.g., authentication, summarization).
  - Integration testing for end-to-end workflows (e.g., aggregation to news feed display).
  - Usability testing to ensure a smooth user experience.
- **Metrics:**
  - News feed loads in less than 3 seconds.
  - Summarization completes in less than 5 seconds per article.
  - Bias detection accuracy validated against a benchmark dataset (if available).

---

## 9. Risk Management and Mitigation
- **Potential Risks:**
  - LLM selection/integration difficulties or inaccuracies in summarization/bias detection.
  - Poor data quality from news sources.
  - Performance bottlenecks with large article volumes.
  - Security vulnerabilities in user data management.
- **Mitigation Strategies:**
  - Research and test LLMs thoroughly; regular evaluation and tuning of models.
  - Validate source reliability; handle errors robustly with fallback mechanisms (e.g., caching articles).
  - Optimize processing (e.g., asynchronous tasks).
  - Periodic security audits and code reviews.

---

## 10. Project Timeline
- **Milestones:**
  - **Week 1-2:** Django setup, user authentication.
  - **Week 3-4:** News aggregation module.
  - **Week 5-6:** LLM summarization integration.
  - **Week 7-8:** Bias detection implementation.
  - **Week 9-10:** News feed and UI development.
  - **Week 11-12:** Testing, fixes, deployment.
- **Deliverable:** Fully functional web app hosted online

