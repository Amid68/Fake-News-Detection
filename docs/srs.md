# Software Requirements Specification

**Project Title:** Automated News Aggregation, Summarization & Topic Analysis Tool  
**Document Version:** 3.0  
**Author:** Ameed Othman
**Date:** 10.03.2025

## 1. Introduction

### 1.1 Purpose
This document defines the functional and non-functional requirements for the Automated News Aggregation, Summarization, and Topic Analysis Tool. It serves as a comprehensive guide for implementation.

### 1.2 Scope
This specification covers the Minimum Viable Product (MVP) scope, focusing on:
- User authentication and preference management
- News aggregation from reliable sources
- Article summarization with quality metrics
- Topic analysis and entity extraction
- Topic trend visualization
- Personalized news feed generation
- Web-based user interface

### 1.3 Definitions, Acronyms, and Abbreviations
- **LLM:** Large Language Model
- **API:** Application Programming Interface
- **MVP:** Minimum Viable Product
- **NLP:** Natural Language Processing
- **ROUGE:** Recall-Oriented Understudy for Gisting Evaluation (summarization metric)
- **UI/UX:** User Interface/User Experience

### 1.4 References
- Project Vision Document v3.0
- System Design Document v3.0
- Evaluation and Testing Plan
- [NewsAPI Documentation](https://newsapi.org/docs)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)

## 2. Functional Requirements

### 2.1 User Authentication and Profile Management

#### 2.1.1 User Registration
- **FR-1.1:** The system shall allow users to register with email and password.
- **FR-1.2:** The system shall validate email format and uniqueness.
- **FR-1.3:** The system shall enforce password complexity requirements (min. 8 characters, including numbers/symbols).
- **FR-1.4:** The system shall store passwords using secure hashing (bcrypt).
- **FR-1.5:** The system shall send a verification email upon registration.

#### 2.1.2 User Authentication
- **FR-2.1:** The system shall authenticate users via email/password.
- **FR-2.2:** The system shall provide secure session management.
- **FR-2.3:** The system shall implement password reset functionality.
- **FR-2.4:** The system shall log failed login attempts.
- **FR-2.5:** The system shall allow users to log out from any page.

#### 2.1.3 User Preferences
- **FR-3.1:** The system shall allow users to select news topics of interest.
- **FR-3.2:** The system shall allow users to modify their topic preferences.
- **FR-3.3:** The system shall store user preferences securely.
- **FR-3.4:** The system shall allow users to prioritize preferred news sources.

### 2.2 News Aggregation

#### 2.2.1 Content Collection
- **FR-4.1:** The system shall aggregate news from at least three reliable sources.
- **FR-4.2:** The system shall fetch articles based on user-selected topics.
- **FR-4.3:** The system shall update content at least once per hour.
- **FR-4.4:** The system shall properly attribute content to its original source.
- **FR-4.5:** The system shall handle API rate limits gracefully.
- **FR-4.6:** The system shall store minimal required content to respect copyright.

#### 2.2.2 Content Management
- **FR-5.1:** The system shall categorize articles by topic.
- **FR-5.2:** The system shall store publication dates and update timestamps.
- **FR-5.3:** The system shall handle duplicate articles appropriately.
- **FR-5.4:** The system shall maintain links to original sources.
- **FR-5.5:** The system shall implement a retention policy for older articles.

### 2.3 Article Summarization

#### 2.3.1 Summary Generation
- **FR-6.1:** The system shall generate summaries of articles in multiple formats:
  - Headline (1-2 sentences)
  - Standard (3-5 sentences)
  - Detailed (6-8 sentences)
- **FR-6.2:** The system shall use appropriate LLMs for summarization.
- **FR-6.3:** The system shall process newly aggregated articles automatically.
- **FR-6.4:** The system shall calculate quality metrics for generated summaries.
- **FR-6.5:** The system shall maintain the key information from the original article.

#### 2.3.2 Summary Management
- **FR-7.1:** The system shall store summaries with their source articles.
- **FR-7.2:** The system shall display summaries with links to full articles.
- **FR-7.3:** The system shall track summary generation timestamps.
- **FR-7.4:** The system shall allow users to select preferred summary length.

### 2.4 Topic Analysis

#### 2.4.1 Topic Classification
- **FR-8.1:** The system shall analyze articles for primary and secondary topics.
- **FR-8.2:** The system shall extract key terms from articles.
- **FR-8.3:** The system shall identify named entities (people, places, organizations).
- **FR-8.4:** The system shall provide confidence scores for topic classifications.
- **FR-8.5:** The system shall perform simple sentiment analysis on articles.

#### 2.4.2 Topic Presentation
- **FR-9.1:** The system shall display topic information visually.
- **FR-9.2:** The system shall provide a list of extracted entities.
- **FR-9.3:** The system shall allow users to explore content through topic connections.

### 2.5 Visualization Dashboard

#### 2.5.1 Dashboard Components
- **FR-10.1:** The system shall provide a topic distribution visualization.
- **FR-10.2:** The system shall display topic trends over time.
- **FR-10.3:** The system shall visualize source distribution of articles.
- **FR-10.4:** The system shall present keyword frequency visualization.
- **FR-10.5:** The system shall show sentiment distribution across content.

#### 2.5.2 Dashboard Interaction
- **FR-11.1:** The system shall allow filtering visualizations by date range.
- **FR-11.2:** The system shall support drilling down into specific topics.
- **FR-11.3:** The system shall provide tooltips with additional information.
- **FR-11.4:** The system shall export visualization data if requested.

### 2.6 Personalized News Feed

#### 2.6.1 Feed Generation
- **FR-12.1:** The system shall filter articles based on user topic preferences.
- **FR-12.2:** The system shall display recent articles first by default.
- **FR-12.3:** The system shall provide alternate sorting options (relevance, source).
- **FR-12.4:** The system shall provide pagination or infinite scrolling.
- **FR-12.5:** The system shall update the feed when new content is available.

#### 2.6.2 Feed Interaction
- **FR-13.1:** The system shall allow users to save articles for later reading.
- **FR-13.2:** The system shall track read/unread status of articles.
- **FR-13.3:** The system shall allow users to hide specific articles.
- **FR-13.4:** The system shall support basic search functionality.

### 2.7 User Interface

#### 2.7.1 General UI Requirements
- **FR-14.1:** The system shall provide an intuitive, responsive web interface.
- **FR-14.2:** The system shall implement a clean, readable design for news content.
- **FR-14.3:** The system shall support both light and dark mode.
- **FR-14.4:** The system shall provide consistent navigation across all pages.
- **FR-14.5:** The system shall display appropriate loading indicators.

#### 2.7.2 Specific UI Views
- **FR-15.1:** The system shall include a login/registration page.
- **FR-15.2:** The system shall include a personalized feed page.
- **FR-15.3:** The system shall include an article detail view with topic insights.
- **FR-15.4:** The system shall include a user preferences management page.
- **FR-15.5:** The system shall include an analytics dashboard page.

## 3. Non-Functional Requirements

### 3.1 Performance Requirements
- **NFR-1.1:** The system shall load the news feed within 3 seconds under normal conditions.
- **NFR-1.2:** The system shall complete summarization within 10 seconds per article.
- **NFR-1.3:** The system shall complete topic analysis within 5 seconds per article.
- **NFR-1.4:** The system shall render visualizations within 2 seconds.
- **NFR-1.5:** The system shall support at least 50 concurrent users.

### 3.2 Security Requirements
- **NFR-2.1:** The system shall implement HTTPS for all communications.
- **NFR-2.2:** The system shall encrypt sensitive user data at rest.
- **NFR-2.3:** The system shall implement protection against common web vulnerabilities (OWASP Top 10).
- **NFR-2.4:** The system shall implement rate limiting for authentication attempts.
- **NFR-2.5:** The system shall implement proper input validation for all user inputs.

### 3.3 Usability Requirements
- **NFR-3.1:** The system shall be usable on desktop and mobile devices.
- **NFR-3.2:** The system shall provide clear error messages to users.
- **NFR-3.3:** The system shall include help text for complex features.
- **NFR-3.4:** The system shall follow web accessibility guidelines (WCAG 2.1 AA).
- **NFR-3.5:** The system shall support keyboard navigation.

### 3.4 Reliability Requirements
- **NFR-4.1:** The system shall be available 99% of the time (excluding planned maintenance).
- **NFR-4.2:** The system shall implement graceful degradation for API failures.
- **NFR-4.3:** The system shall log all errors for troubleshooting.
- **NFR-4.4:** The system shall implement database backups.
- **NFR-4.5:** The system shall recover from crashes without data loss.

### 3.5 Compatibility Requirements
- **NFR-5.1:** The system shall function on current versions of major browsers (Chrome, Firefox, Safari, Edge).
- **NFR-5.2:** The system shall degrade gracefully on older browsers.
- **NFR-5.3:** The system shall be responsive on screen sizes from 320px to 1920px width.

### 3.6 Legal and Compliance Requirements
- **NFR-6.1:** The system shall comply with GDPR for user data handling.
- **NFR-6.2:** The system shall respect copyright of original news sources.
- **NFR-6.3:** The system shall include appropriate disclaimers about automated analysis.
- **NFR-6.4:** The system shall provide terms of service and privacy policy.
- **NFR-6.5:** The system shall implement appropriate data retention policies.

## 4. System Models

### 4.1 Use Cases

#### 4.1.1 User Registration and Login
1. User navigates to the application
2. User selects "Register" option
3. User provides email and password
4. System validates input and creates account
5. User selects topics of interest
6. System creates personalized feed

#### 4.1.2 Viewing Topic Insights Dashboard
1. User logs into the application
2. User navigates to the Analytics Dashboard
3. System displays topic distribution visualization
4. User selects a time range filter
5. System updates visualizations based on filter
6. User explores different visualization components

#### 4.1.3 Reading Article with Topic Analysis
1. User navigates to article detail page
2. System displays article summary
3. User views topic analysis tab
4. System shows topic classification and entities
5. User explores related content by clicking on topics

### 4.2 Data Dictionary

| Term | Definition | Format | Validation Rules |
|------|------------|--------|------------------|
| User | Registered system user | Object | Valid email required |
| Article | News article from source | Object | Must have title, content, source |
| Summary | Condensed version of article | Text | Variable length based on type |
| Topic Analysis | Topic classification results | Object | Must have primary topic |
| Entity | Named entity in article | Object | Type and text required |
| Topic | News category | String | From predefined list |
| Source | News provider | Object | Must be from approved list |

## 5. Appendices

### 5.1 Supported News Sources (Initial)
- BBC News
- Reuters
- Associated Press
- The Guardian
- Al Jazeera English

### 5.2 Supported Topics (Initial)
- Politics
- Business
- Technology
- Science
- Health
- Sports
- Entertainment
- World/International

### 5.3 Topic Analysis Components
- Primary and secondary topic classification
- Named entity recognition (people, organizations, locations)
- Key term extraction
- Simple sentiment analysis

### 5.4 Visualization Components
- Topic distribution charts
- Topic trends over time
- Source distribution analysis
- Keyword frequency visualization
- Sentiment distribution