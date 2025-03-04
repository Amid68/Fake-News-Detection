# Software Requirements Specification

**Project Title:** Automated Multilingual News Aggregation, Summarization & Bias Detection Tool  
**Document Version:** 2.0  
**Author:** Ameed Othman
**Date:** 04.03.2025

## 1. Introduction

### 1.1 Purpose
This document defines the functional and non-functional requirements for the Automated News Aggregation, Summarization, and Bias Detection Tool. It serves as a comprehensive guide for implementation, ensuring a clear understanding of system capabilities and constraints.

### 1.2 Scope
This specification covers the Minimum Viable Product (MVP) scope, focusing on:
- User authentication and preference management
- News aggregation from reliable English language sources
- Article summarization using LLMs
- Basic bias detection
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
- Project Vision Document v2.0
- System Design Document v2.0
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
- **FR-4.1:** The system shall aggregate news from at least three reliable English sources.
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
- **FR-6.1:** The system shall generate concise summaries (3-5 sentences) of news articles.
- **FR-6.2:** The system shall use an appropriate LLM for summarization.
- **FR-6.3:** The system shall process newly aggregated articles automatically.
- **FR-6.4:** The system shall handle summarization failures gracefully.
- **FR-6.5:** The system shall maintain the key information from the original article.

#### 2.3.2 Summary Management
- **FR-7.1:** The system shall store summaries with their source articles.
- **FR-7.2:** The system shall display summaries with links to full articles.
- **FR-7.3:** The system shall track summary generation timestamps.
- **FR-7.4:** The system shall allow manual triggering of summarization when necessary.

### 2.4 Bias Detection

#### 2.4.1 Bias Analysis
- **FR-8.1:** The system shall analyze articles for political bias using an LLM.
- **FR-8.2:** The system shall classify articles on a political spectrum scale.
- **FR-8.3:** The system shall provide confidence scores with bias assessments.
- **FR-8.4:** The system shall handle bias detection failures gracefully.
- **FR-8.5:** The system shall include disclaimer language about bias detection limitations.

#### 2.4.2 Bias Presentation
- **FR-9.1:** The system shall display bias assessments visually (e.g., spectrum indicator).
- **FR-9.2:** The system shall provide brief explanations of bias assessments.
- **FR-9.3:** The system shall allow users to provide feedback on bias assessments.

### 2.5 Personalized News Feed

#### 2.5.1 Feed Generation
- **FR-10.1:** The system shall filter articles based on user topic preferences.
- **FR-10.2:** The system shall display recent articles first by default.
- **FR-10.3:** The system shall provide alternate sorting options (relevance, source).
- **FR-10.4:** The system shall provide pagination or infinite scrolling.
- **FR-10.5:** The system shall update the feed when new content is available.

#### 2.5.2 Feed Interaction
- **FR-11.1:** The system shall allow users to save articles for later reading.
- **FR-11.2:** The system shall track read/unread status of articles.
- **FR-11.3:** The system shall allow users to hide specific articles.
- **FR-11.4:** The system shall support basic search functionality.

### 2.6 User Interface

#### 2.6.1 General UI Requirements
- **FR-12.1:** The system shall provide an intuitive, responsive web interface.
- **FR-12.2:** The system shall implement a clean, readable design for news content.
- **FR-12.3:** The system shall support both light and dark mode.
- **FR-12.4:** The system shall provide consistent navigation across all pages.
- **FR-12.5:** The system shall display appropriate loading indicators.

#### 2.6.2 Specific UI Views
- **FR-13.1:** The system shall include a login/registration page.
- **FR-13.2:** The system shall include a personalized feed page.
- **FR-13.3:** The system shall include an article detail view.
- **FR-13.4:** The system shall include a user preferences management page.
- **FR-13.5:** The system shall include an about/information page.

## 3. Non-Functional Requirements

### 3.1 Performance Requirements
- **NFR-1.1:** The system shall load the news feed within 3 seconds under normal conditions.
- **NFR-1.2:** The system shall complete summarization within 10 seconds per article.
- **NFR-1.3:** The system shall complete bias detection within 5 seconds per article.
- **NFR-1.4:** The system shall support at least 50 concurrent users.
- **NFR-1.5:** The system shall maintain responsiveness during background processing.

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
- **NFR-6.3:** The system shall include appropriate disclaimers about bias detection.
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

#### 4.1.2 Viewing Personalized News Feed
1. User logs into the application
2. System displays personalized feed based on preferences
3. User scrolls through summarized articles
4. User selects an article of interest
5. System displays article details with summary and bias analysis

#### 4.1.3 Managing Preferences
1. User navigates to preferences page
2. System displays current topic selections
3. User modifies topic selections
4. System updates preferences
5. System regenerates feed based on new preferences

### 4.2 Data Dictionary

| Term | Definition | Format | Validation Rules |
|------|------------|--------|------------------|
| User | Registered system user | Object | Valid email required |
| Article | News article from source | Object | Must have title, content, source |
| Summary | Condensed version of article | Text | 3-5 sentences |
| Bias Score | Political leaning assessment | Float | Range from -1.0 (left) to 1.0 (right) |
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

### 5.3 Bias Classification Scale
- -1.0 to -0.6: Strong left-leaning
- -0.6 to -0.2: Moderate left-leaning
- -0.2 to 0.2: Centrist/Neutral
- 0.2 to 0.6: Moderate right-leaning
- 0.6 to 1.0: Strong right-leaning

### 5.4 Limitations and Constraints
- Article processing limited by available computational resources
- News API rate limits and potential costs
- Inherent limitations in automated bias detection accuracy
- User account limit for MVP (100 maximum concurrent users)