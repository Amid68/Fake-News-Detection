# Integration Test Scenarios

## Test Case ID: TC-INT-001

**Title**: End-to-End User Journey - Registration to Personalized News

**Description**: Verify the complete user journey from registration to viewing personalized news

**Preconditions**:
- The application is running
- Test user does not exist in the system
- News articles exist in the system

**Test Steps**:
1. Navigate to the homepage
2. Click on "Register" or navigate to registration page
3. Complete registration with valid details
   - Username: "journey_user"
   - Email: "journey_user@example.com"
   - Password: "JourneyTest123!"
4. After successful registration, navigate to preferences
5. Add preferences for "Technology" and "Science"
6. Navigate to the news feed
7. Observe the personalized feed
8. Save an article for later reading
9. Navigate to saved articles section
10. View the saved article
11. Log out
12. Log back in with the same credentials
13. Verify preferences and saved articles persist

**Expected Results**:
- Registration completes successfully
- Preferences are saved correctly
- News feed shows articles matching preferences
- Article save functionality works
- Saved articles can be accessed
- User state persists across sessions

**Pass/Fail Criteria**: Complete user journey functions as expected

**Notes**: This test combines multiple user features into a single journey

## Test Case ID: TC-INT-002

**Title**: News Aggregation and Processing Pipeline

**Description**: Verify the complete pipeline from news aggregation to processing and display

**Preconditions**:
- The application is running
- Admin user is logged in
- News sources are configured

**Test Steps**:
1. Open a terminal
2. Navigate to the project directory
3. Activate the virtual environment
4. Run the command: `python manage.py fetch_news --limit 10 --no-cache`
5. Observe the command output
6. Verify new articles in the admin interface
7. Run the command: `python manage.py process_articles --type summarization`
8. Run the command: `python manage.py process_articles --type bias_detection`
9. Monitor processing tasks in admin interface
10. Wait for processing to complete
11. View articles on the frontend
12. Verify summaries and bias information are displayed

**Expected Results**:
- News articles are successfully fetched
- Processing tasks are created and executed
- Articles are summarized
- Bias is detected
- Processed information is displayed to users

**Pass/Fail Criteria**: Complete pipeline from fetching to display works correctly

**Notes**: This test requires command-line access and admin privileges

## Test Case ID: TC-INT-003

**Title**: Responsive Design and Mobile Compatibility

**Description**: Verify that the application works correctly across different devices and screen sizes

**Preconditions**:
- The application is running
- News articles exist in the system

**Test Steps**:
1. Access the application on a desktop browser
2. Test core functionality (browsing, article view, user features)
3. Resize browser window to tablet size (e.g., 768x1024)
4. Test the same functionality
5. Resize browser window to mobile size (e.g., 375x667)
6. Test the same functionality
7. If available, test on actual mobile devices
8. Rotate device between portrait and landscape orientations

**Expected Results**:
- All functionality works on desktop
- All functionality works on tablet size
- All functionality works on mobile size
- Layout adjusts appropriately to screen size
- Content remains readable and accessible
- Navigation elements adapt to smaller screens
- No horizontal scrolling required
- Orientation changes are handled correctly

**Pass/Fail Criteria**: Application is fully functional across different screen sizes

**Notes**: Browser developer tools can simulate different screen sizes

## Test Case ID: TC-INT-004

**Title**: API Functionality and Rate Limiting

**Description**: Verify that the API endpoints function correctly and enforce rate limits

**Preconditions**:
- The application is running
- API endpoints are implemented
- Test user with API access exists

**Test Steps**:
1. Test authentication endpoints:
   - POST to /api/auth/login with valid credentials
   - Obtain authentication token
2. Test article endpoints:
   - GET /api/articles with valid token
   - GET /api/articles/{id} for a specific article
3. Test user preference endpoints:
   - GET /api/user/preferences with valid token
   - POST to create a new preference
   - DELETE to remove a preference
4. Test rate limiting:
   - Make rapid repeated requests to an endpoint
   - Observe when rate limiting kicks in
5. Test authentication requirements:
   - Access protected endpoints without a token
   - Access protected endpoints with an invalid token

**Expected Results**:
- API endpoints return correct data in expected format
- Authentication works correctly
- CRUD operations on resources work correctly
- Rate limiting prevents excessive requests
- Unauthenticated requests to protected endpoints are rejected
- Responses include appropriate status codes

**Pass/Fail Criteria**: API functions correctly and enforces security measures

**Notes**: Use a tool like Postman, curl, or httpie for API testing

## Test Case ID: TC-INT-005

**Title**: Celery Task Management and Monitoring

**Description**: Verify that Celery tasks are properly managed and can be monitored

**Preconditions**:
- The application is running
- Admin user is logged in
- Celery workers and beat scheduler are running

**Test Steps**:
1. Start Celery worker: `celery -A news_aggregator worker -l info`
2. Start Celery beat: `celery -A news_aggregator beat -l info`
3. Monitor the terminal for log output
4. Navigate to the admin interface
5. Create or trigger processing tasks
6. Observe task progression in terminal logs
7. Check for any errors or exceptions
8. Stop and restart workers
9. Verify tasks resume processing
10. Check for any periodic tasks from beat scheduler

**Expected Results**:
- Celery workers start without errors
- Beat scheduler starts without errors
- Tasks are processed and logged
- No unexpected errors occur
- Workers recover after restart
- Periodic tasks execute on schedule if configured

**Pass/Fail Criteria**: Celery task management works reliably

**Notes**: This test requires command-line access and is more for system administrators

## Test Case ID: TC-INT-006

**Title**: Database Performance Under Load

**Description**: Verify that the database performs adequately under simulated load

**Preconditions**:
- The application is running
- Database has a significant number of articles (100+)
- Admin access is available

**Test Steps**:
1. Monitor database connection pool usage
2. Time database queries for article listing
3. Test pagination with different page sizes
4. Perform complex queries (filtering, sorting, searching)
5. Measure response times
6. Check for any N+1 query issues
7. Review database logs for slow queries
8. Test with increased concurrent connections if possible

**Expected Results**:
- Database queries complete within reasonable time
- No timeouts or connection errors
- Pagination works efficiently
- Complex queries remain performant
- No critical N+1 query issues

**Pass/Fail Criteria**: Database performs adequately under simulated load

**Notes**: This test requires database monitoring access and is more advanced

## Test Case ID: TC-INT-007

**Title**: Error Handling and Logging

**Description**: Verify that the application handles errors gracefully and logs them appropriately

**Preconditions**:
- The application is running
- Admin access is available
- Access to log files is available

**Test Steps**:
1. Intentionally trigger errors:
   - Access non-existent pages
   - Submit invalid form data
   - Break API request format
   - Simulate network issues if possible
2. Observe application behavior
3. Check for appropriate error messages
4. Verify application continues to function
5. Review log files for error entries
6. Check error log format and details

**Expected Results**:
- Application handles errors gracefully
- User-friendly error messages are displayed
- Application remains stable after errors
- Errors are logged with appropriate detail
- Log entries contain useful debugging information

**Pass/Fail Criteria**: Application handles errors gracefully and logs them effectively

**Notes**: This test intentionally causes errors to verify handling

## Test Case ID: TC-INT-008

**Title**: Security Features and Protections

**Description**: Verify that the application implements appropriate security features

**Preconditions**:
- The application is running
- Test user exists
- Admin access is available

**Test Steps**:
1. Test CSRF protection:
   - Attempt to submit forms without CSRF token
2. Test XSS protection:
   - Submit content with script tags
   - Check if rendered safely
3. Test SQL injection protection:
   - Submit suspicious SQL in search fields
4. Test authentication protection:
   - Attempt to access protected pages without login
   - Attempt to access admin without appropriate permissions
5. Test password security:
   - Attempt to set weak passwords
6. Test session security:
   - Check for secure cookies
   - Test session timeout

**Expected Results**:
- CSRF protection blocks invalid requests
- XSS attempts are neutralized
- SQL injection attempts fail
- Authentication prevents unauthorized access
- Weak passwords are rejected
- Sessions are secured appropriately

**Pass/Fail Criteria**: Application implements effective security protections

**Notes**: This test should be conducted carefully and in a controlled environment