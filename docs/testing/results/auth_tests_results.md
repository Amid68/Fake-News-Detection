# User Authentication Testing Results

**Date:** April 3, 2025  
**Tester:** Ameed Othman    
**Environment:** Development (localhost)    
**Browser:** Google Chrome

## Test Cases and Results

### 1. User Registration

| Test ID | Description | Steps | Expected Result | Actual Result | Status | Notes |
|---------|-------------|-------|-----------------|---------------|--------|-------|
| AUTH-01 | Valid Registration | 1. Navigate to registration form<br>2. Enter valid details<br>3. Submit form | Redirect to home page with success message | Registration successful, redirected to home page | ✅ PASS | |
| AUTH-02 | Registration with weak password | 1. Navigate to registration form<br>2. Enter valid details with weak password<br>3. Submit form | Form shows error message | Not tested yet | ❓ PENDING | |
| AUTH-03 | Registration with duplicate email | 1. Navigate to registration form<br>2. Enter details with existing email<br>3. Submit form | Form shows error message | Not tested yet | ❓ PENDING | |

### 2. User Login

| Test ID | Description | Steps | Expected Result | Actual Result | Status | Notes |
|---------|-------------|-------|-----------------|---------------|--------|-------|
| AUTH-04 | Valid Login | 1. Navigate to login page<br>2. Enter valid credentials<br>3. Submit form | Redirect to home page, user session created | Not explicitly tested yet | ❓ PENDING | |
| AUTH-05 | Invalid Login | 1. Navigate to login page<br>2. Enter invalid credentials<br>3. Submit form | Form shows error message | Not tested yet | ❓ PENDING | |

### 3. User Logout

| Test ID | Description | Steps | Expected Result | Actual Result | Status | Notes |
|---------|-------------|-------|-----------------|---------------|--------|-------|
| AUTH-06 | Logout functionality | 1. Login<br>2. Click logout link | Redirect to home page, session ended | Method Not Allowed error (GET instead of POST) | ❌ FAIL | Logout link uses GET method instead of required POST |

## Issues Found

### Issue 1: Logout Method Error
- **Issue**: Logout link uses GET request, but Django's LogoutView requires POST
- **Error Message**: `WARNING Method Not Allowed (GET): /users/logout/`
- **Fix**: Update logout link in templates/base.html to use a form with POST method
- **Priority**: High

### Issue 2: Missing Template Filter
- **Issue**: Custom 'multiply' filter is missing but used in templates
- **Error Message**: `django.template.exceptions.TemplateSyntaxError: Invalid filter: 'multiply'`
- **Fix**: Create custom template filters in news/templatetags/custom_filters.py
- **Priority**: High

### Issue 3: Missing Search Template
- **Issue**: Search view references a template that doesn't exist
- **Error Message**: `django.template.exceptions.TemplateDoesNotExist: news/search_results.html`
- **Fix**: Create the missing search results template
- **Priority**: High

## Action Items

1. Fix logout link to use POST method
2. Create custom template filters
3. Create search results template
4. Complete remaining test cases