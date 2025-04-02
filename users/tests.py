"""
@file users/tests.py
@brief Test cases for the users app.

This module provides test cases for models, views, and forms
in the users app.

@author Ameed Othman
@date 2025-04-02
"""

from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth import get_user_model
from django.contrib.messages import get_messages

from .models import CustomUser, UserPreference
from .forms import CustomUserCreationForm, UserPreferenceForm

User = get_user_model()


class UsersModelTests(TestCase):
    """Tests for users app models."""

    def setUp(self):
        """Set up test data."""
        # Create a test user
        self.user = User.objects.create_user(
            username="testuser",
            email="test@example.com",
            password="testpassword123",
            first_name="Test",
            last_name="User"
        )

        # Create a test preference
        self.preference = UserPreference.objects.create(
            user=self.user,
            topic_keyword="Technology"
        )

    def test_user_creation(self):
        """Test custom user model creation."""
        self.assertEqual(self.user.username, "testuser")
        self.assertEqual(self.user.email, "test@example.com")
        self.assertEqual(self.user.get_full_name(), "Test User")
        self.assertTrue(self.user.registration_date is not None)
        self.assertEqual(str(self.user), "testuser")

    def test_user_preference(self):
        """Test user preference model creation."""
        self.assertEqual(self.preference.user, self.user)
        self.assertEqual(self.preference.topic_keyword, "Technology")
        self.assertEqual(str(self.preference), "testuser - Technology")

        # Test getting user preferences
        preferences = self.user.preferences.all()
        self.assertEqual(preferences.count(), 1)
        self.assertEqual(preferences.first(), self.preference)

    def test_unique_preference(self):
        """Test that topic_keyword must be unique per user."""
        # Try to create a duplicate preference
        with self.assertRaises(Exception):
            UserPreference.objects.create(
                user=self.user,
                topic_keyword="Technology"
            )

        # Should be able to create a different preference
        pref2 = UserPreference.objects.create(
            user=self.user,
            topic_keyword="Politics"
        )
        self.assertEqual(pref2.topic_keyword, "Politics")

        # Another user should be able to have the same preference
        user2 = User.objects.create_user(
            username="testuser2",
            email="test2@example.com",
            password="testpassword123"
        )
        pref3 = UserPreference.objects.create(
            user=user2,
            topic_keyword="Technology"
        )
        self.assertEqual(pref3.topic_keyword, "Technology")


class UsersViewTests(TestCase):
    """Tests for users app views."""

    def setUp(self):
        """Set up test data."""
        # Create client
        self.client = Client()

        # Create a test user
        self.user = User.objects.create_user(
            username="testuser",
            email="test@example.com",
            password="testpassword123"
        )

        # User registration data
        self.registration_data = {
            'username': 'newuser',
            'email': 'newuser@example.com',
            'password1': 'secure_password_123',
            'password2': 'secure_password_123'
        }

    def test_register_view_get(self):
        """Test register view GET request."""
        response = self.client.get(reverse('register'))

        # Check response status code
        self.assertEqual(response.status_code, 200)

        # Check that form is in the context
        self.assertTrue('form' in response.context)
        self.assertIsInstance(response.context['form'], CustomUserCreationForm)

    def test_register_view_post_valid(self):
        """Test register view POST request with valid data."""
        response = self.client.post(reverse('register'), self.registration_data)

        # Check redirect
        self.assertEqual(response.status_code, 302)

        # Check that user is created
        self.assertTrue(User.objects.filter(username='newuser').exists())

        # Check that user is logged in
        user = User.objects.get(username='newuser')
        self.assertEqual(int(self.client.session['_auth_user_id']), user.id)

    def test_register_view_post_invalid(self):
        """Test register view POST request with invalid data."""
        # Create invalid registration data (missing email)
        invalid_data = self.registration_data.copy()
        invalid_data.pop('email')

        response = self.client.post(reverse('register'), invalid_data)

        # Check that form is invalid
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.context['form'].is_valid())

        # Check that user is not created
        self.assertFalse(User.objects.filter(username='newuser').exists())

    def test_profile_view_authenticated(self):
        """Test profile view for authenticated users."""
        # Login user
        self.client.login(username="testuser", password="testpassword123")

        # Add a preference
        UserPreference.objects.create(
            user=self.user,
            topic_keyword="Technology"
        )

        # Get profile page
        response = self.client.get(reverse('profile'))

        # Check response status code
        self.assertEqual(response.status_code, 200)

        # Check that forms are in the context
        self.assertTrue('user_form' in response.context)
        self.assertTrue('preference_form' in response.context)

        # Check that user preferences are in the context
        self.assertTrue('user_preferences' in response.context)
        self.assertEqual(len(response.context['user_preferences']), 1)

    def test_profile_view_unauthenticated(self):
        """Test profile view for unauthenticated users."""
        response = self.client.get(reverse('profile'))

        # Check redirect to login
        self.assertEqual(response.status_code, 302)
        self.assertTrue('/login/' in response.url)

    def test_add_preference(self):
        """Test adding a user preference."""
        # Login user
        self.client.login(username="testuser", password="testpassword123")

        # Add a preference
        response = self.client.post(reverse('profile'), {
            'add_preference': True,
            'topic_keyword': 'Technology'
        })

        # Check redirect
        self.assertEqual(response.status_code, 302)

        # Check that preference is added
        preferences = UserPreference.objects.filter(user=self.user)
        self.assertEqual(preferences.count(), 1)
        self.assertEqual(preferences.first().topic_keyword, 'Technology')

        # Check for success message
        messages = list(get_messages(response.wsgi_request))
        self.assertTrue(any('Added "Technology" to your preferences' in str(m) for m in messages))


class UsersFormTests(TestCase):
    """Tests for users app forms."""

    def setUp(self):
        """Set up test data."""
        # Create a test user
        self.user = User.objects.create_user(
            username="testuser",
            email="test@example.com",
            password="testpassword123"
        )

    def test_user_creation_form_valid(self):
        """Test user creation form with valid data."""
        form_data = {
            'username': 'newuser',
            'email': 'newuser@example.com',
            'password1': 'secure_password_123',
            'password2': 'secure_password_123'
        }
        form = CustomUserCreationForm(data=form_data)
        self.assertTrue(form.is_valid())

    def test_user_creation_form_invalid(self):
        """Test user creation form with invalid data."""
        # Test with mismatched passwords
        form_data = {
            'username': 'newuser',
            'email': 'newuser@example.com',
            'password1': 'secure_password_123',
            'password2': 'different_password'
        }
        form = CustomUserCreationForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertTrue('password2' in form.errors)

        # Test with duplicate email
        form_data = {
            'username': 'newuser',
            'email': 'test@example.com',  # Already exists
            'password1': 'secure_password_123',
            'password2': 'secure_password_123'
        }
        form = CustomUserCreationForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertTrue('email' in form.errors)

    def test_user_preference_form_valid(self):
        """Test user preference form with valid data."""
        form_data = {
            'topic_keyword': 'Technology'
        }
        form = UserPreferenceForm(data=form_data, user=self.user)
        self.assertTrue(form.is_valid())

    def test_user_preference_form_duplicate(self):
        """Test user preference form with duplicate topic."""
        # Create an existing preference
        UserPreference.objects.create(
            user=self.user,
            topic_keyword="Technology"
        )

        # Try to add the same preference
        form_data = {
            'topic_keyword': 'Technology'
        }
        form = UserPreferenceForm(data=form_data, user=self.user)
        self.assertFalse(form.is_valid())
        self.assertTrue('topic_keyword' in form.errors)

    def test_user_preference_form_different_user(self):
        """Test user preference form with different user."""
        # Create an existing preference for the first user
        UserPreference.objects.create(
            user=self.user,
            topic_keyword="Technology"
        )

        # Create a second user
        user2 = User.objects.create_user(
            username="testuser2",
            email="test2@example.com",
            password="testpassword123"
        )

        # Same preference should be valid for second user
        form_data = {
            'topic_keyword': 'Technology'
        }
        form = UserPreferenceForm(data=form_data, user=user2)
        self.assertTrue(form.is_valid())