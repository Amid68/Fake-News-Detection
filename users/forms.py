"""
@file forms.py
@brief Forms for user authentication and profile management.

This module provides form classes for user registration, profile editing,
and preference management.

@author Ameed Othman
@date 2025-03-05
"""

from django import forms
from django.contrib.auth.forms import (
    UserCreationForm,
    UserChangeForm,
    PasswordResetForm,
)
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _
from .models import CustomUser, UserPreference


class CustomUserCreationForm(UserCreationForm):
    """
    Form for user registration with enhanced validation.

    Extends Django's UserCreationForm with additional fields and validation.
    """

    email = forms.EmailField(
        required=True,
        help_text=_("A valid email address is required for account verification."),
    )

    class Meta:
        model = CustomUser
        fields = (
            "username",
            "email",
            "first_name",
            "last_name",
            "password1",
            "password2",
        )

    def __init__(self, *args, **kwargs):
        """Initialize form with custom attributes and widget improvements."""
        super().__init__(*args, **kwargs)

        # Add placeholders and classes to improve UX
        self.fields["username"].widget.attrs.update(
            {"class": "form-control", "placeholder": _("Choose a username")}
        )
        self.fields["email"].widget.attrs.update(
            {"class": "form-control", "placeholder": _("Your email address")}
        )
        self.fields["first_name"].widget.attrs.update(
            {"class": "form-control", "placeholder": _("First name (optional)")}
        )
        self.fields["last_name"].widget.attrs.update(
            {"class": "form-control", "placeholder": _("Last name (optional)")}
        )
        self.fields["password1"].widget.attrs.update(
            {"class": "form-control", "placeholder": _("Create a password")}
        )
        self.fields["password2"].widget.attrs.update(
            {"class": "form-control", "placeholder": _("Confirm your password")}
        )

    def clean_email(self):
        """Validate that the email is not already in use."""
        email = self.cleaned_data.get("email")
        if email and CustomUser.objects.filter(email=email).exists():
            raise ValidationError(_("This email address is already in use."))
        return email

    def save(self, commit=True):
        """Save the user instance with additional processing."""
        user = super().save(commit=False)
        user.email = self.cleaned_data["email"]

        if commit:
            user.save()
        return user


class CustomUserChangeForm(UserChangeForm):
    """
    Form for updating user profile information.
    """

    class Meta:
        model = CustomUser
        fields = ("first_name", "last_name", "email")

    def __init__(self, *args, **kwargs):
        """Initialize form with improved field attributes."""
        super().__init__(*args, **kwargs)

        # Add styling classes
        for field_name, field in self.fields.items():
            field.widget.attrs.update({"class": "form-control"})

    def clean_email(self):
        """Validate that the email is not already in use by another user."""
        email = self.cleaned_data.get("email")
        if (
            email
            and CustomUser.objects.exclude(pk=self.instance.pk)
            .filter(email=email)
            .exists()
        ):
            raise ValidationError(_("This email address is already in use."))
        return email


class UserPreferenceForm(forms.ModelForm):
    """
    Form for managing user topic preferences.
    """

    class Meta:
        model = UserPreference
        fields = ("topic_keyword",)

    def __init__(self, *args, **kwargs):
        """Initialize with user-specific constraints."""
        self.user = kwargs.pop("user", None)
        super().__init__(*args, **kwargs)

        self.fields["topic_keyword"].widget.attrs.update(
            {
                "class": "form-control",
                "placeholder": _(
                    'Enter a topic keyword (e.g., "Technology", "Politics")'
                ),
            }
        )

    def clean_topic_keyword(self):
        """Validate the topic keyword."""
        topic = self.cleaned_data.get("topic_keyword")

        # Convert to lowercase for consistency
        topic = topic.lower()

        # Check if already exists for this user
        if self.user and not self.instance.pk:
            if UserPreference.objects.filter(
                user=self.user, topic_keyword__iexact=topic
            ).exists():
                raise ValidationError(_("You've already added this topic."))

        # Validate topic length
        if len(topic) < 2:
            raise ValidationError(_("Topic must be at least 2 characters long."))

        return topic

    def save(self, commit=True):
        """Save the preference with user association."""
        preference = super().save(commit=False)
        if self.user and not preference.user_id:
            preference.user = self.user

        if commit:
            preference.save()

        return preference


class EnhancedPasswordResetForm(PasswordResetForm):
    """
    Enhanced password reset form with additional validation.
    """

    def clean_email(self):
        """Validate the email address exists in the system."""
        email = self.cleaned_data.get("email")
        if not CustomUser.objects.filter(email=email, is_active=True).exists():
            # Using a generic message for security
            raise ValidationError(_("Please enter a valid email address."))
        return email
