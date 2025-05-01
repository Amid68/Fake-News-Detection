from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

from .forms import CustomUserChangeForm, CustomUserCreationForm, UserPreferenceForm


def register_view(request):
    if request.method == "POST":
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f"Account created for {user.username}!")
            return redirect("home")  # Redirect to home page
        else:
            # If form is not valid, errors will be displayed automatically
            pass
    else:
        form = CustomUserCreationForm()
    return render(request, "users/register.html", {"form": form})


@login_required
def profile_view(request):
    """
    View function for user profile page with preference management.

    Allows users to update their profile information and manage topic preferences.
    """
    # Get user's current preferences
    user_preferences = request.user.preferences.all()

    # Handle profile update form
    if request.method == "POST" and "update_profile" in request.POST:
        user_form = CustomUserChangeForm(request.POST, instance=request.user)
        if user_form.is_valid():
            user_form.save()
            messages.success(request, "Your profile has been updated successfully!")
            return redirect("profile")
    else:
        user_form = CustomUserChangeForm(instance=request.user)

    # Handle preference form
    preference_form = UserPreferenceForm(user=request.user)

    # Handle adding a new preference
    if request.method == "POST" and "add_preference" in request.POST:
        preference_form = UserPreferenceForm(request.POST, user=request.user)
        if preference_form.is_valid():
            preference = preference_form.save(commit=False)
            preference.user = request.user
            preference.save()
            messages.success(
                request, f'Added "{preference.topic_keyword}" to your preferences!'
            )
            return redirect("profile")

    # Handle removing a preference
    if request.method == "POST" and "remove_preference" in request.POST:
        preference_id = request.POST.get("preference_id")
        try:
            preference = user_preferences.get(id=preference_id)
            preference.delete()
            messages.success(
                request, f'Removed "{preference.topic_keyword}" from your preferences!'
            )
            return redirect("profile")
        except Exception as e:
            messages.error(request, f"Error removing preference: {str(e)}")

    # Get user's reading history
    from news.models import ArticleViewHistory

    view_history = (
        ArticleViewHistory.objects.filter(user=request.user)
        .select_related("article")
        .order_by("-viewed_at")[:10]
    )

    # Get user's saved articles
    from news.models import UserSavedArticle

    saved_articles = (
        UserSavedArticle.objects.filter(user=request.user)
        .select_related("article")
        .order_by("-saved_at")[:5]
    )

    context = {
        "user_form": user_form,
        "preference_form": preference_form,
        "user_preferences": user_preferences,
        "view_history": view_history,
        "saved_articles": saved_articles,
    }

    return render(request, "users/profile.html", context)
