from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from .models import CustomUser, UserPreference


class UserPreferenceInline(admin.TabularInline):
    model = UserPreference
    extra = 1


class CustomUserAdmin(UserAdmin):
    model = CustomUser
    list_display = (
        "username",
        "email",
        "first_name",
        "last_name",
        "is_staff",
        "registration_date",
    )
    fieldsets = UserAdmin.fieldsets + (
        ("Custom Fields", {"fields": ("registration_date",)}),
    )
    readonly_fields = ("registration_date",)
    inlines = [UserPreferenceInline]


admin.site.register(CustomUser, CustomUserAdmin)
admin.site.register(UserPreference)
