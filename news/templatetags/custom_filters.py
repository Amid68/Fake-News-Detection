"""
@author Ameed Othman
@date 05.04.2025
"""

from django import template

register = template.Library()

@register.filter
def multiply(value, arg):
    """Multiply the value by the argument"""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return ''

@register.filter
def divide(value, arg):
    """Divide the value by the argument"""
    try:
        return float(value) / float(arg)
    except (ValueError, TypeError, ZeroDivisionError):
        return 0

@register.filter
def min_with(value, arg):
    """Return the minimum of value and arg"""
    try:
        return min(float(value), float(arg))
    except (ValueError, TypeError):
        return 0