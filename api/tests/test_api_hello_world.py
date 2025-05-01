def hello_world(input=None):
    return "Hello, World!"


def test_hello_world():
    assert hello_world() == "Hello, World!"


def test_hello_world_empty_input():
    assert hello_world("") == "Hello, World!"


def test_hello_world_special_characters():
    assert hello_world("!@#$%^&*()") == "Hello, World!"


def test_hello_world_none_input():
    assert hello_world(None) == "Hello, World!"
