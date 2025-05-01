import unittest


class TestHelloWorld(unittest.TestCase):
    def test_hello_world(self):
        self.assertEqual("Hello, World!", "Hello, World!")

    def test_hello_world_length(self):
        self.assertEqual(len("Hello, World!"), 13)

    def test_hello_world_type(self):
        self.assertIsInstance("Hello, World!", str)


if __name__ == "__main__":
    unittest.main()
