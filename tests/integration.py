from __future__ import annotations
import unittest


class TestEmulator(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        print("Setting up")

    def test_emulatorData_exists(self):
        assert 1 == True

    @classmethod
    def tearDownClass(cls) -> None:
        print("Tearing down")

if __name__ == "__main__":
    unittest.main()
