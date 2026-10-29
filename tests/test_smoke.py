"""Smoke tests verifying package structure and importability."""

import romanized_nepali_nlp


def test_package_version():
    assert romanized_nepali_nlp.__version__ == "0.1.0"
