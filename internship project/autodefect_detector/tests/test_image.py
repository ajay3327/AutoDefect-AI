"""
Test Suite for AutoDefect AI.
"""
import pytest
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.estimator import classify_severity

def test_classify_severity():
    assert classify_severity(0.01) == "Minor"
    assert classify_severity(0.03) == "Moderate"
    assert classify_severity(0.10) == "Severe"
