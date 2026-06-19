"""Make the package modules importable when running `pytest` from senra-eval/."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
