"""
Long Agent Framework - A framework for studying long-running AI agent behavior
"""

__version__ = "0.1.0"

from .evaluation_framework import Evaluator
from .run_evaluation import run_evaluation

__all__ = ["Evaluator", "run_evaluation"] 