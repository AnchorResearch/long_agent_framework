"""
Long Agent Framework - A framework for studying long-running AI agent behavior
"""

__version__ = "0.1.0"

from .evaluation_framework import (
    Evaluator,
    ExperimentRunner,
    CodingTask,
    NoStringConstraint,
    AIDEAgent,
    TaskResult
)

__all__ = [
    "Evaluator",
    "ExperimentRunner",
    "CodingTask",
    "NoStringConstraint",
    "AIDEAgent",
    "TaskResult"
] 