import pytest
from long_agent_framework import Evaluator

def test_evaluator_initialization():
    evaluator = Evaluator(time_budget=3600, constraints=["no_additional_imports"])
    assert evaluator.time_budget == 3600
    assert "no_additional_imports" in evaluator.constraints

def test_evaluator_task_execution():
    evaluator = Evaluator(time_budget=3600, constraints=["no_additional_imports"])
    results = evaluator.evaluate_task("implement_array_operations")
    assert hasattr(results, "summary")
    assert hasattr(results, "violations")
    assert hasattr(results, "performance_score") 