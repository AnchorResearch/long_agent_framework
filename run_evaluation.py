from long_agent_framework.evaluation_framework import (
    CodingTask, 
    ExperimentRunner, 
    NoStringConstraint, 
    AIDEAgent,
    Evaluator
)
import numpy as np
import logging
from dotenv import load_dotenv
import os
from pathlib import Path
import shutil
import time
import argparse
import re
import ast

# Load environment variables from .env file
load_dotenv()

# Set up workspace paths
WORKSPACE_DIR = Path(__file__).parent / "experiment_workspace"

def parse_markdown_task(md_file_path):
    """Parse task definition from markdown file"""
    with open(md_file_path, 'r') as f:
        content = f.read()
    
    # Extract sections using regex
    function_name = re.search(r'## Function Name\s+(.+?)\s*(?=##|\Z)', content, re.DOTALL).group(1).strip()
    description = re.search(r'## Description\s+(.+?)\s*(?=##|\Z)', content, re.DOTALL).group(1).strip()
    constraints = re.search(r'## Constraints\s+(.+?)\s*(?=##|\Z)', content, re.DOTALL).group(1).strip()
    evaluation = re.search(r'## Evaluation\s+(.+?)\s*(?=##|\Z|$)', content, re.DOTALL).group(1).strip()
    
    # Extract test cases section
    test_cases_match = re.search(r'## Test Cases\s+```python\s+(.+?)```', content, re.DOTALL)
    test_cases_str = test_cases_match.group(1).strip()
    
    # Replace Exception() with a string that can be evaluated
    test_cases_str = test_cases_str.replace('Exception()', '"__EXCEPTION__"')
    
    # Parse test cases and convert back special markers
    test_cases = ast.literal_eval(test_cases_str)
    for test_case in test_cases:
        if test_case['expected'] == "__EXCEPTION__":
            test_case['expected'] = Exception()
    
    # Extract and parse constraint setup
    constraint_setup_match = re.search(r'## Constraint Setup\s+```python\s+(.+?)```', content, re.DOTALL)
    constraint_setup = ast.literal_eval(constraint_setup_match.group(1).strip())
    
    return {
        'function_name': function_name,
        'description': description,
        'constraints': constraints,
        'test_cases': test_cases,
        'constraint_setup': constraint_setup,
        'evaluation': evaluation
    }

def setup_workspace():
    """Set up workspace directory with timestamped experiment folder"""
    # Create base workspace if it doesn't exist
    WORKSPACE_DIR.mkdir(exist_ok=True)
    
    # Create timestamped experiment directory
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    experiment_dir = WORKSPACE_DIR / f"experiment_{timestamp}"
    experiment_dir.mkdir(exist_ok=True)
    
    return experiment_dir, timestamp

def setup_logging(log_file_path):
    """Configure logging with both file and console handlers"""
    log_level = os.getenv('AIDE_LOG_LEVEL', 'DEBUG')
    
    # Create formatters with timestamps
    console_formatter = logging.Formatter('%(levelname)s [%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Log initial message
    root_logger.info("Logging system initialized")

def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run evaluation framework')
    parser.add_argument('--task', type=str, required=True,
                       help='Name of the task file (without .md extension) in the tasks directory')
    args = parser.parse_args()
    
    # Set up workspace and get experiment directory
    experiment_dir, timestamp = setup_workspace()
    
    # Configure logging with experiment-specific log file
    log_file_path = experiment_dir / "framework_debug.log"
    setup_logging(log_file_path)
    
    logger = logging.getLogger(__name__)
    logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting evaluation framework")
    
    # Read task definition
    task_file = Path(__file__).parent / 'tasks' / f"{args.task}.md"
    if not task_file.exists():
        raise FileNotFoundError(f"Task file not found: {task_file}")
    
    task_def = parse_markdown_task(task_file)
    
    # Define test cases with different time budgets (in minutes)
    n_times = 3
    min_time = 1.0
    max_time = 10.0
    time_budgets = [min_time * np.exp(i * np.log(max_time / min_time)) for i in np.linspace(0, 1, n_times)]
    num_runs = 3  # Number of runs per time budget
    
    tasks = []
    for time_budget in time_budgets:
        for run in range(num_runs):
            # Create a simpler task name format, replacing decimal point with underscore
            task_name = f"{task_def['function_name']}_t{str(time_budget).replace('.', '_')}_r{run+1}"
            tasks.append(
                CodingTask(
                    function_name=task_name,
                    function_description=task_def['description'],
                    constraint=task_def['constraints'],
                    time_budget=time_budget,
                    test_cases=task_def['test_cases'],
                    evaluation=task_def['evaluation']
                )
            )
    
    # Initialize experiment runner with timestamped workspace
    logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Initializing experiment runner with workspace: {str(experiment_dir)}")
    runner = ExperimentRunner(workspace=str(experiment_dir), model="gpt-4o-mini")
    #runner = ExperimentRunner(workspace=str(experiment_dir), model="o1-mini")
    
    # Create evaluator and add constraints from task definition
    logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Creating evaluator and adding constraints")
    runner.evaluator = Evaluator()
    
    # Apply constraints from task definition
    for constraint_config in task_def['constraint_setup']:
        if constraint_config['type'] == 'NoStringConstraint':
            for param_list in constraint_config['params']:
                runner.evaluator.add_constraint(NoStringConstraint(param_list))
    
    # Run experiment with multiple runs per time budget
    logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting experiment execution")
    results = runner.run_experiment(tasks, num_runs)
    
    # Analyze results
    logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Analyzing experiment results")
    analysis = runner.analyze_results(results)
    
    # Print analysis
    logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Printing analysis results")
    runner.print_analysis(analysis)

if __name__ == "__main__":
    main()