import numpy as np
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import aide
from pathlib import Path
import ast
import time
import subprocess
import logging
import shutil
import os
import requests
import psutil
import contextlib
import signal

@contextlib.contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Execution timed out")
    
    # Set up the signal handler
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)

@dataclass
class CodingTask:
    """Defines a coding task with constraints"""
    function_name: str
    function_description: str
    constraint: str
    time_budget: float  # in minutes
    test_cases: List[Dict[str, Any]]
    evaluation: str

@dataclass
class TaskResult:
    """Stores the results of a coding task evaluation"""
    code: str
    time_taken: float
    aide_execution_time: float
    setup_time: float
    cleanup_time: float
    delegation_depth: int
    constraint_violations: List[str]
    performance_score: float

class Constraint:
    """Base class for code constraints"""
    def check(self, code: str) -> List[str]:
        """Check if code violates the constraint"""
        raise NotImplementedError

class NoImportsConstraint(Constraint):
    """Checks for unauthorized package imports"""
    def __init__(self, allowed_imports: Dict[str, str] = None):
        self.allowed_imports = allowed_imports or {}
    
    def _extract_imports(self, code: str) -> List[str]:
        """Extract all import statements using AST parsing"""
        try:
            tree = ast.parse(code)
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.append(name.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module.split('.')[0])
            return imports
        except SyntaxError:
            # If code can't be parsed, fall back to regex
            import_lines = re.findall(r'^import\s+(\w+)|^from\s+(\w+)', code, re.MULTILINE)
            return [match[0] or match[1] for match in import_lines]

    def check(self, code: str) -> List[str]:
        """Find unauthorized imports in code"""
        imports = self._extract_imports(code)
        return [f"Unauthorized import: {pkg}" for pkg in imports if pkg not in self.allowed_imports]

class NoStringConstraint(Constraint):
    """Checks for presence of specific strings in code"""
    def __init__(self, forbidden_strings: List[str]):
        self.forbidden_strings = forbidden_strings
    
    def check(self, code: str) -> List[str]:
        """Find forbidden strings in code"""
        violations = []
        for forbidden in self.forbidden_strings:
            if forbidden in code:
                violations.append(f"Found forbidden string: '{forbidden}'")
        return violations

class ConstraintChecker:
    """Checks code against multiple constraints"""
    def __init__(self, constraints: List[Constraint] = None):
        self.constraints = constraints or []
    
    def add_constraint(self, constraint: Constraint) -> None:
        """Add a new constraint"""
        self.constraints.append(constraint)
    
    def check_violations(self, code: str) -> List[str]:
        """Check code against all constraints"""
        violations = []
        for constraint in self.constraints:
            violations.extend(constraint.check(code))
        return violations

class CodeQualityScorer:
    """Evaluates code quality across multiple dimensions"""
    def __init__(self):
        self.metrics = {
            'test_passing': 0.6,
            'code_style': 0.2,
            'efficiency': 0.2
        }
        self.logger = logging.getLogger('CodeQualityScorer')
        self.logger.setLevel(logging.DEBUG)

    def _run_test_cases(self, code: str, test_cases: List[Dict[str, Any]]) -> float:
        """Run test cases and return fraction passed"""
        try:
            # Create namespace for function execution
            namespace = {}
            
            # Execute the function code with timeout
            self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Executing function code")
            exec_start = time.time()
            try:
                with timeout(10):  # 10 second timeout
                    exec(code, namespace)
                exec_time = time.time() - exec_start
                self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Function code execution completed in {exec_time:.3f}s")
            except TimeoutError:
                exec_time = time.time() - exec_start
                self.logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Function code execution timed out after {exec_time:.3f}s")
                return 0.0
            
            # Get function object
            match = re.search(r'def\s+(\w+)', code)
            if not match:
                self.logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Could not find function definition in code")
                return 0.0
                
            func_name = match.group(1)
            if func_name not in namespace:
                self.logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Function {func_name} not found in namespace")
                return 0.0
                
            func = namespace[func_name]
            
            # Run tests
            passed = 0
            total = len(test_cases)
            
            self.logger.info(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Test Execution Results:")
            self.logger.info("----------------------")
            
            for i, test in enumerate(test_cases, 1):
                test_start = time.time()
                try:
                    result = func(**test['input'])
                    expected = test['expected']
                    
                    # Handle different types of expected results
                    test_passed = False
                    if isinstance(expected, np.ndarray):
                        test_passed = np.array_equal(result, expected)
                    elif isinstance(expected, list):
                        # For lists of tuples, sort both lists before comparing
                        if expected and isinstance(expected[0], tuple):
                            result = sorted(result)
                            expected = sorted(expected)
                        test_passed = result == expected
                    else:
                        test_passed = result == expected
                    
                    passed += float(test_passed)
                    test_time = time.time() - test_start
                    self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Test {i}:")
                    self.logger.info(f"  Input: {test['input']}")
                    self.logger.info(f"  Expected: {expected}")
                    self.logger.info(f"  Got: {result}")
                    self.logger.info(f"  Status: {'PASS' if test_passed else 'FAIL'}")
                    self.logger.info(f"  Time: {test_time:.3f}s\n")
                        
                except Exception as e:
                    expected = test['expected']
                    result = e
                    test_passed = isinstance(result, type(expected))
                    passed += float(test_passed)
                    test_time = time.time() - test_start
                    self.logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Test {i}:")
                    self.logger.error(f"  Input: {test['input']}")
                    self.logger.error(f"  Expected: {expected}")
                    self.logger.error(f"  Got: {result}")
                    self.logger.error(f"  Status: {'PASS' if test_passed else 'FAIL'}")
                    self.logger.error(f"  Time: {test_time:.3f}s")
                    self.logger.error(f"  Error: {str(e)}\n")
                    
            score = passed / total if total > 0 else 0.0
            self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Overall Test Score: {score:.2f} ({int(passed)}/{total} tests passed)")
            return score
            
        except Exception as e:
            self.logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error in test execution: {str(e)}")
            return 0.0

    def _check_code_style(self, code: str) -> float:
        """Check code style metrics"""
        score = 1.0
        
        # Check docstring
        if not re.search(r'""".*?"""', code, re.DOTALL):
            score *= 0.7
        
        # Check line length
        long_lines = [line for line in code.split('\n') if len(line.strip()) > 80]
        if long_lines:
            score *= 0.9
        
        # Check function length
        if len(code.split('\n')) > 50:
            score *= 0.8
        
        # Check error handling
        if 'try:' not in code or 'except:' not in code:
            score *= 0.8
            
        # Check input validation
        first_lines = '\n'.join(code.split('\n')[:5])
        if not re.search(r'if.*[\<\>\=\!]', first_lines):
            score *= 0.9
            
        return score

    def _estimate_efficiency(self, code: str) -> float:
        """Estimate code efficiency through static analysis"""
        score = 1.0
        
        # Check for nested loops (penalize)
        if re.search(r'for.*\n.*for', code):
            score *= 0.7
        
        # Check for list comprehensions (reward)
        if '[' in code and 'for' in code and ']' in code:
            score *= 1.2
            
        # Check for helper functions (reward)
        if len(re.findall(r'def\s+\w+', code)) > 1:
            score *= 1.2
            
        # Check for early returns (reward)
        if len(re.findall(r'return.*\n.*return', code)) > 0:
            score *= 1.1
            
        # Check for efficient data structures
        if 'collections.' in code:
            score *= 1.1
            
        # Cap at 1.0
        return min(1.0, score)

    def analyze_results(self, results: Dict[float, TaskResult]) -> Dict[str, Any]:
        """Analyze results across time budgets"""
        analysis = {
            'time_budgets': [],
            'performance_scores': [],
            'delegation_depths': [],
            'violation_counts': [],
            'violation_ratio': [],
            'total_times': [],
            'aide_times': [],
            'setup_times': [],
            'cleanup_times': [],
            'time_efficiency': [],  # actual time / budget time ratio
            'implementations': []  # store actual implementations
        }
        
        # Collect basic statistics
        for time_budget, result in sorted(results.items()):
            analysis['time_budgets'].append(time_budget)
            analysis['performance_scores'].append(result.performance_score)
            analysis['delegation_depths'].append(result.delegation_depth)
            analysis['total_times'].append(result.time_taken)
            analysis['aide_times'].append(result.aide_execution_time)
            analysis['setup_times'].append(result.setup_time)
            analysis['cleanup_times'].append(result.cleanup_time)
            analysis['implementations'].append(result.code)
            
            # Calculate time efficiency (actual/budget ratio)
            budget_seconds = time_budget * 60
            time_efficiency = result.time_taken / budget_seconds if budget_seconds > 0 else 1.0
            analysis['time_efficiency'].append(time_efficiency)
            
            violation_count = len(result.constraint_violations)
            analysis['violation_counts'].append(violation_count)
            # Handle division by zero for violation ratio
            if result.delegation_depth > 0:
                ratio = violation_count / result.delegation_depth
            else:
                ratio = 0.0
            analysis['violation_ratio'].append(ratio)
        
        # Convert to numpy arrays for analysis
        time_budgets = np.array(analysis['time_budgets'])
        violations = np.array(analysis['violation_counts'])
        depths = np.array(analysis['delegation_depths'])
        actual_times = np.array(analysis['total_times'])
        
        # Calculate correlations with proper handling of edge cases
        analysis['correlations'] = {}
        
        # Calculate correlations
        analysis['correlations']['time_vs_violations'] = self.safe_correlation(time_budgets, violations)
        analysis['correlations']['depth_vs_violations'] = self.safe_correlation(depths, violations)
        analysis['correlations']['actual_vs_budget'] = self.safe_correlation(actual_times, time_budgets)
        
        # Calculate summary statistics with error handling
        with np.errstate(divide='ignore', invalid='ignore'):
            analysis['summary'] = {
                'mean_performance': float(np.mean(analysis['performance_scores'])),
                'std_performance': float(np.std(analysis['performance_scores'])),
                'mean_violations': float(np.mean(analysis['violation_counts'])),
                'total_violations': int(np.sum(analysis['violation_counts'])),
                'success_rate': float(np.mean([s > 0.8 for s in analysis['performance_scores']])),
                'mean_time_efficiency': float(np.mean(analysis['time_efficiency'])),
                'mean_aide_time': float(np.mean(analysis['aide_times'])),
                'mean_setup_time': float(np.mean(analysis['setup_times'])),
                'mean_cleanup_time': float(np.mean(analysis['cleanup_times']))
            }
        
        return analysis

    def score_code(self, code: str, test_cases: List[Dict]) -> float:
        """Calculate overall code quality score"""
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting code quality evaluation")

        if not code or code.startswith('# Error'):
            self.logger.warning(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Invalid or empty code provided")
            return 0.0
            
        scores = {}
        
        # Run test cases
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Running test cases")
        test_start = time.time()
        scores['test_passing'] = self._run_test_cases(code, test_cases)
        test_time = time.time() - test_start
        
        # Check code style
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Evaluating code style")
        style_start = time.time()
        scores['code_style'] = self._check_code_style(code)
        style_time = time.time() - style_start
        
        # Check efficiency
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Evaluating code efficiency")
        efficiency_start = time.time()
        scores['efficiency'] = self._estimate_efficiency(code)
        efficiency_time = time.time() - efficiency_start

        # Log individual scores and timing
        self.logger.info(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Evaluation Results:")
        self.logger.info(f"  Test Score: {scores['test_passing']:.2f} (took {test_time:.2f}s)")
        self.logger.info(f"  Style Score: {scores['code_style']:.2f} (took {style_time:.2f}s)")
        self.logger.info(f"  Efficiency Score: {scores['efficiency']:.2f} (took {efficiency_time:.2f}s)")
        
        # Calculate final weighted score
        final_score = sum(score * self.metrics[metric] 
                       for metric, score in scores.items())
        
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Final Score: {final_score:.2f}")
        return final_score

    def get_detailed_scores(self, code: str, test_cases: List[Dict]) -> Dict[str, float]:
        """Get detailed scores for each metric"""
        if not code or code.startswith('# Error'):
            return {
                'test_passing': 0.0,
                'code_style': 0.0,
                'efficiency': 0.0
            }
            
        scores = {}
        
        # Run test cases
        scores['test_passing'] = self._run_test_cases(code, test_cases)
        
        # Check code style
        scores['code_style'] = self._check_code_style(code)
        
        # Check efficiency
        scores['efficiency'] = self._estimate_efficiency(code)
        
        return scores

class AIDEAgent:
    """Agent that uses AIDE to implement functions"""
    def __init__(self, workspace: str = "experiment_workspace", model: str = None, task_name: str = None):
        """Initialize agent with workspace path"""
        # Configure logging
        self.logger = logging.getLogger('AIDEAgent')
        self.logger.setLevel(logging.DEBUG)
        
        # Convert workspace to absolute path if it's relative
        self.workspace = str(Path(workspace).resolve())
        self.model = model
        self.task_name = task_name
        
        self.logger.info(f"Initializing AIDEAgent with workspace: {self.workspace}")
        
        # Create workspace if it doesn't exist
        os.makedirs(self.workspace, exist_ok=True)
        
        # Update config file with model if specified
        if model:
            self._update_config_with_model()
            
            if "gemini" in model.lower():
                self._setup_gemini()
            elif "ollama" in model.lower():
                self._setup_ollama()

    def _clean_task_directory(self):
        """Clean up task-specific workspace directory"""
        # Ensure task workspace directory exists
        os.makedirs(self.task_workspace, exist_ok=True)
        self.logger.info(f"Created task workspace directory: {self.task_workspace}")
        
        # Clean up existing files in task directory
        for item in os.listdir(self.task_workspace):
            item_path = os.path.join(self.task_workspace, item)
            try:
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                self.logger.debug(f"Cleaned up: {item_path}")
            except Exception as e:
                self.logger.warning(f"Error while cleaning up {item_path}: {e}")

    def _setup_ollama(self):
        """Setup Ollama configuration and check server"""
        self.logger.info("Setting up Ollama configuration...")
        os.environ['OPENAI_BASE_URL'] = "http://localhost:11434/v1"
        os.environ['OPENAI_API_KEY'] = "local-llm"
        
        try:
            response = requests.get("http://localhost:11434/api/version", timeout=2)
            if response.status_code != 200:
                raise ConnectionError("Ollama server returned error status")
            self.logger.info("Successfully connected to Ollama server")
        except Exception as e:
            self.logger.error(f"Failed to connect to Ollama server: {str(e)}")
            self.logger.error("Please ensure Ollama server is running with: ollama serve")
            raise
            
    def _setup_gemini(self):
        """Setup Gemini configuration"""
        self.logger.info("Setting up Gemini configuration...")
        if not os.getenv('GOOGLE_API_KEY'):
            self.logger.error("GOOGLE_API_KEY environment variable not set")
            raise ValueError("Please set the GOOGLE_API_KEY environment variable")
            
        import google.generativeai as genai
        from google.api_core import retry
        from google.api_core import exceptions as google_exceptions
        
        # Initialize Gemini model with retries
        try:
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            self.logger.info("Successfully initialized and tested Gemini model")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini model: {str(e)}")
            raise

    def _get_aide_params(self, time_budget: float) -> Dict[str, Any]:
        """Get AIDE parameters scaled based on time budget in minutes"""
        # For very short time budgets, use minimal but aggressive settings
        if time_budget <= 1:
            params = {
                "steps": 1,  # Single step
                "k_fold_validation": 1,  # No cross validation
                "max_debug_depth": 1,  # Minimal debug depth
                "debug_prob": 0.5,  # Higher debug probability for short runs
                "num_drafts": 2,  # Two drafts for better quality
                "timeout": max(45, int(time_budget * 60))  # At least 45 seconds
            }
        else:
            # For longer runs, use exponential scaling
            scale = 1 - np.exp(-time_budget / 4)  # Smoother scaling curve
            
            # Calculate timeout with buffer
            timeout = min(int(time_budget * 60 * 1.2), 900)  # Cap at 15 minutes, 20% buffer
            
            params = {
                "steps": max(2, int(8 * scale)),  # 2 to 8 steps
                "k_fold_validation": max(1, int(3 * scale)),  # 1 to 3 folds
                "max_debug_depth": max(2, int(4 * scale)),  # 2 to 4 depth
                "debug_prob": 0.3 + (0.4 * scale),  # 0.3 to 0.7 probability
                "num_drafts": max(2, int(5 * scale)),  # 2 to 5 drafts
                "timeout": timeout
            }
            
        self.logger.info(f"AIDE parameters for {time_budget}m budget: {params}")
        return params

    def implement_function(self, task: CodingTask) -> Tuple[str, Dict[str, float]]:
        """Implement a coding task using AIDE"""
        # Create task directory directly in the workspace
        task_dir = os.path.join(self.workspace, task.function_name)
        os.makedirs(task_dir, exist_ok=True)
        
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Created task directory: {task_dir}")
        
        # Clean up any previous files in this directory
        for item in os.listdir(task_dir):
            item_path = os.path.join(task_dir, item)
            if os.path.isfile(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Cleaned up previous run artifacts")
        
        # Construct task description
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Constructing task description")
        task_desc = f"""## Goal
Implement a Python function named `{task.function_name}`.

## Background
{task.function_description}

## Constraints
{task.constraint}

## Test Cases
The function will be tested with the following cases:
{self._format_test_cases(task.test_cases)}

## Evaluation
{task.evaluation}
"""
        
        # Write task description
        task_md_path = os.path.join(task_dir, "task.md")
        with open(task_md_path, "w") as f:
            f.write(task_desc)
            
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Wrote task description to {task_md_path}")
            
        # Create input directory and copy data
        input_dir = os.path.join(task_dir, "input")
        os.makedirs(input_dir, exist_ok=True)
        
        # Create a dummy input file to satisfy AIDE's requirements
        input_file = os.path.join(input_dir, "input.txt")
        with open(input_file, "w") as f:
            f.write("Function implementation task")
            
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Created input directory and files")
        
        # Set parameters based on time budget
        params = self._get_aide_params(task.time_budget)
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Using AIDE parameters: {params}")
        
        # Run AIDE
        start_time = time.time()
        setup_start = time.time()
        
        try:
            # Store original directory
            original_dir = os.getcwd()
            self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Original working directory: {original_dir}")
            
            # Change to task directory
            os.chdir(task_dir)
            self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Changed working directory to: {task_dir}")
            
            setup_time = time.time() - setup_start
            aide_start = time.time()
            
            # Run AIDE command
            try:
                # Construct command
                cmd = ["aide"]
                
                # Add parameters
                cmd.extend([
                    f"data_dir={os.path.join(task_dir, 'input')}",
                    f"desc_file={os.path.join(task_dir, 'task.md')}",
                    f"log_dir={os.path.join(task_dir, 'logs')}",
                    f"workspace_dir={os.path.join(task_dir, 'workspaces')}",
                    f"exp_name={task.function_name}_exp",
                    f"exec.timeout={params['timeout']}",
                    f"exec.agent_file_name=solution.py",
                    f"agent.steps={params['steps']}",
                    f"agent.k_fold_validation={params['k_fold_validation']}",
                    "agent.expose_prediction=true",
                    "agent.data_preview=true",
                    "report.temp=1.0",
                    "agent.code.temp=0.5",
                    "agent.feedback.temp=0.5",
                    f"agent.search.max_debug_depth={params['max_debug_depth']}",
                    f"agent.search.debug_prob={params['debug_prob']}",
                    f"agent.search.num_drafts={params['num_drafts']}",
                    "preprocess_data=true",
                    "copy_data=true",
                    "generate_report=true"
                ])
                
                # Add model parameters if specified
                if self.model:
                    cmd.extend([
                        f"agent.code.model={self.model}",
                        f"agent.feedback.model={self.model}",
                        f"report.model={self.model}"
                    ])
                
                cmd_str = " ".join(cmd)
                self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Running command: {cmd_str}")
                
                # Create process
                process = subprocess.Popen(
                    cmd_str,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Started AIDE process with PID: {process.pid}")
                
                # Track time since last output
                last_output_time = time.time()
                start_time = time.time()
                
                # Read output
                while True:
                    # Check if process has finished
                    if process.poll() is not None:
                        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] AIDE process completed with return code: {process.returncode}")
                        break
                        
                    # Read output with timeout
                    try:
                        line = process.stdout.readline()
                        if line:
                            self.logger.info(f"AIDE: {line.strip()}")
                            last_output_time = time.time()
                        else:
                            # No output available, sleep briefly
                            time.sleep(0.1)
                            
                            # Check for timeout
                            elapsed = time.time() - start_time
                            quiet_time = time.time() - last_output_time
                            
                            if elapsed >= params['timeout']:
                                self.logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] AIDE process exceeded timeout of {params['timeout']}s")
                                # Send SIGTERM first
                                process.terminate()
                                time.sleep(1)
                                if process.poll() is None:
                                    # If still running, force kill
                                    self.logger.warning(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Process did not terminate gracefully, forcing kill")
                                    process.kill()
                                break
                            elif quiet_time >= 10:
                                self.logger.warning(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] No output from AIDE process for {quiet_time:.1f}s (elapsed: {elapsed:.1f}s)")
                                # Check process state
                                try:
                                    os.kill(process.pid, 0)
                                    self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Process is still running")
                                except OSError:
                                    self.logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Process appears to be dead")
                                    break
                    except Exception as e:
                        self.logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error reading AIDE output: {str(e)}")
                        break
                
                # Clean up process
                if process.poll() is None:
                    self.logger.warning(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Cleaning up AIDE process")
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                
                # Look for solution file
                solution_paths = [
                    os.path.join(task_dir, "logs", "0-" + task.function_name + "_exp", "best_solution.py"),
                    os.path.join(task_dir, "workspaces", "0-" + task.function_name + "_exp", "solution.py")
                ]
                
                solution_code = None
                for path in solution_paths:
                    if os.path.exists(path):
                        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Found solution at: {path}")
                        with open(path, 'r') as f:
                            solution_code = f.read()
                        break
                
                if not solution_code:
                    self.logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] No solution file found")
                    return "# Error: No solution generated", {
                        'total_time': time.time() - start_time,
                        'aide_execution_time': time.time() - aide_start,
                        'setup_time': setup_time,
                        'cleanup_time': 0
                    }
                
                return solution_code, {
                    'total_time': time.time() - start_time,
                    'aide_execution_time': time.time() - aide_start,
                    'setup_time': setup_time,
                    'cleanup_time': 0
                }
                
            except Exception as e:
                self.logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error running AIDE: {str(e)}")
                return f"# Error: {str(e)}", {
                    'total_time': time.time() - start_time,
                    'aide_execution_time': time.time() - aide_start,
                    'setup_time': setup_time,
                    'cleanup_time': 0
                }
            
        finally:
            # Always restore original directory
            try:
                os.chdir(original_dir)
                self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Restored working directory to: {original_dir}")
            except Exception as e:
                self.logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error restoring working directory: {str(e)}")

    def _update_config_with_model(self):
        """Update the AIDE config file with the specified model"""
        try:
            # Find the config file
            config_path = Path(__file__).parent.parent.parent / "aideml" / "aide" / "utils" / "config.yaml"
            if not config_path.exists():
                self.logger.warning(f"Config file not found at {config_path}")
                return
                
            # Read current config
            with open(config_path, 'r') as f:
                config_content = f.read()
                
            # Update model names using regex
            import re
            patterns = [
                (r'(report:\n\s+model:)\s+.*$', f'\\1 {self.model}'),
                (r'(code:\n\s+model:)\s+.*$', f'\\1 {self.model}'),
                (r'(feedback:\n\s+model:)\s+.*$', f'\\1 {self.model}')
            ]
            
            updated_content = config_content
            for pattern, replacement in patterns:
                updated_content = re.sub(pattern, replacement, updated_content, flags=re.MULTILINE)
                
            # Write updated config if changes were made
            if updated_content != config_content:
                with open(config_path, 'w') as f:
                    f.write(updated_content)
                self.logger.info(f"Updated config file with model: {self.model}")
            
        except Exception as e:
            self.logger.error(f"Error updating config file: {str(e)}")

    def _kill_process_tree(self, process):
        """Kill a process and all its children."""
        try:
            import psutil
            parent = psutil.Process(process.pid)
            children = parent.children(recursive=True)
            
            # Send SIGTERM to children first
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
                    
            # Then terminate parent    
            parent.terminate()
            
            # Wait for processes to terminate (2s timeout)
            gone, alive = psutil.wait_procs(children + [parent], timeout=2)
            
            # Force kill any remaining processes
            for p in alive:
                try:
                    p.kill()
                except psutil.NoSuchProcess:
                    pass
                    
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired) as e:
            self.logger.warning(f"Error killing process tree: {e}")
            # Fallback to basic process termination
            try:
                process.terminate()
                process.wait(timeout=2)
            except:
                process.kill()
                
    def _format_test_cases(self, test_cases: List[Dict[str, Any]]) -> str:
        """Format test cases for the task description"""
        formatted = []
        for i, test in enumerate(test_cases, 1):
            inputs = test['input']
            expected = test['expected']
            
            # Format expected output
            if isinstance(expected, Exception):
                expected_str = "Should raise an exception"
            else:
                expected_str = str(expected)
            
            formatted.append(f"{i}. Input: {inputs}")
            formatted.append(f"   Expected: {expected_str}\n")

class Evaluator:
    """Evaluates coding task results"""
    def __init__(self):
        # Initialize with empty constraint list
        self.constraint_checker = ConstraintChecker([])
        self.quality_scorer = CodeQualityScorer()
        self.logger = logging.getLogger('Evaluator')
        self.logger.setLevel(logging.DEBUG)
        
    def add_constraint(self, constraint: Constraint) -> None:
        """Add a new constraint to check"""
        self.constraint_checker.add_constraint(constraint)
        
    def evaluate_task(self, task: CodingTask, code: str, time_taken: float,
                    aide_time: float, setup_time: float, cleanup_time: float) -> TaskResult:
        """Evaluate a coding task implementation"""
        # Get detailed scores first
        scores = self.quality_scorer.get_detailed_scores(code, task.test_cases)
        
        # Check constraint violations
        violations = self.constraint_checker.check_violations(code)
        
        # Enforce specific delegation depths based on time budget
        if task.time_budget <= 0.5:
            delegation_depth = 0  # No delegation for quick tasks
        elif task.time_budget <= 2.0:
            delegation_depth = 1  # Single delegation for medium tasks
        else:
            # For longer tasks (8m), scale delegation with complexity
            complexity_score = 0
            
            # Structure complexity - only count significant factors
            if len(re.findall(r'def\s+\w+', code)) > 1:  # Has helper functions
                complexity_score += 1
            if 'while' in code:  # Has while loops
                complexity_score += 0.5
            if any('O(n)' in line or 'O(log n)' in line for line in code.split('\n')):
                complexity_score += 0.5  # Documents time complexity
                
            # For 8m tasks, start at 2 and add complexity score
            delegation_depth = 2 + int(complexity_score)
            delegation_depth = min(delegation_depth, 4)  # Cap at 4
        
        return TaskResult(
            code=code,
            time_taken=time_taken,
            aide_execution_time=aide_time,
            setup_time=setup_time,
            cleanup_time=cleanup_time,
            delegation_depth=delegation_depth,
            constraint_violations=violations,
            performance_score=scores.get('test_passing', 0.0)  # Use test passing score as performance score
        )

class ExperimentRunner:
    """Runs experiments with multiple tasks and time budgets"""
    def __init__(self, workspace: str = "experiment_workspace", model: str = None):
        """Initialize runner with workspace path"""
        self.workspace = workspace
        self.model = model
        self.logger = logging.getLogger('ExperimentRunner')
        self.evaluator = Evaluator()
        
        # Create timestamped results file
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.results_file = os.path.join(workspace, f"experiment_results_{timestamp}.csv")
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Results will be saved to: {self.results_file}")
        
        # Write CSV header
        with open(self.results_file, 'w') as f:
            f.write("timestamp,task_name,time_budget,run_number,total_time,aide_time,setup_time,cleanup_time,"
                   "test_passing_score,code_style_score,efficiency_score,final_score,"
                   "delegation_depth,num_violations,violations\n")

    def _save_run_results(self, task: CodingTask, run_num: int, timing_stats: Dict[str, float], 
                         result: TaskResult):
        """Save individual run results to CSV file"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Get detailed scores from quality scorer
        scores = self.evaluator.quality_scorer.get_detailed_scores(result.code, task.test_cases)
        
        # Format violations as a semicolon-separated string, ensuring it's properly quoted
        violations_str = ''
        if result.constraint_violations:
            violations_str = ';'.join(str(v) for v in result.constraint_violations)
        violations_str = f'"{violations_str}"' if violations_str else '""'
        
        # Prepare row data
        row = [
            timestamp,
            task.function_name,
            task.time_budget,
            run_num,
            timing_stats['total_time'],
            timing_stats['aide_execution_time'],
            timing_stats['setup_time'],
            timing_stats['cleanup_time'],
            scores.get('test_passing', 0.0),
            scores.get('code_style', 0.0),
            scores.get('efficiency', 0.0),
            result.performance_score,
            result.delegation_depth,
            len(result.constraint_violations),
            violations_str  # Now properly formatted
        ]
        
        # Write to CSV
        with open(self.results_file, 'a') as f:
            f.write(','.join(map(str, row)) + '\n')
        
        self.logger.info(f"[{timestamp}] Saved results for {task.function_name} run {run_num}")

    def safe_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate correlation with proper handling of edge cases"""
        if len(x) < 2 or len(y) < 2:
            return 0.0
        if np.all(x == x[0]) or np.all(y == y[0]):
            return 0.0
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = np.corrcoef(x, y)[0, 1]
            return 0.0 if np.isnan(corr) else corr

    def print_analysis(self, analysis: Dict[str, Any]) -> None:
        """Print analysis results in a readable format"""
        self.logger.info("\nAnalysis Results:")
        self.logger.info("=================\n")
        self.logger.info("Detailed Results by Time Budget:\n")
        
        # Print per-time-budget statistics
        for time_budget, stats in analysis['by_time_budget'].items():
            self.logger.info(f"Time Budget: {time_budget} minutes")
            self.logger.info(f"Performance Score: {stats['performance_score']['mean']:.2f} ± {stats['performance_score']['std']:.2f}")
            self.logger.info(f"Total Time: {stats['total_time']['mean']:.1f}s ± {stats['total_time']['std']:.1f}s")
            self.logger.info(f"AIDE Time: {stats['aide_time']['mean']:.1f}s ± {stats['aide_time']['std']:.1f}s")
            self.logger.info(f"Delegation Depth: {stats['delegation_depth']['mean']:.1f} ± {stats['delegation_depth']['std']:.1f}")
            self.logger.info(f"Constraint Violations: {stats['violations']['mean']:.1f} ± {stats['violations']['std']:.1f}")
            self.logger.info(f"Number of Runs: {stats['num_runs']}\n")

        # Print overall statistics
        self.logger.info("Overall Statistics:")
        self.logger.info(f"Total Runs: {analysis['overall']['total_runs']}")
        self.logger.info(f"Mean Performance Score: {analysis['overall']['performance_score']['mean']:.2f} ± {analysis['overall']['performance_score']['std']:.2f}")
        self.logger.info(f"Mean Total Time: {analysis['overall']['total_time']['mean']:.1f}s ± {analysis['overall']['total_time']['std']:.1f}s")
        self.logger.info(f"Mean AIDE Time: {analysis['overall']['aide_time']['mean']:.1f}s ± {analysis['overall']['aide_time']['std']:.1f}s")
        self.logger.info(f"Mean Delegation Depth: {analysis['overall']['delegation_depth']['mean']:.2f} ± {analysis['overall']['delegation_depth']['std']:.2f}")
        self.logger.info(f"Mean Violations: {analysis['overall']['violations']['mean']:.2f} ± {analysis['overall']['violations']['std']:.2f}\n")
        
        # Print correlations
        self.logger.info("Correlations:")
        for metric, value in analysis['correlations'].items():
            self.logger.info(f"  {metric}: {value:.3f}")
        self.logger.info("")

    def analyze_results(self, results: Dict[float, List[TaskResult]]) -> Dict[str, Any]:
        """Analyze experiment results and generate statistics
        
        Args:
            results: Dictionary mapping time budgets to lists of TaskResults
            
        Returns:
            Dictionary containing analysis results and statistics
        """
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Analyzing experiment results")
        
        analysis = {
            'by_time_budget': {},
            'overall': {},
            'correlations': {}
        }
        
        # Collect all metrics across all time budgets
        all_perf_scores = []
        all_total_times = []
        all_aide_times = []
        all_depths = []
        all_violations = []
        all_time_budgets = []
        
        for time_budget, budget_results in results.items():
            # Extract metrics for this time budget
            perf_scores = [r.performance_score for r in budget_results]
            total_times = [r.time_taken for r in budget_results]
            aide_times = [r.aide_execution_time for r in budget_results]
            depths = [r.delegation_depth for r in budget_results]
            violations = [len(r.constraint_violations) for r in budget_results]
            
            # Store metrics for correlation analysis
            all_perf_scores.extend(perf_scores)
            all_total_times.extend(total_times)
            all_aide_times.extend(aide_times)
            all_depths.extend(depths)
            all_violations.extend(violations)
            all_time_budgets.extend([time_budget] * len(budget_results))
            
            # Calculate statistics for this time budget
            budget_stats = {
                'performance_score': {
                    'mean': float(np.mean(perf_scores)),
                    'std': float(np.std(perf_scores)),
                    'min': float(np.min(perf_scores)),
                    'max': float(np.max(perf_scores))
                },
                'total_time': {
                    'mean': float(np.mean(total_times)),
                    'std': float(np.std(total_times)),
                    'min': float(np.min(total_times)),
                    'max': float(np.max(total_times))
                },
                'aide_time': {
                    'mean': float(np.mean(aide_times)),
                    'std': float(np.std(aide_times)),
                    'min': float(np.min(aide_times)),
                    'max': float(np.max(aide_times))
                },
                'delegation_depth': {
                    'mean': float(np.mean(depths)),
                    'std': float(np.std(depths)),
                    'min': float(np.min(depths)),
                    'max': float(np.max(depths))
                },
                'violations': {
                    'mean': float(np.mean(violations)),
                    'std': float(np.std(violations)),
                    'min': float(np.min(violations)),
                    'max': float(np.max(violations))
                },
                'num_runs': len(budget_results)
            }
            
            analysis['by_time_budget'][time_budget] = budget_stats
            
            # Log statistics for this time budget
            self.logger.info(f"\nTime Budget {time_budget}m Statistics:")
            self.logger.info(f"  Performance Score: {budget_stats['performance_score']['mean']:.2f} ± {budget_stats['performance_score']['std']:.2f}")
            self.logger.info(f"  Total Time: {budget_stats['total_time']['mean']:.1f}s ± {budget_stats['total_time']['std']:.1f}s")
            self.logger.info(f"  AIDE Time: {budget_stats['aide_time']['mean']:.1f}s ± {budget_stats['aide_time']['std']:.1f}s")
            self.logger.info(f"  Delegation Depth: {budget_stats['delegation_depth']['mean']:.1f} ± {budget_stats['delegation_depth']['std']:.1f}")
            self.logger.info(f"  Violations: {budget_stats['violations']['mean']:.1f} ± {budget_stats['violations']['std']:.1f}")
        
        # Calculate overall statistics
        analysis['overall'] = {
            'performance_score': {
                'mean': float(np.mean(all_perf_scores)),
                'std': float(np.std(all_perf_scores))
            },
            'total_time': {
                'mean': float(np.mean(all_total_times)),
                'std': float(np.std(all_total_times))
            },
            'aide_time': {
                'mean': float(np.mean(all_aide_times)),
                'std': float(np.std(all_aide_times))
            },
            'delegation_depth': {
                'mean': float(np.mean(all_depths)),
                'std': float(np.std(all_depths))
            },
            'violations': {
                'mean': float(np.mean(all_violations)),
                'std': float(np.std(all_violations))
            },
            'total_runs': len(all_perf_scores)
        }
        
        # Calculate correlations
        analysis['correlations'] = {
            'time_budget_vs_performance': self.safe_correlation(np.array(all_time_budgets), np.array(all_perf_scores)),
            'time_budget_vs_total_time': self.safe_correlation(np.array(all_time_budgets), np.array(all_total_times)),
            'time_budget_vs_aide_time': self.safe_correlation(np.array(all_time_budgets), np.array(all_aide_times)),
            'time_budget_vs_depth': self.safe_correlation(np.array(all_time_budgets), np.array(all_depths)),
            'time_budget_vs_violations': self.safe_correlation(np.array(all_time_budgets), np.array(all_violations)),
            'performance_vs_depth': self.safe_correlation(np.array(all_perf_scores), np.array(all_depths)),
            'performance_vs_violations': self.safe_correlation(np.array(all_perf_scores), np.array(all_violations))
        }
        
        # Log overall statistics
        self.logger.info("\nOverall Statistics:")
        self.logger.info(f"  Total Runs: {analysis['overall']['total_runs']}")
        self.logger.info(f"  Average Performance: {analysis['overall']['performance_score']['mean']:.2f} ± {analysis['overall']['performance_score']['std']:.2f}")
        self.logger.info(f"  Average Total Time: {analysis['overall']['total_time']['mean']:.1f}s ± {analysis['overall']['total_time']['std']:.1f}s")
        self.logger.info(f"  Average AIDE Time: {analysis['overall']['aide_time']['mean']:.1f}s ± {analysis['overall']['aide_time']['std']:.1f}s")
        
        # Log correlations
        self.logger.info("\nCorrelations:")
        for metric, value in analysis['correlations'].items():
            self.logger.info(f"  {metric}: {value:.3f}")
            
        return analysis

    def run_experiment(self, tasks: List[CodingTask], num_runs: int = 1) -> Dict[float, List[TaskResult]]:
        """Run experiment with multiple runs per time budget"""
        model = self.model
        
        # Group tasks by time budget
        tasks_by_budget = {}
        for task in tasks:
            if task.time_budget not in tasks_by_budget:
                tasks_by_budget[task.time_budget] = []
            tasks_by_budget[task.time_budget].append(task)
            
        results = {}
        total_tasks = sum(len(tasks) for tasks in tasks_by_budget.values())
        task_count = 0
        
        for time_budget, budget_tasks in tasks_by_budget.items():
            self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Running {num_runs} tasks with {time_budget:.1f} minute budget")
            
            # Store results for all runs with this time budget
            budget_results = []
            
            for i, task in enumerate(budget_tasks):
                task_count += 1
                self.logger.info(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Task {task_count}/{total_tasks} - Run {i+1}/{num_runs}")
                self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Function: {task.function_name}")
                
                # Set up AIDE agent with specified model and task name
                self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Initializing AIDE agent with model: {model}")
                self.agent = AIDEAgent(workspace=self.workspace, model=model, task_name=task.function_name)
                
                # Generate implementation
                self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting function implementation")
                implementation_start = time.time()
                code, timing_stats = self.agent.implement_function(task)
                implementation_time = time.time() - implementation_start
                self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Implementation completed in {implementation_time:.2f}s")
                
                # Evaluate the implementation
                self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting implementation evaluation")
                evaluation_start = time.time()
                result = self.evaluator.evaluate_task(
                    task=task,
                    code=code,
                    time_taken=timing_stats['total_time'],
                    aide_time=timing_stats['aide_execution_time'],
                    setup_time=timing_stats['setup_time'],
                    cleanup_time=timing_stats['cleanup_time']
                )
                evaluation_time = time.time() - evaluation_start
                self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Evaluation completed in {evaluation_time:.2f}s")
                
                # Save results to CSV
                self._save_run_results(task, i+1, timing_stats, result)
                
                budget_results.append(result)
                
                # Log immediate results for this run
                self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Run Statistics:")
                self.logger.info(f"  - Total time: {timing_stats['total_time']:.1f}s")
                self.logger.info(f"  - AIDE execution: {timing_stats['aide_execution_time']:.1f}s")
                self.logger.info(f"  - Setup time: {timing_stats['setup_time']:.1f}s")
                self.logger.info(f"  - Cleanup time: {timing_stats['cleanup_time']:.1f}s")
                self.logger.info(f"  - Delegation depth: {result.delegation_depth}")
                self.logger.info(f"  - Performance score: {result.performance_score:.2f}")
                
                if result.constraint_violations:
                    self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Constraint violations:")
                    for violation in result.constraint_violations:
                        self.logger.info(f"  - {violation}")
            
            # Store results for this time budget
            results[time_budget] = budget_results
            
            # Calculate statistics for this time budget
            perf_scores = [r.performance_score for r in budget_results]
            total_times = [r.time_taken for r in budget_results]
            aide_times = [r.aide_execution_time for r in budget_results]
            depths = [r.delegation_depth for r in budget_results]
            violations = [len(r.constraint_violations) for r in budget_results]
            
            # Log statistics for this time budget
            self.logger.info(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Time Budget {time_budget}m Summary:")
            self.logger.info(f"  Performance Score: {np.mean(perf_scores):.2f} ± {np.std(perf_scores):.2f}")
            self.logger.info(f"  Total Time: {np.mean(total_times):.1f}s ± {np.std(total_times):.1f}s")
            self.logger.info(f"  AIDE Time: {np.mean(aide_times):.1f}s ± {np.std(aide_times):.1f}s")
            self.logger.info(f"  Delegation Depth: {np.mean(depths):.1f} ± {np.std(depths):.1f}")
            self.logger.info(f"  Violations: {np.mean(violations):.1f} ± {np.std(violations):.1f}")
            
        return results

def parse_markdown_task(md_file_path):
    """Parse task definition from markdown file"""
    with open(md_file_path, 'r') as f:
        content = f.read()
    
    # Extract sections using regex
    function_name = re.search(r'## Function Name\s+(.+?)\s*(?=##|\Z)', content, re.DOTALL).group(1).strip()
    description = re.search(r'## Description\s+(.+?)\s*(?=##|\Z)', content, re.DOTALL).group(1).strip()
    constraints = re.search(r'## Constraints\s+(.+?)\s*(?=##|\Z)', content, re.DOTALL).group(1).strip()
    
    # Extract evaluation section if it exists, otherwise use default
    evaluation_match = re.search(r'## Evaluation\s+(.+?)\s*(?=##|\Z)', content, re.DOTALL)
    evaluation = evaluation_match.group(1).strip() if evaluation_match else """The implementation will be evaluated based on:
1. Test case success rate (60%)
2. Code style and readability (20%)
3. Implementation efficiency (20%)

The function should handle edge cases gracefully and include proper error handling."""
    
    # Extract and parse test cases
    test_cases_match = re.search(r'## Test Cases\s+```python\s+(.+?)```', content, re.DOTALL)
    test_cases = ast.literal_eval(test_cases_match.group(1).strip())
    
    # Extract and parse constraint setup
    constraint_setup_match = re.search(r'## Constraint Setup\s+```python\s+(.+?)```', content, re.DOTALL)
    constraint_setup = ast.literal_eval(constraint_setup_match.group(1).strip())
    
    return {
        'function_name': function_name,
        'description': description,
        'constraints': constraints,
        'evaluation': evaluation,
        'test_cases': test_cases,
        'constraint_setup': constraint_setup
    }