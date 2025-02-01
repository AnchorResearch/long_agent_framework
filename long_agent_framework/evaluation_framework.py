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

@dataclass
class CodingTask:
    """Defines a coding task with constraints"""
    function_name: str
    function_description: str
    constraint: str
    time_budget: float  # in minutes
    test_cases: List[Dict[str, Any]]

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
            
            # Execute the function code
            exec(code, namespace)
            
            # Get function object
            match = re.search(r'def\s+(\w+)', code)
            if not match:
                self.logger.error("Could not find function definition in code")
                return 0.0
                
            func_name = match.group(1)
            if func_name not in namespace:
                self.logger.error(f"Function {func_name} not found in namespace")
                return 0.0
                
            func = namespace[func_name]
            
            # Run tests
            passed = 0
            total = len(test_cases)
            
            self.logger.info("\nTest Execution Results:")
            self.logger.info("----------------------")
            
            for i, test in enumerate(test_cases, 1):
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
                    self.logger.info(f"Test {i}:")
                    self.logger.info(f"  Input: {test['input']}")
                    self.logger.info(f"  Expected: {expected}")
                    self.logger.info(f"  Got: {result}")
                    self.logger.info(f"  Status: {'PASS' if test_passed else 'FAIL'}\n")
                        
                except Exception as e:
                    expected = test['expected']
                    result = e
                    test_passed = isinstance(result, type(expected))
                    passed += float(test_passed)
                    self.logger.error(f"Test {i}:")
                    self.logger.error(f"  Input: {test['input']}")
                    self.logger.error(f"  Expected: {expected}")
                    self.logger.error(f"  Got: {result}")
                    self.logger.error(f"  Status: {'PASS' if test_passed else 'FAIL'}\n")
                    self.logger.error(f"  Error: {str(e)}\n")
                    
            score = passed / total if total > 0 else 0.0
            self.logger.info(f"Overall Test Score: {score:.2f} ({int(passed)}/{total} tests passed)")
            return score
            
        except Exception as e:
            self.logger.error(f"Error in test execution: {str(e)}")
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

        print("Code:")
        print(code)

        if not code or code.startswith('# Error'):
            return 0.0
            
        scores = {
            'test_passing': self._run_test_cases(code, test_cases),
            'code_style': self._check_code_style(code),
            'efficiency': self._estimate_efficiency(code)
        }

        print("Scores:")
        print(scores)
        
        # Log individual scores for debugging
        self.logger.info("Individual scores:")
        for metric, score in scores.items():
            self.logger.info(f"  {metric}: {score:.2f}")
        
        return sum(score * self.metrics[metric] 
                  for metric, score in scores.items())

class AIDEAgent:
    """Agent that uses AIDE for code generation"""
    def __init__(self, workspace: str = "experiment_workspace", model: str = None):
        if model is None:
            raise ValueError("Model parameter is required")
            
        self.workspace = Path(workspace).resolve()  # Get absolute path
        self.workspace.mkdir(exist_ok=True)
        self.model = model
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('AIDEAgent')
        
        # Update config file with specified model
        self._update_config_with_model()
        
        # Create AIDE workspace structure
        self.task_dir = self.workspace / "tasks"
        self.task_dir.mkdir(exist_ok=True)
        
        # Cache for successful solutions
        self.solution_cache = {}
        
        # Set up environment variables for AIDE
        os.environ['PYTHONPATH'] = str(Path(__file__).parent.parent)  # Set to project root
        os.environ['OC_CAUSE'] = '1'  # Show full stack traces for config errors

        # Configure based on model type
        if 'llama' in model.lower():
            self._setup_ollama()
        elif 'gemini' in model.lower():
            self._setup_gemini()
            
    def _setup_ollama(self):
        """Setup Ollama configuration and check server"""
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
        if not os.getenv('GOOGLE_API_KEY'):
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

    def _clean_directories(self):
        """Clean up workspace directories"""
        import shutil
        # Clean up task directory contents but keep the directory
        if self.task_dir.exists():
            for item in self.task_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)

    def _get_aide_params(self, time_budget: float) -> Dict[str, Any]:
        """Get AIDE parameters scaled based on time budget in minutes"""
        # For very short time budgets, use minimal settings
        if time_budget <= 1:
            params = {
                "steps": 1,  # Single step
                "k_fold_validation": 1,  # No cross validation
                "max_debug_depth": 1,  # Minimal debug depth
                "debug_prob": 0.1,  # Low debug probability
                "num_drafts": 1,  # Single draft
                "timeout": max(30, int(time_budget * 60))  # At least 30 seconds
            }
        else:
            # For longer runs, use exponential scaling
            scale = 1 - np.exp(-time_budget / 4)  # Smoother scaling curve
            
            # Calculate timeout with buffer
            timeout = min(int(time_budget * 60 * 1.5), 900)  # Cap at 15 minutes, but give 50% buffer
            
            params = {
                "steps": max(1, int(8 * scale)),  # 1 to 8 steps
                "k_fold_validation": max(1, int(3 * scale)),  # 1 to 3 folds
                "max_debug_depth": max(1, int(4 * scale)),  # 1 to 4 depth
                "debug_prob": 0. + (0.5 * scale),  # 0. to 0.5 probability
                "num_drafts": max(1, int(5 * scale)),  # 1 to 5 drafts
                "timeout": timeout
            }
            
        self.logger.info(f"AIDE parameters for {time_budget}m budget: {params}")
        return params

    def implement_function(self, task: CodingTask) -> Tuple[str, Dict[str, float]]:
        """Implement a coding task using AIDE
        
        Args:
            task: The coding task to implement
            
        Returns:
            Tuple of (code, timing_stats)
        """
        # Create task directory
        task_dir = self.task_dir / task.function_name
        task_dir.mkdir(exist_ok=True, parents=True)
        
        # Clean up any previous runs
        for item in task_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        
        # Construct task description
        task_desc = f"""## Goal
Implement a Python function named `{task.function_name}` that analyzes time series data.

## Background
{task.function_description}

## Constraints
{task.constraint}

## Evaluation
The implementation will be evaluated based on:
1. Test case success rate (60%)
2. Code style and readability (20%)
3. Implementation efficiency (20%)

The function should handle edge cases gracefully and include proper error handling.
"""
        
        # Write task description
        with open(task_dir / "task.md", "w") as f:
            f.write(task_desc)
            
        # Create input directory and copy data
        input_dir = task_dir / "input"
        input_dir.mkdir(exist_ok=True)
        
        # Create a dummy input file to satisfy AIDE's requirements
        (input_dir / "input.txt").write_text("Function implementation task")
            
        # Set parameters based on time budget
        params = self._get_aide_params(task.time_budget)  # Use the correct method name
        
        # Run AIDE
        start_time = time.time()
        setup_start = time.time()
        
        try:
            # Store original directory
            original_dir = Path.cwd()
            
            # Change to task directory
            os.chdir(task_dir)
            
            setup_time = time.time() - setup_start
            aide_start = time.time()
            
            # Run AIDE command
            try:
                # Base command list
                cmd = [
                    "aide",
                    f"data_dir=input",  # Relative to task directory
                    f"desc_file=task.md",  # Just the filename
                    f"log_dir=logs",  # Relative to task directory
                    f"workspace_dir=workspaces",  # Relative to task directory
                    f"exp_name={task.function_name}_exp",
                    f"exec.timeout={params['timeout']}",
                    "exec.agent_file_name=solution.py",
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
                ]

                # Add model-specific parameters - just use the model name without constructor
                cmd.extend([
                    f"agent.code.model={self.model}",
                    f"agent.feedback.model={self.model}",
                    f"report.model={self.model}"
                ])

                self.logger.info(f"Running command: {' '.join(cmd)}")
                self.logger.info(f"Working directory: {os.getcwd()}")
                
                # Run AIDE process with real-time output monitoring
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    bufsize=1,  # Line buffered
                    env={
                        **os.environ,
                        'PYTHONUNBUFFERED': '1',  # Force unbuffered output
                        'AIDE_DEBUG': '1'  # Enable debug logging
                    }
                )

                # Monitor process output in real-time
                import select
                import fcntl
                import errno

                # Set non-blocking mode for pipes
                for pipe in [process.stdout, process.stderr]:
                    flags = fcntl.fcntl(pipe.fileno(), fcntl.F_GETFL)
                    fcntl.fcntl(pipe.fileno(), fcntl.F_SETFL, flags | os.O_NONBLOCK)

                poller = select.poll()
                poller.register(process.stdout, select.POLLIN | select.POLLHUP)
                poller.register(process.stderr, select.POLLIN | select.POLLHUP)

                # Track when we last saw output
                last_output_time = time.time()
                timeout = params['timeout']
                start_time = time.time()
                self.logger.info(f"Starting AIDE process with {timeout}s timeout")

                # Track if we've seen initial output
                seen_initial_output = False
                stall_threshold = timeout

                while True:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    
                    # Check if process has finished
                    poll_result = process.poll()
                    if poll_result is not None:
                        self.logger.info(f"AIDE process completed with return code {poll_result} after {elapsed:.1f}s")
                        break
                        
                    # Check for timeout
                    if current_time - start_time > timeout:
                        self.logger.error(f"AIDE process exceeded timeout of {timeout}s")
                        self.logger.info("Sending SIGTERM to process tree")
                        self._kill_process_tree(process)
                        break
                        
                    # Check for stall
                    time_since_output = current_time - last_output_time
                    if time_since_output > 10:
                        if not seen_initial_output:
                            self.logger.warning(f"No initial output from AIDE process after {elapsed:.1f}s")
                        else:
                            self.logger.warning(f"No output from AIDE process for {time_since_output:.1f}s (elapsed: {elapsed:.1f}s)")
                            
                        if time_since_output > stall_threshold:
                            self.logger.error(f"Process appears to be stalled after {elapsed:.1f}s, killing process tree")
                            self._kill_process_tree(process)
                            break
                            
                    # Poll for new output (wait up to 5 seconds)
                    events = poller.poll(5000)
                    for fd, event in events:
                        # Handle process termination
                        if event & select.POLLHUP:
                            self.logger.info("AIDE process pipe closed")
                            continue
                            
                        # Read from appropriate pipe
                        if fd == process.stdout.fileno():
                            try:
                                line = process.stdout.readline()
                                if line:
                                    self.logger.info(f"AIDE: {line.strip()}")
                                    last_output_time = current_time
                                    seen_initial_output = True
                            except IOError as e:
                                if e.errno != errno.EAGAIN:
                                    self.logger.error(f"Error reading stdout: {e}")
                                    raise
                        elif fd == process.stderr.fileno():
                            try:
                                line = process.stderr.readline()
                                if line:
                                    self.logger.error(f"AIDE error: {line.strip()}")
                                    last_output_time = current_time
                                    seen_initial_output = True
                            except IOError as e:
                                if e.errno != errno.EAGAIN:
                                    self.logger.error(f"Error reading stderr: {e}")
                                    raise

                # Get any remaining output
                try:
                    stdout, stderr = process.communicate(timeout=5)  # Increased timeout for final output
                    if stdout:
                        self.logger.info(f"Final AIDE output: {stdout.strip()}")
                    if stderr:
                        self.logger.error(f"Final AIDE errors: {stderr.strip()}")
                except subprocess.TimeoutExpired:
                    self.logger.error("Timed out waiting for final output")
                    process.kill()
                    stdout, stderr = process.communicate()
                
                if process.returncode != 0:
                    self.logger.error(f"AIDE process failed with return code: {process.returncode}")
                    # Try to get the solution file from either logs or workspaces
                    solution_file = Path("logs") / f"0-{task.function_name}_exp" / "best_solution.py"
                    if not solution_file.exists():
                        self.logger.info("Best solution not found in logs, checking workspaces...")
                        solution_file = Path("workspaces") / f"0-{task.function_name}_exp" / "solution.py"
                    
                    if solution_file.exists():
                        with open(solution_file) as f:
                            code = f.read()
                            self.logger.info(f"Found solution at: {solution_file}")
                    else:
                        self.logger.error("Solution file not found in logs or workspaces")
                        code = ""
                    return code, {
                        'total_time': time.time() - start_time,
                        'aide_execution_time': time.time() - aide_start,
                        'setup_time': setup_time,
                        'cleanup_time': 0.0
                    }
                    
                # Get solution code
                solution_file = Path("logs") / f"0-{task.function_name}_exp" / "best_solution.py"
                if not solution_file.exists():
                    self.logger.info("Best solution not found in logs, checking workspaces...")
                    solution_file = Path("workspaces") / f"0-{task.function_name}_exp" / "solution.py"
                
                if solution_file.exists():
                    with open(solution_file) as f:
                        code = f.read()
                        self.logger.info(f"Found solution at: {solution_file}")
                else:
                    self.logger.error("Solution file not found in logs or workspaces")
                    code = ""
                    
            except subprocess.TimeoutExpired:
                self.logger.error("AIDE process timed out")
                process.kill()
                return "", {
                    'total_time': time.time() - start_time,
                    'aide_execution_time': time.time() - aide_start,
                    'setup_time': setup_time,
                    'cleanup_time': 0.0
                }
                
            aide_time = time.time() - aide_start
            cleanup_start = time.time()
            
            # Clean up
            cleanup_time = time.time() - cleanup_start
            
            return code, {
                'total_time': time.time() - start_time,
                'aide_execution_time': aide_time,
                'setup_time': setup_time,
                'cleanup_time': cleanup_time
            }
            
        finally:
            # Always try to restore original directory
            os.chdir(original_dir)

    def _update_config_with_model(self):
        """Update the AIDE config file with the specified model"""
        try:
            # Find the config file
            config_path = Path(__file__).parent.parent / "aideml" / "aide" / "utils" / "config.yaml"
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
        # Find the correct solution file based on time budget
        log_dir = Path("experiment_workspace/tasks") / task.function_name / "logs"
        if log_dir.exists():
            # Find latest experiment directory
            exp_dirs = list(log_dir.glob("*"))
            if exp_dirs:
                latest_exp_dir = max(exp_dirs, key=lambda p: p.stat().st_mtime)
                solution_file = latest_exp_dir / f"solution_{task.time_budget}m.py"
                if solution_file.exists():
                    code = solution_file.read_text()
                    self.logger.info(f"Using solution from: {solution_file}")
        
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
        
        # Score implementation
        performance_score = self.quality_scorer.score_code(code, task.test_cases)
        
        return TaskResult(
            code=code,
            time_taken=time_taken,
            aide_execution_time=aide_time,
            setup_time=setup_time,
            cleanup_time=cleanup_time,
            delegation_depth=delegation_depth,
            constraint_violations=violations,
            performance_score=performance_score
        )

    def analyze_results(self, results: Dict[float, List[TaskResult]]) -> Dict[str, Any]:
        """Analyze experiment results
        
        Args:
            results: Dictionary mapping time budgets to lists of task results
            
        Returns:
            Dictionary containing analysis metrics
        """
        analysis = {
            'time_budgets': [],
            'performance_scores': {'mean': [], 'std': []},
            'total_times': {'mean': [], 'std': []},
            'aide_times': {'mean': [], 'std': []},
            'delegation_depths': {'mean': [], 'std': []},
            'violation_counts': {'mean': [], 'std': []},
            'implementations': [],
            'summary': {}
        }
        
        # Extract metrics for each time budget
        for budget, budget_results in sorted(results.items()):
            # Skip empty results
            if not budget_results:
                continue
                
            analysis['time_budgets'].append(budget)
            
            # Calculate metrics for this budget
            perf_scores = [r.performance_score for r in budget_results]
            total_times = [r.time_taken for r in budget_results]
            aide_times = [r.aide_execution_time for r in budget_results]
            depths = [r.delegation_depth for r in budget_results]
            violations = [len(r.constraint_violations) for r in budget_results]
            
            # Store means and standard deviations
            analysis['performance_scores']['mean'].append(float(np.mean(perf_scores)))
            analysis['performance_scores']['std'].append(float(np.std(perf_scores)))
            analysis['total_times']['mean'].append(float(np.mean(total_times)))
            analysis['total_times']['std'].append(float(np.std(total_times)))
            analysis['aide_times']['mean'].append(float(np.mean(aide_times)))
            analysis['aide_times']['std'].append(float(np.std(aide_times)))
            analysis['delegation_depths']['mean'].append(float(np.mean(depths)))
            analysis['delegation_depths']['std'].append(float(np.std(depths)))
            analysis['violation_counts']['mean'].append(float(np.mean(violations)))
            analysis['violation_counts']['std'].append(float(np.std(violations)))
            
            # Store best implementation for this budget
            best_idx = np.argmax(perf_scores)
            analysis['implementations'].append(budget_results[best_idx].code)
            
        # Calculate summary statistics
        with np.errstate(divide='ignore', invalid='ignore'):
            analysis['summary'] = {
                'performance_score': {
                    'mean': float(np.mean([s for s in analysis['performance_scores']['mean'] if s is not None])),
                    'std': float(np.std([s for s in analysis['performance_scores']['mean'] if s is not None]))
                },
                'total_time': {
                    'mean': float(np.mean([t for t in analysis['total_times']['mean'] if t is not None])),
                    'std': float(np.std([t for t in analysis['total_times']['mean'] if t is not None]))
                },
                'aide_time': {
                    'mean': float(np.mean([t for t in analysis['aide_times']['mean'] if t is not None])),
                    'std': float(np.std([t for t in analysis['aide_times']['mean'] if t is not None]))
                },
                'time_efficiency': {
                    'mean': float(np.mean([t/b for t, b in zip(analysis['total_times']['mean'], analysis['time_budgets']) if t is not None])),
                    'std': float(np.std([t/b for t, b in zip(analysis['total_times']['mean'], analysis['time_budgets']) if t is not None]))
                },
                'success_rate': float(np.mean([s > 0.8 for s in analysis['performance_scores']['mean'] if s is not None]))
            }
            
            # Calculate correlations
            analysis['correlations'] = {
                'actual_vs_budget': self.safe_correlation(
                    np.array(analysis['time_budgets']), 
                    np.array([t for t in analysis['total_times']['mean'] if t is not None])
                ),
                'depth_vs_violations': self.safe_correlation(
                    np.array([d for d in analysis['delegation_depths']['mean'] if d is not None]),
                    np.array([v for v in analysis['violation_counts']['mean'] if v is not None])
                )
            }
            
        return analysis

    def print_analysis(self, analysis: Dict[str, Any]) -> None:
        """Print analysis results in a readable format"""
        self.logger.info("\nAnalysis Results:")
        self.logger.info("=================\n")
        self.logger.info("Detailed Results by Time Budget:\n")
        
        for i, time_budget in enumerate(analysis['time_budgets']):
            self.logger.info(f"Time Budget: {time_budget} minutes")
            self.logger.info(f"Performance Score: {analysis['performance_scores']['mean'][i]:.2f} ± {analysis['performance_scores']['std'][i]:.2f}")
            self.logger.info(f"Total Time: {analysis['total_times']['mean'][i]:.1f}s ± {analysis['total_times']['std'][i]:.1f}s")
            self.logger.info(f"AIDE Time: {analysis['aide_times']['mean'][i]:.1f}s ± {analysis['aide_times']['std'][i]:.1f}s")
            self.logger.info(f"Delegation Depth: {analysis['delegation_depths']['mean'][i]:.1f} ± {analysis['delegation_depths']['std'][i]:.1f}")
            self.logger.info(f"Constraint Violations: {analysis['violation_counts']['mean'][i]:.1f} ± {analysis['violation_counts']['std'][i]:.1f}")
            
            # Print best implementation for this time budget
            if analysis['implementations'][i]:
                self.logger.info("\nBest Implementation:")
                self.logger.info("-------------------")
                self.logger.info(analysis['implementations'][i])
            self.logger.info("\n")

        self.logger.info("Summary Statistics:")
        self.logger.info(f"Mean Performance Score: {analysis['summary']['performance_score']['mean']:.2f} ± {analysis['summary']['performance_score']['std']:.2f}")
        self.logger.info(f"Mean Total Time: {analysis['summary']['total_time']['mean']:.1f}s ± {analysis['summary']['total_time']['std']:.1f}s")
        self.logger.info(f"Mean AIDE Time: {analysis['summary']['aide_time']['mean']:.1f}s ± {analysis['summary']['aide_time']['std']:.1f}s")
        self.logger.info(f"Mean Time Efficiency: {analysis['summary']['time_efficiency']['mean']:.2f} ± {analysis['summary']['time_efficiency']['std']:.2f}")
        self.logger.info(f"Success Rate: {analysis['summary']['success_rate']*100:.1f}%\n")
        
        self.logger.info("Correlations:")
        self.logger.info(f"Time Budget vs Actual Time: {analysis['correlations']['actual_vs_budget']:.2f}")
        self.logger.info(f"Delegation Depth vs Violations: {analysis['correlations']['depth_vs_violations']:.2f}\n")

class ExperimentRunner:
    """Runs coding task experiments"""
    def __init__(self, workspace: str = "experiment_workspace", model: str = "gpt-4-turbo-preview"):
        # Configure logging with a more specific format and file output
        log_format = '%(asctime)s - %(levelname)-8s %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        
        # Create a file handler for errors
        error_handler = logging.FileHandler('framework_errors.log')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(log_format, date_format))
        
        # Create a console handler for info and above
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter(log_format, date_format))
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Remove any existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
        # Add our handlers
        root_logger.addHandler(error_handler)
        root_logger.addHandler(console_handler)
        
        # Create our specific logger
        self.logger = logging.getLogger('ExperimentRunner')
        self.agent = AIDEAgent(workspace=workspace, model=model)
        self.evaluator = Evaluator()

        self.model = model
        
        # Suppress specific error messages
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        logging.getLogger('absl').setLevel(logging.ERROR)
        
        # Add environment variables to suppress GRPC warnings
        os.environ['GRPC_VERBOSITY'] = 'ERROR'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
        
        for i, time_budget in enumerate(analysis['time_budgets']):
            self.logger.info(f"Time Budget: {time_budget} minutes")
            self.logger.info(f"Performance Score: {analysis['performance_scores']['mean'][i]:.2f} ± {analysis['performance_scores']['std'][i]:.2f}")
            self.logger.info(f"Total Time: {analysis['total_times']['mean'][i]:.1f}s ± {analysis['total_times']['std'][i]:.1f}s")
            self.logger.info(f"AIDE Time: {analysis['aide_times']['mean'][i]:.1f}s ± {analysis['aide_times']['std'][i]:.1f}s")
            self.logger.info(f"Delegation Depth: {analysis['delegation_depths']['mean'][i]:.1f} ± {analysis['delegation_depths']['std'][i]:.1f}")
            self.logger.info(f"Constraint Violations: {analysis['violation_counts']['mean'][i]:.1f} ± {analysis['violation_counts']['std'][i]:.1f}")
            
            # Print best implementation for this time budget
            if analysis['implementations'][i]:
                self.logger.info("\nBest Implementation:")
                self.logger.info("-------------------")
                self.logger.info(analysis['implementations'][i])
            self.logger.info("\n")

        self.logger.info("Summary Statistics:")
        self.logger.info(f"Mean Performance Score: {analysis['summary']['performance_score']['mean']:.2f} ± {analysis['summary']['performance_score']['std']:.2f}")
        self.logger.info(f"Mean Total Time: {analysis['summary']['total_time']['mean']:.1f}s ± {analysis['summary']['total_time']['std']:.1f}s")
        self.logger.info(f"Mean AIDE Time: {analysis['summary']['aide_time']['mean']:.1f}s ± {analysis['summary']['aide_time']['std']:.1f}s")
        self.logger.info(f"Mean Time Efficiency: {analysis['summary']['time_efficiency']['mean']:.2f} ± {analysis['summary']['time_efficiency']['std']:.2f}")
        self.logger.info(f"Success Rate: {analysis['summary']['success_rate']*100:.1f}%\n")
        
        self.logger.info("Correlations:")
        self.logger.info(f"Time Budget vs Actual Time: {analysis['correlations']['actual_vs_budget']:.2f}")
        self.logger.info(f"Delegation Depth vs Violations: {analysis['correlations']['depth_vs_violations']:.2f}\n")

    def analyze_results(self, results: Dict[float, List[TaskResult]]) -> Dict[str, Any]:
        """Analyze experiment results
        
        Args:
            results: Dictionary mapping time budgets to lists of task results
            
        Returns:
            Dictionary containing analysis metrics
        """
        analysis = {
            'time_budgets': [],
            'performance_scores': {'mean': [], 'std': []},
            'total_times': {'mean': [], 'std': []},
            'aide_times': {'mean': [], 'std': []},
            'delegation_depths': {'mean': [], 'std': []},
            'violation_counts': {'mean': [], 'std': []},
            'implementations': [],
            'summary': {}
        }
        
        # Extract metrics for each time budget
        for budget, budget_results in sorted(results.items()):
            # Skip empty results
            if not budget_results:
                continue
                
            analysis['time_budgets'].append(budget)
            
            # Calculate metrics for this budget
            perf_scores = [r.performance_score for r in budget_results]
            total_times = [r.time_taken for r in budget_results]
            aide_times = [r.aide_execution_time for r in budget_results]
            depths = [r.delegation_depth for r in budget_results]
            violations = [len(r.constraint_violations) for r in budget_results]
            
            # Store means and standard deviations
            analysis['performance_scores']['mean'].append(float(np.mean(perf_scores)))
            analysis['performance_scores']['std'].append(float(np.std(perf_scores)))
            analysis['total_times']['mean'].append(float(np.mean(total_times)))
            analysis['total_times']['std'].append(float(np.std(total_times)))
            analysis['aide_times']['mean'].append(float(np.mean(aide_times)))
            analysis['aide_times']['std'].append(float(np.std(aide_times)))
            analysis['delegation_depths']['mean'].append(float(np.mean(depths)))
            analysis['delegation_depths']['std'].append(float(np.std(depths)))
            analysis['violation_counts']['mean'].append(float(np.mean(violations)))
            analysis['violation_counts']['std'].append(float(np.std(violations)))
            
            # Store best implementation for this budget
            best_idx = np.argmax(perf_scores)
            analysis['implementations'].append(budget_results[best_idx].code)
            
        # Calculate summary statistics
        with np.errstate(divide='ignore', invalid='ignore'):
            analysis['summary'] = {
                'performance_score': {
                    'mean': float(np.mean([s for s in analysis['performance_scores']['mean'] if s is not None])),
                    'std': float(np.std([s for s in analysis['performance_scores']['mean'] if s is not None]))
                },
                'total_time': {
                    'mean': float(np.mean([t for t in analysis['total_times']['mean'] if t is not None])),
                    'std': float(np.std([t for t in analysis['total_times']['mean'] if t is not None]))
                },
                'aide_time': {
                    'mean': float(np.mean([t for t in analysis['aide_times']['mean'] if t is not None])),
                    'std': float(np.std([t for t in analysis['aide_times']['mean'] if t is not None]))
                },
                'time_efficiency': {
                    'mean': float(np.mean([t/b for t, b in zip(analysis['total_times']['mean'], analysis['time_budgets']) if t is not None])),
                    'std': float(np.std([t/b for t, b in zip(analysis['total_times']['mean'], analysis['time_budgets']) if t is not None]))
                },
                'success_rate': float(np.mean([s > 0.8 for s in analysis['performance_scores']['mean'] if s is not None]))
            }
            
            # Calculate correlations
            analysis['correlations'] = {
                'actual_vs_budget': self.safe_correlation(
                    np.array(analysis['time_budgets']), 
                    np.array([t for t in analysis['total_times']['mean'] if t is not None])
                ),
                'depth_vs_violations': self.safe_correlation(
                    np.array([d for d in analysis['delegation_depths']['mean'] if d is not None]),
                    np.array([v for v in analysis['violation_counts']['mean'] if v is not None])
                )
            }
            
        return analysis

    def run_experiment(self, tasks: List[CodingTask], num_runs: int = 1) -> Dict[float, List[TaskResult]]:
        """Run experiment with multiple runs per time budget
        
        Args:
            tasks: List of coding tasks
            num_runs: Number of runs per time budget
            model: Model to use for AIDE (default: llama3.2:1b)
            
        Returns:
            Dictionary mapping time budgets to lists of TaskResult objects
        """

        model = self.model

        # Group tasks by time budget
        tasks_by_budget = {}
        for task in tasks:
            if task.time_budget not in tasks_by_budget:
                tasks_by_budget[task.time_budget] = []
            tasks_by_budget[task.time_budget].append(task)
            
        results = {}
        for time_budget, budget_tasks in tasks_by_budget.items():
            self.logger.info(f"\nRunning {num_runs} tasks with {time_budget:.1f} minute budget")
            
            # Store results for all runs with this time budget
            budget_results = []
            
            for i, task in enumerate(budget_tasks):
                self.logger.info(f"\nRun {i+1}/{num_runs}")
                
                # Set up AIDE agent with specified model
                self.agent = AIDEAgent(model=model)
                
                # Generate implementation
                code, timing_stats = self.agent.implement_function(task)
                
                # Evaluate the implementation
                result = self.evaluator.evaluate_task(
                    task=task,
                    code=code,
                    time_taken=timing_stats['total_time'],
                    aide_time=timing_stats['aide_execution_time'],
                    setup_time=timing_stats['setup_time'],
                    cleanup_time=timing_stats['cleanup_time']
                )
                budget_results.append(result)
                
                # Log immediate results for this run
                self.logger.info(f"Total time taken: {timing_stats['total_time']:.1f}s")
                self.logger.info(f"AIDE execution time: {timing_stats['aide_execution_time']:.1f}s")
                self.logger.info(f"Setup time: {timing_stats['setup_time']:.1f}s")
                self.logger.info(f"Cleanup time: {timing_stats['cleanup_time']:.1f}s")
                self.logger.info(f"Delegation depth: {result.delegation_depth}")
                self.logger.info(f"Performance score: {result.performance_score:.2f}")
                if result.constraint_violations:
                    self.logger.info("Constraint violations:")
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
            self.logger.info(f"\nStatistics for {time_budget} minute budget:")
            self.logger.info(f"Performance Score: {np.mean(perf_scores):.2f} ± {np.std(perf_scores):.2f}")
            self.logger.info(f"Total Time: {np.mean(total_times):.1f}s ± {np.std(total_times):.1f}s")
            self.logger.info(f"AIDE Time: {np.mean(aide_times):.1f}s ± {np.std(aide_times):.1f}s")
            self.logger.info(f"Delegation Depth: {np.mean(depths):.1f} ± {np.std(depths):.1f}")
            self.logger.info(f"Violations: {np.mean(violations):.1f} ± {np.std(violations):.1f}")
            
        return results