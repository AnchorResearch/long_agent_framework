#!/usr/bin/env python3

import subprocess
import os
from pathlib import Path
import logging
import time
import sys

def setup_logging():
    """Configure logging with both file and console handlers"""
    # Create formatters with timestamps
    console_formatter = logging.Formatter('%(levelname)s [%(asctime)s] %(message)s', 
                                        datefmt='%Y-%m-%d %H:%M:%S')
    
    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

def main():
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Get the directory containing the tasks
    tasks_dir = Path(__file__).parent / 'tasks'
    
    # Get all .md files in the tasks directory
    task_files = sorted(tasks_dir.glob('*.md'))
    
    if not task_files:
        logger.error("No task files found in the tasks directory!")
        sys.exit(1)
    
    # Process command line arguments for run_evaluation.py
    # Remove script name and keep all other arguments
    additional_args = sys.argv[1:]
    
    total_tasks = len(task_files)
    logger.info(f"Found {total_tasks} tasks to process")
    
    for i, task_file in enumerate(task_files, 1):
        task_name = task_file.stem  # Get filename without extension
        logger.info(f"\n[{i}/{total_tasks}] Running task: {task_name}")
        
        try:
            # Construct the command
            cmd = [sys.executable, 'run_evaluation.py', '--task', task_name] + additional_args
            
            # Run the evaluation script
            logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent,
                check=True,
                text=True,
                capture_output=True
            )
            
            # Log the output
            if result.stdout:
                logger.info("Output:")
                print(result.stdout)
            
            if result.stderr:
                logger.warning("Stderr:")
                print(result.stderr)
                
            logger.info(f"Successfully completed task: {task_name}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running task {task_name}:")
            if e.stdout:
                print(e.stdout)
            if e.stderr:
                print(e.stderr)
            logger.error(f"Process returned non-zero exit status {e.returncode}")
            
        except Exception as e:
            logger.error(f"Unexpected error running task {task_name}: {str(e)}")
        
        # Add a small delay between tasks
        time.sleep(1)
    
    logger.info("\nAll tasks completed!")

if __name__ == "__main__":
    main() 