import glob
import os

def get_task_name(exp_dir):
    # Look for any subdirectory that matches the pattern task_name_t*_r*
    subdirs = next(os.walk(exp_dir))[1]
    # Filter out 'plots' directory and find the first task-related directory
    task_dirs = [d for d in subdirs if d != 'plots' and '_t' in d and '_r' in d]
    if task_dirs:
        # Extract task name from the first matching directory
        task_name = task_dirs[0].split('_t')[0]
        return task_name.replace('_', ' ')
    return "Unknown Task"

# Create the output file
output_path = os.path.join(os.path.dirname(__file__), 'all_statistics.txt')
with open(output_path, 'w') as outfile:
    outfile.write("Summary Statistics from All Experiments\n\n")
    
    # Find all summary_statistics.txt files
    base_dir = os.path.dirname(__file__)
    pattern = os.path.join(base_dir, "experiment_workspace/experiment_*/plots/summary_statistics.txt")
    
    for filepath in glob.glob(pattern):
        # Get experiment directory
        exp_dir = os.path.dirname(os.path.dirname(filepath))
        exp_name = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
        task_name = get_task_name(exp_dir)
        
        # Write experiment header with task name
        outfile.write(f"=== Experiment: {exp_name} (Task: {task_name}) ===\n")
        
        # Copy contents of the summary statistics file
        try:
            with open(filepath, 'r') as infile:
                outfile.write(infile.read())
            outfile.write("\n\n")
        except Exception as e:
            outfile.write(f"Error reading file: {str(e)}\n\n") 