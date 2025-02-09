import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def load_results(csv_path):
    """Load and preprocess the results CSV file"""
    df = pd.read_csv(csv_path)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Clean up violations string (remove quotes and split by semicolon)
    # Handle empty strings and convert to list
    df['violations'] = df['violations'].apply(lambda x: [] if pd.isna(x) or x.strip('"\' ') == '' 
                                            else str(x).strip('"\' ').split(';'))
    
    return df

def plot_scores_vs_time_budget(df, output_dir):
    """Plot various scores against time budget"""
    plt.figure(figsize=(12, 8))
    
    # Calculate means and standard deviations for each score type
    scores = ['test_passing_score', 'code_style_score', 'efficiency_score', 'final_score']
    for score in scores:
        means = df.groupby('time_budget')[score].mean()
        stds = df.groupby('time_budget')[score].std()
        
        plt.errorbar(means.index, means.values, yerr=stds.values, 
                    marker='o', label=score.replace('_', ' ').title())
    
    plt.xlabel('Time Budget (minutes)')
    plt.ylabel('Score')
    plt.title('Scores vs Time Budget')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'scores_vs_time_budget.png')
    plt.close()

def plot_timing_analysis(df, output_dir):
    """Plot timing-related metrics against time budget"""
    plt.figure(figsize=(12, 8))
    
    # Calculate means and standard deviations for timing metrics
    timing_metrics = ['total_time', 'aide_time', 'setup_time', 'cleanup_time']
    for metric in timing_metrics:
        means = df.groupby('time_budget')[metric].mean()
        stds = df.groupby('time_budget')[metric].std()
        
        plt.errorbar(means.index, means.values, yerr=stds.values, 
                    marker='o', label=metric.replace('_', ' ').title())
    
    plt.xlabel('Time Budget (minutes)')
    plt.ylabel('Time (seconds)')
    plt.title('Execution Times vs Time Budget')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'timing_vs_time_budget.png')
    plt.close()

def plot_violations_analysis(df, output_dir):
    """Plot violation-related metrics against time budget"""
    plt.figure(figsize=(12, 8))
    
    # Plot number of violations
    means = df.groupby('time_budget')['num_violations'].mean()
    stds = df.groupby('time_budget')['num_violations'].std()
    
    plt.errorbar(means.index, means.values, yerr=stds.values, 
                marker='o', label='Number of Violations')
    
    plt.xlabel('Time Budget (minutes)')
    plt.ylabel('Number of Violations')
    plt.title('Constraint Violations vs Time Budget')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'violations_vs_time_budget.png')
    plt.close()

def plot_correlation_matrix(df, output_dir):
    """Plot correlation matrix of numerical metrics"""
    # Select numerical columns for correlation
    numerical_cols = ['time_budget', 'total_time', 'aide_time', 'setup_time', 'cleanup_time',
                     'test_passing_score', 'code_style_score', 'efficiency_score', 'final_score',
                     'delegation_depth', 'num_violations']
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Matrix of Metrics')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png')
    plt.close()

def plot_delegation_depth_analysis(df, output_dir):
    """Plot delegation depth analysis"""
    plt.figure(figsize=(12, 8))
    
    # Plot delegation depth vs time budget
    means = df.groupby('time_budget')['delegation_depth'].mean()
    stds = df.groupby('time_budget')['delegation_depth'].std()
    
    plt.errorbar(means.index, means.values, yerr=stds.values, 
                marker='o', label='Delegation Depth')
    
    plt.xlabel('Time Budget (minutes)')
    plt.ylabel('Delegation Depth')
    plt.title('Delegation Depth vs Time Budget')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'delegation_depth_vs_time_budget.png')
    plt.close()

def generate_summary_stats(df):
    """Generate summary statistics"""
    summary = {
        'Overall Statistics': {
            'Total Runs': len(df),
            'Average Final Score': df['final_score'].mean(),
            'Average Total Time': df['total_time'].mean(),
            'Average Number of Violations': df['num_violations'].mean(),
            'Success Rate (Final Score > 0.8)': (df['final_score'] > 0.8).mean() * 100
        },
        'By Time Budget': df.groupby('time_budget').agg({
            'final_score': ['mean', 'std'],
            'total_time': ['mean', 'std'],
            'num_violations': ['mean', 'sum'],
            'delegation_depth': ['mean', 'std']
        }).round(2)
    }
    
    return summary

def main():
    # Set up paths
    workspace_dir = Path(__file__).parent / "experiment_workspace"
    
    # Find the most recent experiment directory
    experiment_dirs = sorted([d for d in workspace_dir.glob("experiment_*") if d.is_dir()], reverse=True)
    if not experiment_dirs:
        raise FileNotFoundError("No experiment directories found")
    
    latest_exp_dir = experiment_dirs[0]
    print(f"Using latest experiment directory: {latest_exp_dir}")
    
    # Find the results file in the experiment directory
    results_files = list(latest_exp_dir.glob("experiment_results_*.csv"))
    if not results_files:
        raise FileNotFoundError(f"No results files found in {latest_exp_dir}")
    
    results_file = results_files[0]
    plots_dir = latest_exp_dir / "plots"
    
    # Create plots directory if it doesn't exist
    plots_dir.mkdir(exist_ok=True)
    
    # Load results
    df = load_results(results_file)
    
    # Generate plots
    plot_scores_vs_time_budget(df, plots_dir)
    plot_timing_analysis(df, plots_dir)
    plot_violations_analysis(df, plots_dir)
    plot_correlation_matrix(df, plots_dir)
    plot_delegation_depth_analysis(df, plots_dir)
    
    # Generate and save summary statistics
    summary_stats = generate_summary_stats(df)
    
    # Print summary statistics
    print("\nOverall Statistics:")
    print("==================")
    for key, value in summary_stats['Overall Statistics'].items():
        print(f"{key}: {value:.2f}")
    
    print("\nStatistics by Time Budget:")
    print("=========================")
    print(summary_stats['By Time Budget'])
    
    # Save summary statistics to file
    with open(plots_dir / 'summary_statistics.txt', 'w') as f:
        f.write("Overall Statistics:\n")
        f.write("==================\n")
        for key, value in summary_stats['Overall Statistics'].items():
            f.write(f"{key}: {value:.2f}\n")
        
        f.write("\nStatistics by Time Budget:\n")
        f.write("=========================\n")
        f.write(str(summary_stats['By Time Budget']))

if __name__ == "__main__":
    main() 