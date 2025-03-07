import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

def compare_cmd_vel(file1_path, file2_path, output_path=None, plot=True, has_headers=False):
    """
    Compare two CSV files containing command velocities using various metrics.
    Works with headerless CSV files containing cmd_v and cmd_w values.
    
    Args:
        file1_path (str): Path to first CSV file (ground truth or reference)
        file2_path (str): Path to second CSV file (prediction or comparison)
        output_path (str, optional): Path to save comparison results and plots
        plot (bool): Whether to generate comparison plots
        has_headers (bool): Set to True if CSV files have headers, False if they're just values
        
    Returns:
        dict: Dictionary containing comparison metrics
    """
    # Load the CSV files based on whether they have headers
    if has_headers:
        # If files have headers
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
        
        # Use the specific column names if they exist
        if 'cmd_v' in df1.columns and 'cmd_w' in df1.columns:
            linear_col1, angular_col1 = 'cmd_v', 'cmd_w'
        else:
            linear_col1, angular_col1 = df1.columns[0], df1.columns[1]
            
        if 'cmd_v' in df2.columns and 'cmd_w' in df2.columns:
            linear_col2, angular_col2 = 'cmd_v', 'cmd_w'
        else:
            linear_col2, angular_col2 = df2.columns[0], df2.columns[1]
    else:
        # If files are headerless (just values)
        df1 = pd.read_csv(file1_path, header=None, names=['cmd_v', 'cmd_w'])
        df2 = pd.read_csv(file2_path, header=None, names=['cmd_v', 'cmd_w'])
        linear_col1, angular_col1 = 'cmd_v', 'cmd_w'
        linear_col2, angular_col2 = 'cmd_v', 'cmd_w'
    
    # Check if the dataframes have the same length
    if len(df1) != len(df2):
        print(f"Warning: Files have different lengths. File1: {len(df1)}, File2: {len(df2)}")
        # Use the shorter length
        min_len = min(len(df1), len(df2))
        df1 = df1.iloc[:min_len]
        df2 = df2.iloc[:min_len]
    
    # Print the first few rows of each dataframe to verify loading
    print("\nFirst rows of File 1:")
    print(df1.head())
    print("\nFirst rows of File 2:")
    print(df2.head())
    
    # Calculate metrics
    metrics = {}
    
    # Linear velocity metrics
    lin_mae = mean_absolute_error(df1[linear_col1], df2[linear_col2])
    lin_rmse = np.sqrt(mean_squared_error(df1[linear_col1], df2[linear_col2]))
    lin_max_error = max(abs(df1[linear_col1] - df2[linear_col2]))
    
    metrics['linear_velocity'] = {
        'MAE': lin_mae,
        'RMSE': lin_rmse,
        'Max_Error': lin_max_error
    }
    
    # Angular velocity metrics
    ang_mae = mean_absolute_error(df1[angular_col1], df2[angular_col2])
    ang_rmse = np.sqrt(mean_squared_error(df1[angular_col1], df2[angular_col2]))
    ang_max_error = max(abs(df1[angular_col1] - df2[angular_col2]))
    
    metrics['angular_velocity'] = {
        'MAE': ang_mae,
        'RMSE': ang_rmse,
        'Max_Error': ang_max_error
    }
    
    # Combined metrics
    metrics['overall'] = {
        'MAE': (lin_mae + ang_mae) / 2,
        'RMSE': (lin_rmse + ang_rmse) / 2,
        'Max_Error': max(lin_max_error, ang_max_error)
    }
    
    # Print metrics
    print("\n=== Velocity Comparison Metrics ===")
    for category, category_metrics in metrics.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for metric_name, value in category_metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    # Generate plots if requested
    if plot:
        plt.figure(figsize=(12, 10))
        
        # Linear velocity plot
        plt.subplot(3, 1, 1)
        plt.plot(df1[linear_col1], label='File1 Linear')
        plt.plot(df2[linear_col2], label='File2 Linear')
        plt.title(f'Linear Velocity Comparison (MAE: {lin_mae:.4f})')
        plt.xlabel('Time Steps')
        plt.ylabel('Linear Velocity (cmd_v)')
        plt.legend()
        plt.grid(True)
        
        # Angular velocity plot
        plt.subplot(3, 1, 2)
        plt.plot(df1[angular_col1], label='File1 Angular')
        plt.plot(df2[angular_col2], label='File2 Angular')
        plt.title(f'Angular Velocity Comparison (MAE: {ang_mae:.4f})')
        plt.xlabel('Time Steps')
        plt.ylabel('Angular Velocity (cmd_w)')
        plt.legend()
        plt.grid(True)
        
        # Error plot
        plt.subplot(3, 1, 3)
        plt.plot(abs(df1[linear_col1] - df2[linear_col2]), label='Linear Velocity Error')
        plt.plot(abs(df1[angular_col1] - df2[angular_col2]), label='Angular Velocity Error')
        plt.title('Absolute Error Over Time')
        plt.xlabel('Time Steps')
        plt.ylabel('Absolute Error')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save plot if output path provided
        if output_path:
            # Create output directory if it doesn't exist
            os.makedirs(output_path, exist_ok=True)
            
            plt.savefig(f"{output_path}/velocity_comparison.png", dpi=300)
            
            # Also save metrics to a CSV file
            metrics_df = pd.DataFrame()
            for category, category_metrics in metrics.items():
                for metric_name, value in category_metrics.items():
                    metrics_df.loc[category, metric_name] = value
            
            # Save error data for further analysis
            error_df = pd.DataFrame({
                'linear_error': abs(df1[linear_col1] - df2[linear_col2]),
                'angular_error': abs(df1[angular_col1] - df2[angular_col2])
            })
            error_df.to_csv(f"{output_path}/error_values.csv", index=False)
            
            # Save metrics summary
            metrics_df.to_csv(f"{output_path}/velocity_metrics.csv")
            print(f"Results saved to {output_path}")
        
        plt.show()
    
    return metrics

# Example usage
if __name__ == "__main__":
    # Replace with your actual file paths
    file1 = "test1/input_data/cmd_vel_output.csv"
    file2 = "test1/output_data/cmd_vel.csv"
    output_dir = "comparison_results"
    
    # Run comparison - set has_headers=False for headerless CSV files
    compare_cmd_vel(file1, file2, output_dir, has_headers=False)
