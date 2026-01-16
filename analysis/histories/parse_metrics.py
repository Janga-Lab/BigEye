import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
from collections import defaultdict

def parse_metrics_dict(data: Dict[str, Any]) -> Tuple[Dict, Dict, Dict, int, Dict]:
    """
    Parse the metrics dictionary and extract required information.
    
    Args:
        data: Dictionary containing train_validation and test metrics
        
    Returns:
        Tuple containing:
        - train_metrics: Dict of training metrics
        - val_metrics: Dict of validation metrics  
        - test_metrics: Dict of test metrics
        - best_epoch: Epoch with highest validation IoU
        - best_epoch_metrics: Metrics at best epoch across all sets
    """
    
    # 1. Separate metrics for training, validation, and test sets
    train_val_data = data['train_validation']
    test_metrics = data['test']
    
    # Separate training and validation metrics
    train_metrics = {}
    val_metrics = {}
    
    for metric_name, values in train_val_data.items():
        if metric_name.startswith('val_'):
            # Validation metric
            val_metric_name = metric_name.replace('val_', '')
            val_metrics[val_metric_name] = values
        else:
            # Training metric
            train_metrics[metric_name] = values
    
    # 2. Find epoch where validation IoU is highest
    best_epoch = val_metrics['iou'].index(max(val_metrics['iou']))
    
    # 3. Get metric scores at the best epoch
    best_epoch_metrics = {
        'epoch': best_epoch,
        'train': {},
        'validation': {},
        'test': test_metrics
    }
    
    for metric_name, values in train_metrics.items():
        best_epoch_metrics['train'][metric_name] = values[best_epoch]
    
    for metric_name, values in val_metrics.items():
        best_epoch_metrics['validation'][metric_name] = values[best_epoch]
    
    return train_metrics, val_metrics, test_metrics, best_epoch, best_epoch_metrics

def process_directory(directory_path: str, file_pattern: str = '*.pickle') -> Tuple[List[Dict], Dict, Dict]:
    """
    Process all pickle files in a directory.
    
    Args:
        directory_path: Path to directory containing pickle files
        file_pattern: Glob pattern for files (default: '*.pkl')
        
    Returns:
        Tuple containing:
        - all_results: List of best_epoch_metrics for each file
        - averages: Dict of average metrics for each set
        - std_devs: Dict of standard deviations for each set
    """
    
    directory = Path(directory_path)
    pickle_files = list(directory.glob(file_pattern))
    
    if not pickle_files:
        print(f"No pickle files found in {directory_path} with pattern {file_pattern}")
        return [], {}, {}
    
    print(f"Found {len(pickle_files)} pickle files\n")
    
    all_results = []
    
    # Collect metrics from all files
    for file_path in sorted(pickle_files):
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            _, _, _, best_epoch, best_epoch_metrics = parse_metrics_dict(data)
            all_results.append({
                'file': file_path.name,
                'metrics': best_epoch_metrics
            })
            
            print(f"Processed: {file_path.name} (Best epoch: {best_epoch})")
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")
    
    # Calculate averages and standard deviations
    averages, std_devs = calculate_statistics(all_results)
    
    return all_results, averages, std_devs

def calculate_statistics(all_results: List[Dict]) -> Tuple[Dict, Dict]:
    """
    Calculate average and standard deviation for each metric across all files.
    
    Args:
        all_results: List of results from all files
        
    Returns:
        Tuple of (averages, std_devs) dictionaries
    """
    
    # Collect all metrics by set and metric name
    metrics_by_set = {
        'train': defaultdict(list),
        'validation': defaultdict(list),
        'test': defaultdict(list)
    }
    
    epochs = []
    
    for result in all_results:
        metrics = result['metrics']
        epochs.append(metrics['epoch'])
        
        for set_name in ['train', 'validation', 'test']:
            for metric_name, value in metrics[set_name].items():
                metrics_by_set[set_name][metric_name].append(value)
    
    # Calculate statistics
    averages = {
        'epoch': np.mean(epochs),
        'train': {},
        'validation': {},
        'test': {}
    }
    
    std_devs = {
        'epoch': np.std(epochs),
        'train': {},
        'validation': {},
        'test': {}
    }
    
    for set_name in ['train', 'validation', 'test']:
        for metric_name, values in metrics_by_set[set_name].items():
            averages[set_name][metric_name] = np.mean(values)
            std_devs[set_name][metric_name] = np.std(values)
    
    return averages, std_devs

def print_statistics(averages: Dict, std_devs: Dict, n_files: int):
    """Print statistics in a readable format."""
    
    print(f"\n{'='*80}")
    print(f"STATISTICS ACROSS {n_files} FILES")
    print(f"{'='*80}\n")
    
    print(f"BEST EPOCH: {averages['epoch']:.1f} ± {std_devs['epoch']:.1f}")
    print(f"{'='*80}\n")
    
    for set_name in ['train', 'validation', 'test']:
        print(f"{set_name.upper()} METRICS:")
        print("-" * 80)
        
        # Get all metric names and sort them
        metric_names = sorted(averages[set_name].keys())
        
        for metric in metric_names:
            avg = averages[set_name][metric]
            std = std_devs[set_name][metric]
            print(f"  {metric:15s}: {avg:.2f} ± {std:.2f}")
        
        print()
    
    print(f"{'='*80}\n")

def save_statistics_to_file(averages: Dict, std_devs: Dict, all_results: List[Dict], 
                           output_path: str):
    """Save statistics to a JSON file."""
    
    output_data = {
        'summary': {
            'n_files': len(all_results),
            'best_epoch_avg': float(averages['epoch']),
            'best_epoch_std': float(std_devs['epoch']),
            'averages': {
                set_name: {k: float(v) for k, v in metrics.items()}
                for set_name, metrics in averages.items() if set_name != 'epoch'
            },
            'std_devs': {
                set_name: {k: float(v) for k, v in metrics.items()}
                for set_name, metrics in std_devs.items() if set_name != 'epoch'
            }
        },
        'individual_results': [
            {
                'file': result['file'],
                'epoch': result['metrics']['epoch'],
                'train': result['metrics']['train'],
                'validation': result['metrics']['validation'],
                'test': result['metrics']['test']
            }
            for result in all_results
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Statistics saved to: {output_path}")

def main(directory_path: str, output_file: str = "lookup.json", file_pattern: str = '*.pickle'):
    """
    Main function to process all files and display statistics.
    
    Args:
        directory_path: Path to directory containing pickle files
        output_file: Optional path to save statistics JSON
        file_pattern: Glob pattern for files (default: '*.pkl')
    """
    
    all_results, averages, std_devs = process_directory(directory_path, file_pattern)
    
    if not all_results:
        return
    
    print_statistics(averages, std_devs, len(all_results))
    
    if output_file:
        print("ok")
        save_statistics_to_file(averages, std_devs, all_results, output_file)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python parse_metrics.py <directory_path> [output_file.json] [file_pattern]")
        print("\nArguments:")
        print("  directory_path : Path to directory containing pickle files (required)")
        print("  output_file    : Path to save statistics JSON (optional)")
        print("  file_pattern   : Glob pattern for files (optional, default: '*.pkl')")
        print("\nExamples:")
        print("  python parse_metrics.py ./metrics_data")
        print("  python parse_metrics.py ./metrics_data statistics.json")
        print("  python parse_metrics.py ./metrics_data statistics.json '*.pickle'")
        print("  python parse_metrics.py ./metrics_data statistics.json 'experiment_*.pkl'")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    file_pattern = sys.argv[3] if len(sys.argv) > 3 else '*.pickle'
    
    main(directory_path, output_file, file_pattern)
