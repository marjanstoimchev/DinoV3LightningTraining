#%%
#!/usr/bin/env python3
"""
Enhanced DINOv3 Training Loss Visualization
==========================================
- Professional styling with individual colors per loss
- Clean code structure with modular design
- Publication-quality plots with smart layout
- Comprehensive loss tracking and analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set professional styling
plt.style.use('default')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8
})

class DINOv3LossVisualizer:
    """Professional training loss visualization for DINOv3 models"""
    
    def __init__(self, log_dir: str, output_dir: Optional[str] = None):
        """
        Initialize the loss visualizer
        
        Args:
            log_dir: Path to Lightning logs directory
            output_dir: Output directory for plots (default: log_dir/loss_plots)
        """
        self.log_dir = Path(log_dir)
        self.output_dir = Path(output_dir) if output_dir else self.log_dir / "loss_plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data: Optional[pd.DataFrame] = None
        self.loss_config = self._setup_loss_configuration()
        
    def _setup_loss_configuration(self) -> Dict:
        """Configure loss types, colors, and display settings"""
        return {
            'loss_metrics': {
                'total_loss': {
                    'color': '#2E86AB',
                    'label': 'Total Loss',
                    'style': '-',
                    'log_scale': False
                },
                'train/dino_local_crops_loss': {
                    'color': '#A23B72', 
                    'label': 'DINO Local Crops',
                    'style': '-',
                    'log_scale': False
                },
                'train/dino_global_crops_loss': {
                    'color': '#F18F01',
                    'label': 'DINO Global Crops', 
                    'style': '-',
                    'log_scale': False
                },
                'train/koleo_loss': {
                    'color': '#C73E1D',
                    'label': 'Koleo Loss',
                    'style': '-',
                    'log_scale': False
                },
                'train/ibot_loss': {
                    'color': '#7209B7',
                    'label': 'iBOT Loss',
                    'style': '-', 
                    'log_scale': False
                },
                'train/gram_loss': {
                    'color': '#06A77D',
                    'label': 'Gram Loss',
                    'style': '-',
                    'log_scale': False
                }
            },
            'learning_params': {
                'train/lr': {
                    'color': '#FF6B6B',
                    'label': 'Learning Rate',
                    'style': '--',
                    'log_scale': False
                },
                'train/wd': {
                    'color': '#4ECDC4', 
                    'label': 'Weight Decay',
                    'style': '--',
                    'log_scale': False
                },
                'train/momentum': {
                    'color': '#45B7D1',
                    'label': 'Momentum',
                    'style': '--',
                    'log_scale': False
                },
                'train/teacher_temp': {
                    'color': '#96CEB4',
                    'label': 'Teacher Temperature',
                    'style': '--',
                    'log_scale': False
                }
            }
        }
    
    def load_training_data(self) -> bool:
        """Load and preprocess training CSV logs"""
        csv_file = self.log_dir / "csv_logs" / "metrics.csv"
        
        if not csv_file.exists():
            print(f"CSV log file not found: {csv_file}")
            return False
        
        try:
            df = pd.read_csv(csv_file)
            # Clean data: remove empty columns and sort by step
            df = df.dropna(axis=1, how="all").sort_index()
            self.data = df
            
            available_cols = [col for col in df.columns 
                            if any(col == metric for metrics in self.loss_config.values() 
                                  for metric in metrics.keys())]
            
            print(f"Training data loaded: {len(df)} records")
            print(f"Available metrics: {len(available_cols)} of {len(df.columns)} columns")
            return True
            
        except Exception as e:
            print(f"Error loading training data: {e}")
            return False
    
    def _get_available_metrics(self, metric_type: str) -> List[Tuple[str, Dict]]:
        """Get available metrics of specified type from loaded data"""
        if self.data is None:
            return []
            
        metrics = self.loss_config[metric_type]
        available = []
        
        for metric_name, config in metrics.items():
            if metric_name in self.data.columns:
                series = self.data[metric_name].dropna()
                if len(series) > 0:
                    available.append((metric_name, {**config, 'data': series}))
        
        return available
    
    
    def plot_individual_losses(self) -> None:
        """Create individual loss plots in a grid layout"""
        if self.data is None:
            return
            
        loss_metrics = self._get_available_metrics('loss_metrics')
        if not loss_metrics:
            print("No loss metrics found for individual plots")
            return
        
        n_losses = len(loss_metrics)
        cols = min(3, n_losses)
        rows = (n_losses + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        fig.suptitle('Individual Loss Component Analysis', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        # Handle single subplot case
        if n_losses == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
        
        for i, (_, config) in enumerate(loss_metrics):
            ax = axes[i]
            steps = config['data'].index
            values = config['data'].values
            
            # Plot line without fill
            ax.plot(steps, values, color=config['color'], 
                   linewidth=3, alpha=0.9, label=config['label'])
            
            # Styling
            title = config['label']
            ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
            ax.set_xlabel('Training Step', fontsize=11)
            ax.set_ylabel('Loss Value', fontsize=11)
            
            # Add statistics text
            final_val = values[-1]
            min_val = np.min(values)
            ax.text(0.02, 0.98, f'Final: {final_val:.6f}\nMin: {min_val:.6f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                   fontsize=9)
        
        # Hide unused subplots
        for j in range(n_losses, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        output_file = self.output_dir / "individual_losses.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Individual loss plots saved: {output_file}")
        plt.show()
    
    def plot_learning_parameters(self) -> None:
        """Plot learning rate, weight decay, and other training parameters"""
        if self.data is None:
            return
            
        learning_params = self._get_available_metrics('learning_params')
        if not learning_params:
            print("No learning parameters found")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training Parameters Evolution', 
                     fontsize=18, fontweight='bold', y=0.98)
        axes = axes.flatten()
        
        for i, (_, config) in enumerate(learning_params):
            if i >= 4:  # Max 4 subplots
                break
                
            ax = axes[i]
            steps = config['data'].index  
            values = config['data'].values
            
            ax.plot(steps, values, color=config['color'],
                   linewidth=3, linestyle=config['style'], alpha=0.9)
            
            ax.set_title(config['label'], fontsize=13, fontweight='bold', pad=15)
            ax.set_xlabel('Training Step', fontsize=11)
            ax.set_ylabel('Value', fontsize=11)
            
            if config['log_scale']:
                ax.set_yscale('log')
        
        # Hide unused subplots
        for j in range(len(learning_params), 4):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        output_file = self.output_dir / "learning_parameters.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Learning parameters saved: {output_file}")
        plt.show()
    
    def generate_training_summary(self) -> None:
        """Generate comprehensive training statistics summary"""
        if self.data is None:
            return
            
        print("\n" + "=" * 70)
        print("DINOV3 TRAINING SUMMARY")
        print("=" * 70)
        
        # Loss metrics summary
        loss_metrics = self._get_available_metrics('loss_metrics')
        if loss_metrics:
            print("\nLoss Metrics:")
            print("-" * 50)
            for _, config in loss_metrics:
                series = config['data']
                final_val = series.iloc[-1]
                min_val = series.min()
                max_val = series.max()
                mean_val = series.mean()
                
                print(f"  {config['label']:.<25} Final: {final_val:.6f}")
                print(f"  {'':<25} Min: {min_val:.6f} | Max: {max_val:.6f} | Avg: {mean_val:.6f}")
                print()
        
        # Learning parameters summary
        learning_params = self._get_available_metrics('learning_params')
        if learning_params:
            print("Learning Parameters:")
            print("-" * 50)
            for _, config in learning_params:
                series = config['data']
                final_val = series.iloc[-1]
                initial_val = series.iloc[0]
                
                print(f"  {config['label']:.<25} Initial: {initial_val:.8f} | Final: {final_val:.8f}")
        
        print("\n" + "=" * 70)
    
    def create_all_plots(self) -> None:
        """Generate all visualization plots"""
        print("Creating comprehensive loss visualizations...")
        
        self.plot_individual_losses() 
        self.plot_learning_parameters()
        self.generate_training_summary()
        
        print(f"\nAll plots saved to: {self.output_dir}")


def main():
    """Main execution function for single training run visualization"""
    print("DINOv3 Training Loss Visualizer")
    
    log_dir = '../output_multi_gpu_hf_v2'
    output_dir = 'log_plots'
    
    print("=" * 40)
    print(f"Log directory: {log_dir}")
    
    # Create visualizer and generate plots
    visualizer = DINOv3LossVisualizer(log_dir, output_dir)
    
    if not visualizer.load_training_data():
        print("No training data found. Exiting.")
        return
    
    visualizer.create_all_plots()


if __name__ == "__main__":
    main()