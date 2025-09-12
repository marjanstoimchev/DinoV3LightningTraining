#!/usr/bin/env python3
"""
DINOv3 Training Configuration Comparison Tool
============================================
Compare multiple training configurations side-by-side:
- Loss component comparisons
- Learning rate schedules
- Training parameter evolution
- Performance summaries

Simple and straightforward multi-config analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8
})


class ConfigurationComparator:
    """Compare multiple DINOv3 training configurations"""
    
    def __init__(self, config_paths: Dict[str, str], output_dir: str = "comparison_plots"):
        """
        Initialize the comparator
        
        Args:
            config_paths: Dict mapping config names to their log directories
            output_dir: Output directory for comparison plots
        """
        self.config_paths = config_paths
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data: Dict[str, pd.DataFrame] = {}
        
        # Multi-config colors (one color per configuration)
        self.config_colors = [
            '#2E86AB', '#A23B72', '#F18F01', '#C73E1D', 
            '#7209B7', '#06A77D', '#FF6B6B', '#4ECDC4'
        ]
        
        # Single-config colors (different color per loss type)
        self.loss_colors = {
            'total_loss': '#2E86AB',
            'train/dino_global_crops_loss': '#F18F01',
            'train/dino_local_crops_loss': '#A23B72', 
            'train/ibot_loss': '#7209B7',
            'train/koleo_loss': '#C73E1D',
            'train/gram_loss': '#06A77D',
            'train/lr': '#FF6B6B'
        }
        
    def load_all_data(self) -> bool:
        """Load training data for all configurations"""
        print("Loading training data for all configurations...")
        
        success_count = 0
        for config_name, log_dir in self.config_paths.items():
            csv_file = Path(log_dir) / "csv_logs" / "metrics.csv"
            
            if not csv_file.exists():
                print(f"❌ CSV file not found for {config_name}: {csv_file}")
                continue
                
            try:
                df = pd.read_csv(csv_file)
                df = df.dropna(axis=1, how="all").sort_index()
                self.data[config_name] = df
                print(f"✅ Loaded {config_name}: {len(df)} records")
                success_count += 1
            except Exception as e:
                print(f"❌ Error loading {config_name}: {e}")
                
        print(f"\nSuccessfully loaded {success_count}/{len(self.config_paths)} configurations")
        return success_count > 0
    
    def _get_color(self, config_index: int, loss_type: str) -> str:
        """Get appropriate color based on number of configurations"""
        if len(self.data) == 1:
            # Single config: use different colors per loss type
            return self.loss_colors.get(loss_type, self.config_colors[0])
        else:
            # Multiple configs: use different colors per config
            return self.config_colors[config_index % len(self.config_colors)]
    
    def compare_total_losses(self) -> None:
        """Compare total loss across all configurations"""
        if not self.data:
            return
            
        plt.figure(figsize=(12, 8))
        
        for i, (config_name, df) in enumerate(self.data.items()):
            if 'total_loss' in df.columns:
                loss_data = df['total_loss'].dropna()
                steps = loss_data.index
                
                plt.plot(steps, loss_data.values, 
                        color=self._get_color(i, 'total_loss'),
                        label=f'{config_name} (final: {loss_data.iloc[-1]:.3f})',
                        linewidth=2.5, alpha=0.8)
        
        plt.title('Total Loss Comparison Across Configurations', fontsize=16, fontweight='bold')
        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel('Total Loss', fontsize=12)
        plt.legend(fontsize=10, loc='upper right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / "total_loss_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Total loss comparison saved: {output_file}")
        plt.show()
    
    def compare_dino_losses(self) -> None:
        """Compare DINO loss components"""
        if not self.data:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Global crops loss
        for i, (config_name, df) in enumerate(self.data.items()):
            if 'train/dino_global_crops_loss' in df.columns:
                loss_data = df['train/dino_global_crops_loss'].dropna()
                steps = loss_data.index
                
                ax1.plot(steps, loss_data.values,
                        color=self._get_color(i, 'train/dino_global_crops_loss'),
                        label=f'{config_name} (final: {loss_data.iloc[-1]:.3f})',
                        linewidth=2.5, alpha=0.8)
        
        ax1.set_title('DINO Global Crops Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Loss Value')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Local crops loss
        for i, (config_name, df) in enumerate(self.data.items()):
            if 'train/dino_local_crops_loss' in df.columns:
                loss_data = df['train/dino_local_crops_loss'].dropna()
                steps = loss_data.index
                
                ax2.plot(steps, loss_data.values,
                        color=self._get_color(i, 'train/dino_local_crops_loss'),
                        label=f'{config_name} (final: {loss_data.iloc[-1]:.3f})',
                        linewidth=2.5, alpha=0.8)
        
        ax2.set_title('DINO Local Crops Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Loss Value')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('DINO Loss Components Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_file = self.output_dir / "dino_losses_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"DINO losses comparison saved: {output_file}")
        plt.show()
    
    def compare_ibot_koleo_losses(self) -> None:
        """Compare iBOT and Koleo losses"""
        if not self.data:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # iBOT loss
        for i, (config_name, df) in enumerate(self.data.items()):
            if 'train/ibot_loss' in df.columns:
                loss_data = df['train/ibot_loss'].dropna()
                steps = loss_data.index
                
                ax1.plot(steps, loss_data.values,
                        color=self._get_color(i, 'train/ibot_loss'),
                        label=f'{config_name} (final: {loss_data.iloc[-1]:.3f})',
                        linewidth=2.5, alpha=0.8)
        
        ax1.set_title('iBOT Loss Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('iBOT Loss')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Koleo loss
        for i, (config_name, df) in enumerate(self.data.items()):
            if 'train/koleo_loss' in df.columns:
                loss_data = df['train/koleo_loss'].dropna()
                steps = loss_data.index
                
                ax2.plot(steps, loss_data.values,
                        color=self._get_color(i, 'train/koleo_loss'),
                        label=f'{config_name} (final: {loss_data.iloc[-1]:.3f})',
                        linewidth=2.5, alpha=0.8)
        
        ax2.set_title('Koleo Loss Comparison', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Koleo Loss')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('iBOT and Koleo Loss Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_file = self.output_dir / "ibot_koleo_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"iBOT and Koleo comparison saved: {output_file}")
        plt.show()
    
    def compare_gram_loss(self) -> None:
        """Compare GRAM loss across configurations"""
        if not self.data:
            return
            
        # Check if any configuration has GRAM loss data
        has_gram = any('train/gram_loss' in df.columns for df in self.data.values())
        
        if not has_gram:
            print("⚠️  No GRAM loss data found in any configuration")
            return
            
        plt.figure(figsize=(12, 8))
        
        for i, (config_name, df) in enumerate(self.data.items()):
            if 'train/gram_loss' in df.columns:
                loss_data = df['train/gram_loss'].dropna()
                steps = loss_data.index
                
                plt.plot(steps, loss_data.values,
                        color=self._get_color(i, 'train/gram_loss'),
                        label=f'{config_name} (final: {loss_data.iloc[-1]:.3f})',
                        linewidth=2.5, alpha=0.8)
        
        plt.title('GRAM Loss Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel('GRAM Loss', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        output_file = self.output_dir / "gram_loss_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"GRAM loss comparison saved: {output_file}")
        plt.show()
    
    def compare_auxiliary_losses(self) -> None:
        """Compare all auxiliary losses (iBOT, Koleo, GRAM) in a comprehensive view"""
        if not self.data:
            return
            
        # Check which auxiliary losses are available
        has_ibot = any('train/ibot_loss' in df.columns for df in self.data.values())
        has_koleo = any('train/koleo_loss' in df.columns for df in self.data.values())
        has_gram = any('train/gram_loss' in df.columns for df in self.data.values())
        
        loss_types = []
        if has_ibot:
            loss_types.append(('train/ibot_loss', 'iBOT Loss'))
        if has_koleo:
            loss_types.append(('train/koleo_loss', 'Koleo Loss'))
        if has_gram:
            loss_types.append(('train/gram_loss', 'GRAM Loss'))
            
        if not loss_types:
            print("⚠️  No auxiliary loss data found")
            return
            
        n_plots = len(loss_types)
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
        
        # Handle single subplot case
        if n_plots == 1:
            axes = [axes]
            
        for idx, (loss_key, loss_name) in enumerate(loss_types):
            ax = axes[idx]
            
            for i, (config_name, df) in enumerate(self.data.items()):
                if loss_key in df.columns:
                    loss_data = df[loss_key].dropna()
                    steps = loss_data.index
                    
                    ax.plot(steps, loss_data.values,
                           color=self._get_color(i, loss_key),
                           label=f'{config_name} (final: {loss_data.iloc[-1]:.3f})',
                           linewidth=2.5, alpha=0.8)
            
            ax.set_title(f'{loss_name} Comparison', fontsize=14, fontweight='bold')
            ax.set_xlabel('Training Step')
            ax.set_ylabel(loss_name)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Auxiliary Loss Components Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_file = self.output_dir / "auxiliary_losses_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Auxiliary losses comparison saved: {output_file}")
        plt.show()
    
    def compare_learning_rates(self) -> None:
        """Compare learning rate schedules"""
        if not self.data:
            return
            
        plt.figure(figsize=(12, 8))
        
        for i, (config_name, df) in enumerate(self.data.items()):
            if 'train/lr' in df.columns:
                lr_data = df['train/lr'].dropna()
                steps = lr_data.index
                
                plt.plot(steps, lr_data.values,
                        color=self._get_color(i, 'train/lr'),
                        label=f'{config_name} (final: {lr_data.iloc[-1]:.2e})',
                        linewidth=2.5, alpha=0.8)
        
        plt.title('Learning Rate Schedule Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.yscale('log')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / "learning_rate_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Learning rate comparison saved: {output_file}")
        plt.show()
    
    def generate_performance_summary(self) -> None:
        """Generate performance comparison summary"""
        if not self.data:
            return
            
        print("\n" + "=" * 80)
        print("CONFIGURATION PERFORMANCE COMPARISON")
        print("=" * 80)
        
        summary_data = []
        
        for config_name, df in self.data.items():
            summary = {"Configuration": config_name}
            
            # Total loss
            if 'total_loss' in df.columns:
                total_loss = df['total_loss'].dropna()
                summary["Final Total Loss"] = f"{total_loss.iloc[-1]:.4f}"
                summary["Min Total Loss"] = f"{total_loss.min():.4f}"
            
            # DINO losses
            if 'train/dino_global_crops_loss' in df.columns:
                global_loss = df['train/dino_global_crops_loss'].dropna()
                summary["Final Global Loss"] = f"{global_loss.iloc[-1]:.4f}"
            
            if 'train/dino_local_crops_loss' in df.columns:
                local_loss = df['train/dino_local_crops_loss'].dropna()
                summary["Final Local Loss"] = f"{local_loss.iloc[-1]:.4f}"
            
            # iBOT loss
            if 'train/ibot_loss' in df.columns:
                ibot_loss = df['train/ibot_loss'].dropna()
                summary["Final iBOT Loss"] = f"{ibot_loss.iloc[-1]:.4f}"
            
            # GRAM loss
            if 'train/gram_loss' in df.columns:
                gram_loss = df['train/gram_loss'].dropna()
                summary["Final GRAM Loss"] = f"{gram_loss.iloc[-1]:.4f}"
            
            # Training steps
            summary["Training Steps"] = len(df)
            
            # Learning rate
            if 'train/lr' in df.columns:
                lr = df['train/lr'].dropna()
                summary["Final LR"] = f"{lr.iloc[-1]:.2e}"
            
            summary_data.append(summary)
        
        # Create and display summary table
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Save summary to CSV
        summary_file = self.output_dir / "performance_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\nPerformance summary saved: {summary_file}")
        print("=" * 80)
    
    def create_all_comparisons(self) -> None:
        """Generate all comparison plots and summaries"""
        print("Creating comprehensive configuration comparisons...")
        
        self.compare_total_losses()
        self.compare_dino_losses() 
        self.compare_ibot_koleo_losses()
        self.compare_gram_loss()
        self.compare_auxiliary_losses()
        self.compare_learning_rates()
        self.generate_performance_summary()
        
        print(f"\nAll comparison plots saved to: {self.output_dir}")