"""
Decision Attribution Visualization
==================================

Visualization tools for decision attribution results, extending the existing
plotting capabilities to show decision impact analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import matplotlib.pyplot as plt
import seaborn as sns

if TYPE_CHECKING:
    from .decision_attribution import DecisionRiskAttribution


class DecisionAttributionVisualizer:
    """
    Visualizes decision attribution results using matplotlib/seaborn.
    
    Provides visualization for:
    - Risk impact summaries
    - Component-level attribution breakdowns
    - Before/after comparisons
    - Hierarchical impact maps
    """
    
    def __init__(self, attribution: 'DecisionRiskAttribution'):
        """
        Initialize visualizer with attribution results.
        
        Parameters
        ----------
        attribution : DecisionRiskAttribution
            Attribution results to visualize
        """
        self.attribution = attribution
        
    def plot_risk_impact_summary(self, figsize: tuple = (12, 8)) -> plt.Figure:
        """
        Create comprehensive risk impact summary plot.
        
        Parameters
        ----------
        figsize : tuple, default (12, 8)
            Figure size
            
        Returns
        -------
        plt.Figure
            Figure containing the plots
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Decision Risk Attribution: {self.attribution.decision.description}', 
                    fontsize=14, fontweight='bold')
        
        # Plot 1: Before/After Risk Comparison
        ax1 = axes[0, 0]
        risks = [self.attribution.total_active_risk_before, self.attribution.total_active_risk_after]
        labels = ['Before', 'After']
        colors = ['lightblue', 'lightcoral' if self.attribution.total_active_risk_change > 0 else 'lightgreen']
        
        bars = ax1.bar(labels, risks, color=colors)
        ax1.set_title('Total Active Risk')
        ax1.set_ylabel('Risk')
        
        # Add change annotation
        change = self.attribution.total_active_risk_change
        ax1.annotate(f'Δ = {change:+.6f}', 
                    xy=(0.5, max(risks) * 0.9), 
                    ha='center', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # Plot 2: Risk Decomposition Changes
        ax2 = axes[0, 1]
        decomp_data = {
            'Allocation\nFactor': self.attribution.allocation_factor_risk_change,
            'Allocation\nSpecific': self.attribution.allocation_specific_risk_change,
            'Selection\nFactor': self.attribution.selection_factor_risk_change,
            'Selection\nSpecific': self.attribution.selection_specific_risk_change
        }
        
        x_pos = range(len(decomp_data))
        values = list(decomp_data.values())
        colors = ['red' if v < 0 else 'green' for v in values]
        
        bars = ax2.bar(x_pos, values, color=colors, alpha=0.7)
        ax2.set_title('Risk Decomposition Changes')
        ax2.set_ylabel('Risk Change')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(decomp_data.keys(), rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Plot 3: Top Component Contributors
        ax3 = axes[1, 0]
        if self.attribution.component_attributions:
            top_contributors = self.attribution.get_top_risk_contributors(5)
            comp_names = [comp_id for comp_id, _ in top_contributors]
            comp_values = [self.attribution.component_attributions[comp_id].total_risk_contribution_change 
                          for comp_id, _ in top_contributors]
            
            colors = ['red' if v < 0 else 'green' for v in comp_values]
            bars = ax3.barh(comp_names, comp_values, color=colors, alpha=0.7)
            ax3.set_title('Top Component Risk Changes')
            ax3.set_xlabel('Risk Change')
            ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No component data available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Component Risk Changes')
        
        # Plot 4: Factor Exposure Changes
        ax4 = axes[1, 1]
        if self.attribution.factor_exposure_changes:
            factor_names = list(self.attribution.factor_exposure_changes.keys())
            factor_changes = list(self.attribution.factor_exposure_changes.values())
            
            colors = ['red' if v < 0 else 'green' for v in factor_changes]
            bars = ax4.bar(factor_names, factor_changes, color=colors, alpha=0.7)
            ax4.set_title('Factor Exposure Changes')
            ax4.set_ylabel('Exposure Change')
            ax4.tick_params(axis='x', rotation=45)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No factor exposure data available',
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Factor Exposure Changes')
        
        plt.tight_layout()
        return fig
    
    def plot_component_attribution_details(self, figsize: tuple = (14, 10)) -> plt.Figure:
        """
        Create detailed component attribution plot.
        
        Parameters
        ----------
        figsize : tuple, default (14, 10)
            Figure size
            
        Returns
        -------
        plt.Figure
            Figure containing the detailed attribution plots
        """
        if not self.attribution.component_attributions:
            raise ValueError("No component attributions available for plotting")
        
        # Convert to DataFrame for easier plotting
        df = self.attribution.to_dataframe()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Component Attribution Details: {self.attribution.decision.description}', 
                    fontsize=14, fontweight='bold')
        
        # Plot 1: Total Risk Contribution Changes
        ax1 = axes[0, 0]
        colors = ['red' if v < 0 else 'green' for v in df['total_risk_contribution_change']]
        bars = ax1.barh(df.index, df['total_risk_contribution_change'], color=colors, alpha=0.7)
        ax1.set_title('Total Risk Contribution Changes')
        ax1.set_xlabel('Risk Change')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Plot 2: Allocation vs Selection Breakdown
        ax2 = axes[0, 1]
        x = np.arange(len(df))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, df['allocation_contribution_change'], width, 
                       label='Allocation', alpha=0.7, color='skyblue')
        bars2 = ax2.bar(x + width/2, df['selection_contribution_change'], width,
                       label='Selection', alpha=0.7, color='lightcoral')
        
        ax2.set_title('Allocation vs Selection Changes')
        ax2.set_ylabel('Risk Change')
        ax2.set_xticks(x)
        ax2.set_xticklabels(df.index, rotation=45, ha='right')
        ax2.legend()
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Plot 3: Weight Changes
        ax3 = axes[1, 0]
        x = np.arange(len(df))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, df['portfolio_weight_change'], width,
                       label='Portfolio', alpha=0.7, color='green')
        bars2 = ax3.bar(x + width/2, df['active_weight_change'], width,
                       label='Active', alpha=0.7, color='orange')
        
        ax3.set_title('Weight Changes')
        ax3.set_ylabel('Weight Change')
        ax3.set_xticks(x)
        ax3.set_xticklabels(df.index, rotation=45, ha='right')
        ax3.legend()
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Plot 4: Factor vs Specific Breakdown
        ax4 = axes[1, 1]
        
        # Stack allocation and selection for factor vs specific
        alloc_factor = df['allocation_factor_change']
        alloc_specific = df['allocation_specific_change']
        select_factor = df['selection_factor_change'] 
        select_specific = df['selection_specific_change']
        
        # Create stacked bars
        x = np.arange(len(df))
        ax4.bar(x, alloc_factor, label='Alloc Factor', alpha=0.7, color='darkblue')
        ax4.bar(x, alloc_specific, bottom=alloc_factor, label='Alloc Specific', alpha=0.7, color='lightblue')
        ax4.bar(x, select_factor, bottom=alloc_factor + alloc_specific, 
               label='Select Factor', alpha=0.7, color='darkred')
        ax4.bar(x, select_specific, bottom=alloc_factor + alloc_specific + select_factor,
               label='Select Specific', alpha=0.7, color='lightcoral')
        
        ax4.set_title('Factor vs Specific Risk Changes')
        ax4.set_ylabel('Risk Change')
        ax4.set_xticks(x)
        ax4.set_xticklabels(df.index, rotation=45, ha='right')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_decision_impact_waterfall(self, figsize: tuple = (12, 6)) -> plt.Figure:
        """
        Create waterfall chart showing cumulative risk impact.
        
        Parameters
        ----------
        figsize : tuple, default (12, 6)
            Figure size
            
        Returns
        -------
        plt.Figure
            Waterfall chart figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Data for waterfall chart
        categories = ['Starting\nRisk', 'Allocation\nFactor', 'Allocation\nSpecific', 
                     'Selection\nFactor', 'Selection\nSpecific', 'Ending\nRisk']
        
        values = [
            self.attribution.total_active_risk_before,
            self.attribution.allocation_factor_risk_change,
            self.attribution.allocation_specific_risk_change,
            self.attribution.selection_factor_risk_change,
            self.attribution.selection_specific_risk_change,
            0  # Placeholder for ending risk
        ]
        
        # Calculate cumulative values
        cumulative = [values[0]]
        for i in range(1, len(values)-1):
            cumulative.append(cumulative[-1] + values[i])
        cumulative.append(self.attribution.total_active_risk_after)
        
        # Plot bars
        colors = ['blue'] + ['red' if v < 0 else 'green' for v in values[1:-1]] + ['blue']
        
        for i, (cat, val, cum) in enumerate(zip(categories, values, cumulative)):
            if i == 0 or i == len(categories) - 1:
                # Starting and ending bars
                ax.bar(i, cum, color=colors[i], alpha=0.7)
                ax.text(i, cum/2, f'{cum:.4f}', ha='center', va='center', fontweight='bold')
            else:
                # Change bars
                if val >= 0:
                    ax.bar(i, val, bottom=cumulative[i-1], color=colors[i], alpha=0.7)
                    ax.text(i, cumulative[i-1] + val/2, f'{val:+.4f}', ha='center', va='center')
                else:
                    ax.bar(i, -val, bottom=cum, color=colors[i], alpha=0.7)
                    ax.text(i, cum - val/2, f'{val:+.4f}', ha='center', va='center')
                
                # Connect bars with lines
                if i > 0:
                    ax.plot([i-0.4, i-0.4], [cumulative[i-1], cumulative[i-1]], 'k--', alpha=0.5)
                    ax.plot([i-0.4, i+0.4], [cumulative[i-1], cumulative[i-1]], 'k--', alpha=0.5)
        
        ax.set_title(f'Risk Attribution Waterfall: {self.attribution.decision.description}')
        ax.set_ylabel('Active Risk')
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def create_summary_report(self, save_path: Optional[str] = None) -> List[plt.Figure]:
        """
        Create comprehensive summary report with all visualizations.
        
        Parameters
        ----------
        save_path : str, optional
            Directory path to save figures. If None, figures are not saved.
            
        Returns
        -------
        List[plt.Figure]
            List of created figures
        """
        figures = []
        
        # Create all plots
        fig1 = self.plot_risk_impact_summary()
        figures.append(fig1)
        
        if self.attribution.component_attributions:
            fig2 = self.plot_component_attribution_details()
            figures.append(fig2)
        
        fig3 = self.plot_decision_impact_waterfall()
        figures.append(fig3)
        
        # Save figures if path provided
        if save_path:
            import os
            os.makedirs(save_path, exist_ok=True)
            
            decision_name = self.attribution.decision.component_id
            timestamp = self.attribution.timestamp.strftime('%Y%m%d_%H%M%S')
            
            for i, fig in enumerate(figures, 1):
                filename = f"{decision_name}_attribution_{i}_{timestamp}.png"
                filepath = os.path.join(save_path, filename)
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"Saved: {filepath}")
        
        return figures


# Convenience functions
def visualize_decision_attribution(attribution: 'DecisionRiskAttribution', 
                                 show_plots: bool = True,
                                 save_path: Optional[str] = None) -> List[plt.Figure]:
    """
    Quick visualization of decision attribution results.
    
    Parameters
    ----------
    attribution : DecisionRiskAttribution
        Attribution results to visualize
    show_plots : bool, default True
        Whether to display plots
    save_path : str, optional
        Directory to save plots
        
    Returns
    -------
    List[plt.Figure]
        Created figures
    """
    visualizer = DecisionAttributionVisualizer(attribution)
    figures = visualizer.create_summary_report(save_path=save_path)
    
    if show_plots:
        plt.show()
    
    return figures


def compare_decision_attributions(attributions: List['DecisionRiskAttribution'],
                                decision_names: Optional[List[str]] = None,
                                figsize: tuple = (14, 8)) -> plt.Figure:
    """
    Compare multiple decision attribution results side by side.
    
    Parameters
    ----------
    attributions : List[DecisionRiskAttribution]
        List of attribution results to compare
    decision_names : List[str], optional
        Names for each decision. If None, uses decision descriptions.
    figsize : tuple, default (14, 8)
        Figure size
        
    Returns
    -------
    plt.Figure
        Comparison figure
    """
    if not attributions:
        raise ValueError("At least one attribution result is required")
    
    if decision_names is None:
        decision_names = [attr.decision.description for attr in attributions]
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Decision Attribution Comparison', fontsize=14, fontweight='bold')
    
    # Prepare data
    risk_changes = [attr.total_active_risk_change for attr in attributions]
    alloc_changes = [attr.total_allocation_risk_change for attr in attributions]
    select_changes = [attr.total_selection_risk_change for attr in attributions]
    factor_changes = [attr.total_factor_risk_change for attr in attributions]
    specific_changes = [attr.total_specific_risk_change for attr in attributions]
    
    # Plot 1: Total Risk Changes
    ax1 = axes[0, 0]
    colors = ['red' if v < 0 else 'green' for v in risk_changes]
    bars = ax1.bar(decision_names, risk_changes, color=colors, alpha=0.7)
    ax1.set_title('Total Risk Changes')
    ax1.set_ylabel('Risk Change')
    ax1.tick_params(axis='x', rotation=45)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Plot 2: Allocation vs Selection
    ax2 = axes[0, 1]
    x = np.arange(len(decision_names))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, alloc_changes, width, label='Allocation', alpha=0.7)
    bars2 = ax2.bar(x + width/2, select_changes, width, label='Selection', alpha=0.7)
    
    ax2.set_title('Allocation vs Selection')
    ax2.set_ylabel('Risk Change')
    ax2.set_xticks(x)
    ax2.set_xticklabels(decision_names, rotation=45, ha='right')
    ax2.legend()
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Plot 3: Factor vs Specific
    ax3 = axes[1, 0]
    bars1 = ax3.bar(x - width/2, factor_changes, width, label='Factor', alpha=0.7)
    bars2 = ax3.bar(x + width/2, specific_changes, width, label='Specific', alpha=0.7)
    
    ax3.set_title('Factor vs Specific')
    ax3.set_ylabel('Risk Change')
    ax3.set_xticks(x)
    ax3.set_xticklabels(decision_names, rotation=45, ha='right')
    ax3.legend()
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Plot 4: Component Count Comparison
    ax4 = axes[1, 1]
    component_counts = [len(attr.component_attributions) for attr in attributions]
    bars = ax4.bar(decision_names, component_counts, alpha=0.7, color='skyblue')
    ax4.set_title('Components Affected')
    ax4.set_ylabel('Number of Components')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig