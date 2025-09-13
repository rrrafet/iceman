"""
Portfolio Visualization
=======================

Visualization utilities for portfolio graph analysis and presentation.
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import warnings

# Optional dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    sns = None

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from .graph import PortfolioGraph


class PortfolioVisualizer:
    """Visualization utilities for portfolio graph"""
    
    def __init__(self, portfolio_graph: PortfolioGraph):
        self.graph = portfolio_graph
        self.nx_graph = self._build_networkx_graph() if HAS_NETWORKX else None
    
    def _build_networkx_graph(self) -> Optional['nx.DiGraph']:
        """Convert to NetworkX graph for visualization"""
        if not HAS_NETWORKX:
            return None
            
        G = nx.DiGraph()
        
        # Add nodes
        for cid, component in self.graph.components.items():
            node_type = 'leaf' if component.is_leaf() else 'node'
            G.add_node(cid, 
                      label=component.name,
                      type=node_type,
                      component_type=component.component_type)
        
        # Add edges
        for parent_id, component in self.graph.components.items():
            if not component.is_leaf():
                for child_id in component.get_all_children():
                    if child_id in self.graph.components:
                        # Get allocation tilt from metric store
                        edge_key = f"{parent_id}->{child_id}"
                        allocation_tilt_metric = self.graph.metric_store.get_metric(edge_key, 'allocation_tilt')
                        allocation_tilt = allocation_tilt_metric.value() if allocation_tilt_metric else 0.0
                        G.add_edge(parent_id, child_id, weight=allocation_tilt)
        
        return G
    
    def _check_matplotlib(self):
        """Check if matplotlib is available"""
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for this visualization. "
                            "Install with: pip install matplotlib seaborn")
    
    def _check_networkx(self):
        """Check if networkx is available"""
        if not HAS_NETWORKX:
            raise ImportError("networkx is required for this visualization. "
                            "Install with: pip install networkx")
    
    def plot_hierarchy(self, 
                      figsize: Tuple[int, int] = (12, 8),
                      layout: str = 'hierarchical',
                      show_weights: bool = True,
                      show_tilts: bool = True) -> 'plt.Figure':
        """Plot the portfolio hierarchy"""
        self._check_matplotlib()
        self._check_networkx()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Choose layout
        if layout == 'hierarchical':
            pos = self._hierarchical_layout()
        elif layout == 'spring':
            pos = nx.spring_layout(self.nx_graph, k=1, iterations=50)
        else:
            pos = nx.kamada_kawai_layout(self.nx_graph)
        
        # Node colors based on type
        node_colors = []
        for node in self.nx_graph.nodes():
            if self.nx_graph.nodes[node]['type'] == 'leaf':
                node_colors.append('lightblue')
            else:
                node_colors.append('lightcoral')
        
        # Draw nodes
        nx.draw_networkx_nodes(self.nx_graph, pos, 
                              node_color=node_colors, 
                              node_size=1000, 
                              alpha=0.7, ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(self.nx_graph, pos, 
                              edge_color='gray', 
                              arrows=True, 
                              arrowsize=20, 
                              alpha=0.6, ax=ax)
        
        # Labels
        labels = {node: self.nx_graph.nodes[node]['label'] 
                 for node in self.nx_graph.nodes()}
        nx.draw_networkx_labels(self.nx_graph, pos, labels, 
                               font_size=8, ax=ax)
        
        # Edge labels (allocation tilts)
        if show_tilts:
            edge_labels = {(u, v): f"{d['weight']:.2f}" 
                          for u, v, d in self.nx_graph.edges(data=True) 
                          if d['weight'] != 0}
            nx.draw_networkx_edge_labels(self.nx_graph, pos, edge_labels, 
                                        font_size=6, ax=ax)
        
        ax.set_title("Portfolio Hierarchy", fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                      markersize=10, label='Portfolio Node'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                      markersize=10, label='Portfolio Leaf')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        return fig
    
    def _hierarchical_layout(self) -> Dict[str, Tuple[float, float]]:
        """Create hierarchical layout positions"""
        if not self.graph.root_id:
            return nx.spring_layout(self.nx_graph)
        
        # BFS to assign levels
        levels = {}
        queue = deque([(self.graph.root_id, 0)])
        visited = set()
        
        while queue:
            node, level = queue.popleft()
            if node in visited:
                continue
            
            visited.add(node)
            levels[node] = level
            
            # Add children
            component = self.graph.components[node]
            for child_id in component.get_all_children():
                if child_id in self.graph.components and child_id not in visited:
                    queue.append((child_id, level + 1))
        
        # Group nodes by level
        level_groups = defaultdict(list)
        for node, level in levels.items():
            level_groups[level].append(node)
        
        # Assign positions
        pos = {}
        for level, nodes in level_groups.items():
            y_pos = -level  # Higher levels at top
            for i, node in enumerate(nodes):
                x_pos = i - len(nodes) / 2  # Center nodes horizontally
                pos[node] = (x_pos, y_pos)
        
        return pos
    
    def plot_risk_return_scatter(self, 
                                metric_type: str = 'portfolio',
                                figsize: Tuple[int, int] = (10, 6)) -> 'plt.Figure':
        """Plot risk-return scatter for all components"""
        self._check_matplotlib()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        risks = []
        returns = []
        labels = []
        colors = []
        
        for cid, component in self.graph.components.items():
            # Get metrics from metric store
            risk_metric = self.graph.metric_store.get_metric(cid, 'forward_risk')
            return_metric = self.graph.metric_store.get_metric(cid, 'forward_returns')
            
            risk = risk_metric.value() if risk_metric else 0.0
            return_val = return_metric.value() if return_metric else 0.0
            
            risks.append(risk)
            returns.append(return_val)
            labels.append(component.name)
            colors.append('blue' if component.is_leaf() else 'red')
        
        scatter = ax.scatter(risks, returns, c=colors, alpha=0.6, s=100)
        
        # Add labels
        for i, label in enumerate(labels):
            ax.annotate(label, (risks[i], returns[i]), 
                       xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, alpha=0.7)
        
        ax.set_xlabel('Risk Expectation')
        ax.set_ylabel('Forward Returns')
        ax.set_title(f'{metric_type.title()} Risk-Return Scatter')
        ax.grid(True, alpha=0.3)
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markersize=10, label='Portfolio Node'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                      markersize=10, label='Portfolio Leaf')
        ]
        ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        return fig
    
    def plot_weight_distribution(self, figsize: Tuple[int, int] = (10, 6)) -> 'plt.Figure':
        """Plot weight distribution across portfolio components"""
        self._check_matplotlib()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        names = []
        weights = []
        colors = []
        
        for cid, component in self.graph.components.items():
            names.append(component.name)
            # Get weight from metric store
            weight_metric = self.graph.metric_store.get_metric(cid, 'weight')
            weight = weight_metric.value() if weight_metric else 0.0
            weights.append(weight)
            colors.append('lightblue' if component.is_leaf() else 'lightcoral')
        
        bars = ax.bar(range(len(names)), weights, color=colors, alpha=0.7)
        
        ax.set_xlabel('Portfolio Components')
        ax.set_ylabel('Weight')
        ax.set_title('Portfolio Weight Distribution')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, weight in zip(bars, weights):
            height = bar.get_height()
            ax.annotate(f'{weight:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def plot_allocation_tilts(self, figsize: Tuple[int, int] = (12, 6)) -> 'plt.Figure':
        """Plot allocation tilts across the hierarchy"""
        self._check_matplotlib()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        parent_names = []
        child_names = []
        tilts = []
        
        for parent_id, component in self.graph.components.items():
            if not component.is_leaf():
                for child_id in component.get_all_children():
                    if child_id in self.graph.components:
                        # Get allocation tilt from metric store
                        edge_key = f"{parent_id}->{child_id}"
                        allocation_tilt_metric = self.graph.metric_store.get_metric(edge_key, 'allocation_tilt')
                        tilt = allocation_tilt_metric.value() if allocation_tilt_metric else 0.0
                        
                        parent_names.append(component.name)
                        child_names.append(self.graph.components[child_id].name)
                        tilts.append(tilt)
        
        # Create position mapping
        edge_labels = [f"{p}->{c}" for p, c in zip(parent_names, child_names)]
        
        # Color based on positive/negative tilts
        colors = ['green' if tilt > 0 else 'red' if tilt < 0 else 'gray' for tilt in tilts]
        
        bars = ax.bar(range(len(tilts)), tilts, color=colors, alpha=0.7)
        
        ax.set_xlabel('Parent -> Child Relationships')
        ax.set_ylabel('Allocation Tilt')
        ax.set_title('Allocation Tilts Across Portfolio Hierarchy')
        ax.set_xticks(range(len(edge_labels)))
        ax.set_xticklabels(edge_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels on bars
        for bar, tilt in zip(bars, tilts):
            height = bar.get_height()
            ax.annotate(f'{tilt:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3 if height >= 0 else -15),
                       textcoords="offset points",
                       ha='center', va='bottom' if height >= 0 else 'top', 
                       fontsize=8)
        
        plt.tight_layout()
        return fig
    
    # Plotly alternatives (using available dependencies)
    def get_risk_return_data(self, metric_type: str = 'portfolio') -> Dict:
        """Get risk-return data for visualization"""
        data = {
            'names': [],
            'risks': [],
            'returns': [],
            'weights': [],
            'types': [],
            'component_types': []
        }
        
        for cid, component in self.graph.components.items():
            # Get metrics from metric store
            risk_metric = self.graph.metric_store.get_metric(cid, 'forward_risk')
            return_metric = self.graph.metric_store.get_metric(cid, 'forward_returns')
            weight_metric = self.graph.metric_store.get_metric(cid, 'weight')
            
            risk = risk_metric.value() if risk_metric else 0.0
            return_val = return_metric.value() if return_metric else 0.0
            weight = weight_metric.value() if weight_metric else 0.0
            
            data['names'].append(component.name)
            data['risks'].append(risk)
            data['returns'].append(return_val)
            data['weights'].append(weight)
            data['types'].append('Leaf' if component.is_leaf() else 'Node')
            data['component_types'].append(component.component_type)
        
        return data
    
    def get_weight_data(self) -> Dict:
        """Get weight distribution data for visualization"""
        data = {
            'names': [],
            'weights': [],
            'types': []
        }
        
        for cid, component in self.graph.components.items():
            data['names'].append(component.name)
            # Get weight from metric store
            weight_metric = self.graph.metric_store.get_metric(cid, 'weight')
            weight = weight_metric.value() if weight_metric else 0.0
            data['weights'].append(weight)
            data['types'].append('Leaf' if component.is_leaf() else 'Node')
        
        return data
    
    def get_allocation_tilt_data(self) -> Dict:
        """Get allocation tilt data for visualization"""
        data = {
            'parent_names': [],
            'child_names': [],
            'tilts': [],
            'edge_labels': []
        }
        
        for parent_id, component in self.graph.components.items():
            if not component.is_leaf():
                for child_id in component.get_all_children():
                    if child_id in self.graph.components:
                        # Get allocation tilt from metric store
                        edge_key = f"{parent_id}->{child_id}"
                        allocation_tilt_metric = self.graph.metric_store.get_metric(edge_key, 'allocation_tilt')
                        tilt = allocation_tilt_metric.value() if allocation_tilt_metric else 0.0
                        
                        parent_name = component.name
                        child_name = self.graph.components[child_id].name
                        data['parent_names'].append(parent_name)
                        data['child_names'].append(child_name)
                        data['tilts'].append(tilt)
                        data['edge_labels'].append(f"{parent_name} -> {child_name}")
        
        return data
    
    def plot_risk_return_plotly(self, metric_type: str = 'portfolio'):
        """Create risk-return scatter plot using plotly (if available)"""
        if not HAS_PLOTLY:
            raise ImportError("plotly is required for this visualization. "
                            "Install with: pip install plotly")
        
        data = self.get_risk_return_data(metric_type)
        
        fig = px.scatter(
            x=data['risks'],
            y=data['returns'],
            size=data['weights'],
            color=data['types'],
            hover_name=data['names'],
            hover_data={'component_type': data['component_types']},
            labels={
                'x': 'Risk Expectation',
                'y': 'Forward Returns',
                'color': 'Component Type'
            },
            title=f'{metric_type.title()} Risk-Return Profile'
        )
        
        return fig
    
    def plot_weights_plotly(self):
        """Create weight distribution plot using plotly (if available)"""
        if not HAS_PLOTLY:
            raise ImportError("plotly is required for this visualization. "
                            "Install with: pip install plotly")
        
        data = self.get_weight_data()
        
        fig = px.bar(
            x=data['names'],
            y=data['weights'],
            color=data['types'],
            title='Portfolio Weight Distribution',
            labels={'x': 'Components', 'y': 'Weight'}
        )
        
        return fig