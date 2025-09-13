"""
Portfolio Component Classes
===========================

Abstract base class and concrete implementations for portfolio hierarchy components.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, TYPE_CHECKING
from .metrics import MetricStore

if TYPE_CHECKING:
    from .visitors import PortfolioVisitor
    from .graph import PortfolioGraph
    from .metrics import Metric


class PortfolioComponent(ABC):
    """Abstract base class for all portfolio components"""
    
    def __init__(self, 
                 component_id: str, 
                 name: str, 
                 component_type: Optional[str] = None,
                 metric_store: Optional[MetricStore] = None):
        self.component_id = component_id
        self.name = name
        self.component_type = component_type
        
        # Metric store support
        self.metric_store = metric_store
        
        # Graph relationships
        self.parent_ids: Set[str] = set()
        self.metadata: Dict = {}  # For extensible custom data
        
        # Store reference to parent graph for component lookup
        self._parent_graph: Optional['PortfolioGraph'] = None
        
        # Overlay strategy support
        self.is_overlay: bool = False

    
    @abstractmethod
    def get_all_children(self) -> List[str]:
        """Get all child component IDs"""
        pass
    
    @abstractmethod
    def is_leaf(self) -> bool:
        """Check if component is a leaf node"""
        pass
    
    def accept(self, visitor: 'PortfolioVisitor') -> None:
        """Accept a visitor for traversal (Visitor pattern)"""
        if self.is_leaf():
            visitor.visit_leaf(self)
        else:
            visitor.visit_node(self)
    
    def _get_child_component(self, child_id: str) -> Optional['PortfolioComponent']:
        """Get child component by ID (requires parent graph reference)"""
        if self._parent_graph and child_id in self._parent_graph.components:
            return self._parent_graph.components[child_id]
        return None
    
    def set_parent_graph(self, graph: 'PortfolioGraph') -> None:
        """Set reference to parent graph for component lookup"""
        self._parent_graph = graph
    
    def get_metric(self, metric_name: str) -> Optional['Metric']:
        """Get a metric for this component from the metric store"""
        if self.metric_store:
            return self.metric_store.get_metric(self.component_id, metric_name)
        return None
    
    def set_metric(self, metric_name: str, metric: 'Metric') -> None:
        """Set a metric for this component in the metric store"""
        if self.metric_store:
            self.metric_store.set_metric(self.component_id, metric_name, metric)
    
    def get_operational_weight(self, weight_metric_name: str = 'portfolio_weight') -> float:
        """Get operational weight for this component (1.0 for overlays)"""
        if self.is_overlay:
            # For overlay components, operational weight is always 1.0
            return 1.0
        else:
            # For non-overlay components, use their own weight
            metric = self.get_metric(weight_metric_name)
            return metric.value() if metric else 1.0


class PortfolioNode(PortfolioComponent):
    """Intermediate node in the portfolio hierarchy"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.children_ids: Set[str] = set()
    
    def add_child(self, child_id: str):
        """Add a child component"""
        self.children_ids.add(child_id)
    
    def remove_child(self, child_id: str):
        """Remove a child component"""
        self.children_ids.discard(child_id)
    
    def get_all_children(self) -> List[str]:
        return list(self.children_ids)
    
    def is_leaf(self) -> bool:
        return False
    
    def _get_child_component(self, child_id: str) -> Optional['PortfolioComponent']:
        """Get child component by ID (overrides base implementation)"""
        if self._parent_graph and child_id in self._parent_graph.components:
            return self._parent_graph.components[child_id]
        return None


class PortfolioLeaf(PortfolioComponent):
    """Terminal node representing an individual portfolio/security"""
    
    def get_all_children(self) -> List[str]:
        return []
    
    def is_leaf(self) -> bool:
        return True