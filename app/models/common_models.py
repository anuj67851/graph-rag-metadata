from pydantic import BaseModel, Field
from typing import List, Dict, Any

class Node(BaseModel):
    """
    Represents a node in the knowledge graph for API responses and visualizations.
    """
    id: str = Field(..., description="Unique identifier for the node, typically the canonical_name.")
    label: str = Field(..., description="Display label for the node, often same as id or a concise name.")
    type: str = Field(..., description="Entity type of the node (e.g., PERSON, ORGANIZATION).")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties of the node, such as aliases, contexts, source_document.")
    # Example properties content:
    # {
    #   "aliases": ["Alias1", "Alias2"],
    #   "contexts": ["Sentence 1 about node.", "Sentence 2 about node."],
    #   "source_document": "document_name.pdf",
    #   "description": "A longer text description if available"
    #   # ... other dynamic properties
    # }

class Edge(BaseModel):
    """
    Represents an edge (relationship) in the knowledge graph for API responses and visualizations.
    """
    source: str = Field(..., description="Identifier of the source node (canonical_name).")
    target: str = Field(..., description="Identifier of the target node (canonical_name).")
    label: str = Field(..., description="Type of the relationship (e.g., WORKS_FOR, LOCATED_IN).")
    properties: Dict[Any, Any] = Field(default_factory=dict, description="Additional properties of the edge, such as contexts or source_document.")
    # Example properties content:
    # {
    #   "contexts": ["Sentence establishing this relationship."],
    #   "source_document": "document_name.pdf",
    #   "weight": 0.8  # Optional, if applicable
    # }

class Subgraph(BaseModel):
    """
    Represents a subgraph consisting of nodes and edges, typically for context or visualization.
    """
    nodes: List[Node] = Field(default_factory=list, description="List of nodes in the subgraph.")
    edges: List[Edge] = Field(default_factory=list, description="List of edges in the subgraph.")

    def is_empty(self) -> bool:
        return not self.nodes and not self.edges