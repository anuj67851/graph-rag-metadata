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
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties of the edge, such as contexts or source_document.")
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

if __name__ == "__main__":
    # Example Usage
    node1_props = {
        "aliases": ["Big Corp", "BCI"],
        "contexts": ["Big Corporation Inc. is a multinational company.", "Founded in 1990."],
        "source_document": "annual_report_2023.pdf",
        "industry": "Technology"
    }
    node1 = Node(id="Big Corporation Inc.", label="Big Corporation Inc.", type="ORGANIZATION", properties=node1_props)

    node2_props = {
        "title": "CEO",
        "contexts": ["John Doe is the CEO of Big Corporation Inc."],
        "source_document": "annual_report_2023.pdf"
    }
    node2 = Node(id="John Doe", label="John Doe", type="PERSON", properties=node2_props)

    edge1_props = {
        "contexts": ["John Doe has been the CEO of Big Corporation Inc. since 2015."],
        "source_document": "annual_report_2023.pdf",
        "start_year": 2015
    }
    edge1 = Edge(source="John Doe", target="Big Corporation Inc.", label="WORKS_FOR", properties=edge1_props)

    subgraph_example = Subgraph(nodes=[node1, node2], edges=[edge1])

    print("--- Node 1 ---")
    print(node1.model_dump_json(indent=2))

    print("\n--- Node 2 ---")
    print(node2.model_dump_json(indent=2))

    print("\n--- Edge 1 ---")
    print(edge1.model_dump_json(indent=2))

    print("\n--- Subgraph ---")
    print(subgraph_example.model_dump_json(indent=2))
    print(f"\nSubgraph empty? {subgraph_example.is_empty()}")

    empty_subgraph = Subgraph()
    print(f"Empty subgraph empty? {empty_subgraph.is_empty()}")