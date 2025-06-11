from pydantic import BaseModel, Field
from typing import List, Optional

class ExtractedEntity(BaseModel):
    """
    Represents a single entity extracted by the LLM from a text chunk.
    Corresponds to the structure defined in Feature 1 and refined in Feature 6 prompts.
    """
    original_mention: str = Field(..., description="The exact text span from the document mentioning the entity.")
    entity_type: str = Field(..., description="The type of the entity (e.g., PERSON, ORGANIZATION).")
    canonical_name: str = Field(..., description="A standardized, common name for the entity.")
    contexts: List[str] = Field(default_factory=list, description="List of context sentences from the document that describe or define this entity.")
    # Optional: Could add a confidence score if the LLM provides it
    # confidence_score: Optional[float] = Field(None, ge=0, le=1)

class ExtractedRelationship(BaseModel):
    """
    Represents a single relationship extracted by the LLM from a text chunk.
    Corresponds to the structure defined in Feature 1 and refined in Feature 6 prompts.
    """
    source_canonical_name: str = Field(..., description="The canonical_name of the source entity of the relationship.")
    relationship_type: str = Field(..., description="The type of the relationship (e.g., WORKS_FOR, LOCATED_IN).")
    target_canonical_name: str = Field(..., description="The canonical_name of the target entity of the relationship.")
    contexts: List[str] = Field(default_factory=list, description="List of context sentences from the document that establish this relationship.")
    # Optional: Could add a confidence score if the LLM provides it
    # confidence_score: Optional[float] = Field(None, ge=0, le=1)

class LLMExtractionOutput(BaseModel):
    """
    Defines the expected JSON structure from the LLM after processing a text chunk for entities and relationships.
    """
    entities: List[ExtractedEntity] = Field(default_factory=list)
    relationships: List[ExtractedRelationship] = Field(default_factory=list)

class IngestionStatus(BaseModel):
    """
    Represents the status of an ingestion task for a file.
    """
    filename: str
    status: str # e.g., "Processing", "Completed", "Failed"
    message: Optional[str] = None
    entities_added: int = 0
    relationships_added: int = 0