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

if __name__ == "__main__":
    # Example Usage
    entity1 = ExtractedEntity(
        original_mention="Dr. Aris Thorne",
        entity_type="PERSON",
        canonical_name="Aris Thorne",
        contexts=["The CEO, Dr. Aris Thorne, attributed this success...", "Dr. Thorne, who previously worked at Beta Innovations..."]
    )

    entity2 = ExtractedEntity(
        original_mention="Alpha Corp",
        entity_type="ORGANIZATION",
        canonical_name="Alpha Corp",
        contexts=["Alpha Corp, a leader in sustainable energy solutions...", "Alpha Corp also hinted at 'Project Chimera'..."]
    )

    relationship1 = ExtractedRelationship(
        source_canonical_name="Aris Thorne",
        relationship_type="WORKS_FOR",
        target_canonical_name="Alpha Corp",
        contexts=["Dr. Aris Thorne, attributed this success to...", "He reports directly to Dr. Thorne."] # Example, context should ideally directly link
    )

    llm_output = LLMExtractionOutput(entities=[entity1, entity2], relationships=[relationship1])
    print("--- LLM Extraction Output ---")
    print(llm_output.model_dump_json(indent=2))

    ingestion_success = IngestionStatus(
        filename="sample_document.md",
        status="Completed",
        message="Successfully processed and ingested document.",
        entities_added=25,
        relationships_added=15
    )
    print("\n--- Ingestion Status ---")
    print(ingestion_success.model_dump_json(indent=2))