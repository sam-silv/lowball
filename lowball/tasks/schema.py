"""Task instance schema for Lowball."""

from enum import Enum
from pydantic import BaseModel, Field


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class VehicleSpec(BaseModel):
    """Target vehicle the agent must find and negotiate for."""

    make: str = Field(description="Manufacturer (e.g., Honda)")
    model: str = Field(description="Model name (e.g., Civic)")
    year_min: int = Field(description="Minimum acceptable model year")
    year_max: int = Field(description="Maximum acceptable model year")
    trim: str | None = Field(default=None, description="Specific trim level (e.g., EX, Sport)")
    max_mileage: int | None = Field(default=None, description="Maximum acceptable mileage")
    required_features: list[str] = Field(default_factory=list, description="Must-have features")
    preferred_features: list[str] = Field(default_factory=list, description="Nice-to-have features")
    color_preferences: list[str] = Field(default_factory=list)


class SellerConfig(BaseModel):
    """Configuration for the simulated seller."""

    persona_id: str = Field(description="Reference to a seller persona in data/personas/")
    asking_price: float = Field(description="Seller's initial listing price")
    minimum_price: float = Field(description="Lowest price the seller will accept")
    firmness: float = Field(ge=0.0, le=1.0, description="0.0 = very flexible, 1.0 = won't budge")
    urgency: float = Field(ge=0.0, le=1.0, description="0.0 = no rush, 1.0 = must sell today")
    hidden_issues: list[str] = Field(
        default_factory=list,
        description="Problems the seller knows about but won't volunteer",
    )
    negotiation_style: str = Field(
        default="balanced",
        description="One of: cooperative, balanced, aggressive, deceptive",
    )


class TaskInstance(BaseModel):
    """A single benchmark task instance."""

    task_id: str = Field(description="Unique task identifier (e.g., easy/civic_2019)")
    difficulty: Difficulty
    description: str = Field(description="Natural language description of the buying goal")
    vehicle: VehicleSpec
    seller: SellerConfig
    fair_market_value: float = Field(description="Ground-truth FMV for scoring")
    budget_ceiling: float = Field(description="Maximum the buyer agent is allowed to spend")
    max_turns: int = Field(default=30, description="Maximum negotiation turns before timeout")
    required_research: list[str] = Field(
        default_factory=list,
        description="Information the agent should discover (for information_quality scoring)",
    )
    red_flags: list[str] = Field(
        default_factory=list,
        description="Red flags in the marketplace/listing that the agent should detect and avoid",
    )
    time_limit_seconds: int = Field(default=600, description="Wall-clock time limit for the task")
