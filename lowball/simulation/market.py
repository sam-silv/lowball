"""Simulated car marketplace that agents browse via tool-based search."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

VEHICLES_DIR = Path(__file__).parent.parent.parent / "data" / "vehicles"


class VehicleListing(BaseModel):
    """A single car listing in the marketplace."""

    listing_id: str
    make: str
    model: str
    year: int
    trim: str | None = None
    mileage: int
    price: float
    color: str
    features: list[str] = Field(default_factory=list)
    condition: str = Field(default="good", description="excellent, good, fair, poor")
    location: str = Field(default="Local")
    description: str = ""
    seller_name: str = ""
    days_listed: int = 0
    num_views: int = 0
    has_vehicle_history: bool = True
    accident_history: bool = False
    num_owners: int = 1


class Marketplace:
    """Simulated car marketplace with browsable listings.

    Listings are loaded from YAML files. Agents interact with the
    marketplace via structured tool calls (search, get details, select).
    """

    def __init__(self, vehicles_dir: Path = VEHICLES_DIR) -> None:
        self.listings: list[VehicleListing] = []
        self._load_listings(vehicles_dir)

    def _load_listings(self, vehicles_dir: Path) -> None:
        for listing_file in vehicles_dir.glob("*.yaml"):
            with open(listing_file) as f:
                raw = yaml.safe_load(f)
            if isinstance(raw, list):
                for item in raw:
                    self.listings.append(VehicleListing(**item))
            else:
                self.listings.append(VehicleListing(**raw))

    def search(
        self,
        make: str | None = None,
        model: str | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        price_max: float | None = None,
        mileage_max: int | None = None,
    ) -> list[VehicleListing]:
        """Filter listings by search criteria."""
        results = self.listings

        if make:
            results = [l for l in results if l.make.lower() == make.lower()]
        if model:
            results = [l for l in results if l.model.lower() == model.lower()]
        if year_min:
            results = [l for l in results if l.year >= year_min]
        if year_max:
            results = [l for l in results if l.year <= year_max]
        if price_max:
            results = [l for l in results if l.price <= price_max]
        if mileage_max:
            results = [l for l in results if l.mileage <= mileage_max]

        return results

    def get_listing(self, listing_id: str) -> VehicleListing | None:
        for listing in self.listings:
            if listing.listing_id == listing_id:
                return listing
        return None

    def get_price_stats(self, make: str, model: str, year: int) -> dict[str, float]:
        """Return market price statistics for a vehicle type."""
        matching = [
            l for l in self.listings
            if l.make.lower() == make.lower()
            and l.model.lower() == model.lower()
            and l.year == year
        ]
        if not matching:
            return {}

        prices = [l.price for l in matching]
        return {
            "min": min(prices),
            "max": max(prices),
            "avg": sum(prices) / len(prices),
            "count": len(prices),
        }
