"""Tests for the simulated marketplace."""

from lowball.simulation.market import Marketplace


def test_marketplace_loads_listings() -> None:
    market = Marketplace()
    assert len(market.listings) > 0


def test_marketplace_has_distractor_listings() -> None:
    """Marketplace should include trap/distractor listings."""
    market = Marketplace()
    # Should have more listings now with distractors
    assert len(market.listings) >= 15


def test_search_by_make() -> None:
    market = Marketplace()
    results = market.search(make="Honda")
    assert all(r.make == "Honda" for r in results)
    assert len(results) > 0


def test_search_by_make_model() -> None:
    market = Marketplace()
    results = market.search(make="Honda", model="Civic")
    assert all(r.make == "Honda" and r.model == "Civic" for r in results)


def test_search_by_price() -> None:
    market = Marketplace()
    results = market.search(price_max=20000)
    assert all(r.price <= 20000 for r in results)


def test_search_by_mileage() -> None:
    market = Marketplace()
    results = market.search(mileage_max=40000)
    assert all(r.mileage <= 40000 for r in results)


def test_get_listing_by_id() -> None:
    market = Marketplace()
    listing = market.get_listing("civic-001")
    assert listing is not None
    assert listing.make == "Honda"
    assert listing.model == "Civic"


def test_get_price_stats() -> None:
    market = Marketplace()
    stats = market.get_price_stats("Honda", "Civic", 2019)
    assert "avg" in stats
    assert stats["count"] >= 1


def test_distractor_civic_exists() -> None:
    """Suspiciously cheap Civic should be in the marketplace."""
    market = Marketplace()
    listing = market.get_listing("civic-004")
    assert listing is not None
    assert listing.price < 13000  # Way below market
    assert listing.has_vehicle_history is False


def test_modified_wrx_exists() -> None:
    """Modified WRX trap should be in marketplace."""
    market = Marketplace()
    listing = market.get_listing("wrx-003")
    assert listing is not None
    assert "aftermarket exhaust" in listing.features
    assert "accessport tune" in listing.features


def test_accident_accord_exists() -> None:
    """Accord with accident history should be in marketplace."""
    market = Marketplace()
    listing = market.get_listing("accord-003")
    assert listing is not None
    assert listing.accident_history is True


def test_search_returns_mixed_quality() -> None:
    """Searching for Civics should return both good and suspicious listings."""
    market = Marketplace()
    results = market.search(make="Honda", model="Civic")
    has_good = any(r.has_vehicle_history and not r.accident_history for r in results)
    has_suspicious = any(not r.has_vehicle_history or r.num_owners >= 4 for r in results)
    assert has_good
    assert has_suspicious
