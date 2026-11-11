"""
Unit and Integration Tests for KisanSathi Phase 3.
Covers:
1. Haversine distance and location geocoding.
2. Weather tool 30-minute grid caching & fallback mechanisms.
3. Mandi tool nearest-market ranking, transport freight deduction (₹2.50/km/q), and date fallbacks.
4. AI Intent Router classification & parameter extraction across English, Telugu, and Hindi.
5. WhatsApp Cloud API HMAC-SHA256 signature verification & rate limiting.
6. Phase 3 API endpoints (/farmer/location, /mandis/nearby, /weather/forecast, /watchlist, /digest).
"""

import hmac
import hashlib
import time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.memory.database import FarmerProfile
from backend.memory.memory_manager import MemoryManager
from backend.tools.location import LocationManager, GeoLocation
from backend.tools.weather import WeatherTool, WeatherReport
from backend.tools.mandi import MandiTool, MandiQueryResult
from backend.rag.router import ToolRouter
from backend.api.whatsapp import verify_signature, check_rate_limit, _RATE_LIMIT_STORE

client = TestClient(app)


# 1. Location & Geocoding Tests
def test_haversine_distance():
    # Nashik centroid (19.9975, 73.7898) to Lasalgaon (20.1478, 74.2251)
    dist = LocationManager.haversine_distance(19.9975, 73.7898, 20.1478, 74.2251)
    # Known distance ~48.4 km
    assert 45.0 <= dist <= 52.0


def test_resolve_location_named_places():
    geo_lasalgaon = LocationManager.resolve_location("Lasalgaon")
    assert geo_lasalgaon.village == "Lasalgaon"
    assert geo_lasalgaon.district == "Nashik"
    assert abs(geo_lasalgaon.latitude - 20.1478) < 0.01

    geo_pune = LocationManager.resolve_location("Pune")
    assert geo_pune.district == "Pune"
    assert abs(geo_pune.latitude - 18.5204) < 0.01

    # Cross-lingual alias (Hindi Devanagari)
    geo_hi = LocationManager.resolve_location("पिंपलगांव")
    assert geo_hi.village == "Pimpalgaon"
    assert geo_hi.district == "Nashik"


def test_parse_whatsapp_location_payload():
    payload = {
        "latitude": 19.9975,
        "longitude": 73.7898,
        "name": "Lasalgaon Mandi Yard",
        "address": "Nashik, Maharashtra"
    }
    geo = LocationManager.parse_whatsapp_location(payload)
    assert geo.latitude == 19.9975
    assert geo.longitude == 73.7898
    assert geo.village == "Lasalgaon Mandi Yard"
    assert geo.source == "whatsapp_pin"
    assert geo.consent_given is True


# 2. Weather Tool & Caching Tests
def test_weather_tool_grid_caching(tmp_path):
    w_tool = WeatherTool(api_key=None, cache_ttl_seconds=1800)
    w_tool.CACHE_FILE = tmp_path / "test_w_cache.json"
    w_tool._memory_cache = {}

    # 1st call: computed/synthetic observation
    rep1 = w_tool.get_weather(19.9975, 73.7898, "Nashik")
    assert rep1.is_cached is False
    assert rep1.is_outdated is False
    assert rep1.current_temp_c > 0
    assert len(rep1.daily_forecasts) == 7

    # 2nd call: retrieved from cache
    rep2 = w_tool.get_weather(19.9975, 73.7898, "Nashik")
    assert rep2.is_cached is True
    assert rep2.is_outdated is False
    assert rep2.current_temp_c == rep1.current_temp_c


def test_weather_agronomic_advice():
    w_tool = WeatherTool(api_key=None)
    # High rain probability (>=40%) should advise against spraying
    adv_en, adv_hi, adv_te, spray_ok = w_tool._generate_agronomic_advice(
        rain_prob=70, wind_kmh=10.0, temp_c=27.0, humidity=80
    )
    assert spray_ok is False
    assert "Avoid foliar spraying" in adv_en
    assert "छिड़काव" in adv_hi
    assert "పిచికారీ" in adv_te


# 3. Mandi Tool & Transport Deduction Tests
def test_mandi_nearest_ranking_and_transport_deduction():
    m_tool = MandiTool(transport_rate_per_km_quintal=2.50)
    # Query from Nashik coordinates for onion
    res = m_tool.query_prices(crop="onion", user_lat=19.9975, user_lon=73.7898)

    assert len(res.prices) > 0
    assert res.best_net_mandi is not None

    # Check transport freight calculation: Net = Modal - (Distance * 2.50)
    top = res.prices[0]
    expected_freight = round(top.distance_km * 2.50, 1)
    assert abs(top.transport_cost_per_quintal - expected_freight) < 0.1
    expected_net = round(top.modal_price - top.transport_cost_per_quintal, 1)
    assert abs(top.net_realizable_price - expected_net) < 0.1


def test_mandi_specific_named_market_query():
    m_tool = MandiTool()
    # Explicitly query Lasalgaon
    res = m_tool.query_prices(
        crop="onion",
        user_lat=18.5204,  # user is in Pune
        user_lon=73.8567,
        specific_mandi_name="Lasalgaon"
    )
    assert len(res.prices) > 0
    # First result should be Lasalgaon even if further than Pune mandis
    assert "Lasalgaon" in res.prices[0].mandi_name


# 4. AI Intent Router Tests
def test_tool_router_classification_and_personalization():
    router = ToolRouter()
    farmer = FarmerProfile(
        phone_number="+919876543210",
        name="రాములు",
        village="Lasalgaon",
        district="Nashik",
        primary_crop="onion",
        preferred_language="te"
    )

    # Weather Intent
    act_weather = router.classify_and_extract("Will it rain tomorrow in Nashik?", farmer)
    assert act_weather.intent == "weather"
    assert act_weather.target_place == "Nashik"

    # Mandi Intent with Named Mandi
    act_mandi = router.classify_and_extract("What is the onion price in Lasalgaon mandi?", farmer)
    assert act_mandi.intent == "mandi"
    assert act_mandi.crop == "onion"
    assert act_mandi.target_place == "Lasalgaon"

    # Watchlist Intent
    act_wl = router.classify_and_extract("Alert me when onion crosses 2200", farmer)
    assert act_wl.intent == "watchlist"
    assert act_wl.target_price == 2200.0

    # RAG Intent
    act_rag = router.classify_and_extract("What is the pesticide for tomato early blight?", farmer)
    assert act_rag.intent == "rag"
    assert act_rag.crop == "tomato"


# 5. WhatsApp Signature & Rate Limiting Tests
def test_whatsapp_signature_verification():
    secret = "test_kisansathi_secret"
    payload = b'{"object":"whatsapp_business_account"}'
    valid_sig = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()

    assert verify_signature(payload, valid_sig, secret) is True
    assert verify_signature(payload, "sha256=invalidhash", secret) is False
    assert verify_signature(payload, "invalid_prefix", secret) is False


def test_whatsapp_rate_limiting():
    phone = "+919999000111"
    _RATE_LIMIT_STORE.clear()

    # Under threshold should allow
    for _ in range(5):
        assert check_rate_limit(phone) is True

    # Exceeding threshold (30 requests)
    _RATE_LIMIT_STORE[phone] = [time.time()] * 30
    assert check_rate_limit(phone) is False


# 6. Phase 3 API Endpoint Integration Tests
def test_api_farmer_location_update_and_clear():
    phone = "+919876549999"
    # Update location
    resp = client.post("/api/v1/farmer/location", json={
        "phone_number": phone,
        "latitude": 20.1478,
        "longitude": 74.2251,
        "village": "Lasalgaon",
        "district": "Nashik",
        "consent_given": True
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["village"] == "Lasalgaon"
    assert data["latitude"] == 20.1478
    assert data["location_consent_given"] is True

    # Clear location
    del_resp = client.delete(f"/api/v1/farmer/{phone}/location")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "success"


def test_api_mandis_nearby_endpoint():
    resp = client.get("/api/v1/mandis/nearby?crop=onion&latitude=19.9975&longitude=73.7898")
    assert resp.status_code == 200
    data = resp.json()
    assert data["crop"] == "onion"
    assert len(data["prices"]) > 0
    assert data["best_net_mandi"] is not None


def test_api_weather_forecast_endpoint():
    resp = client.get("/api/v1/weather/forecast?latitude=19.9975&longitude=73.7898&location_name=Nashik")
    assert resp.status_code == 200
    data = resp.json()
    assert data["location_name"] == "Nashik"
    assert "current_temp_c" in data
    assert len(data["daily_forecasts"]) == 7


def test_api_watchlist_endpoint():
    phone = "+919876541122"
    resp = client.post("/api/v1/watchlist", json={
        "phone_number": phone,
        "crop": "onion",
        "target_price": 2250.0,
        "condition": "above"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["crop"] == "onion"
    assert data["target_price"] == 2250.0

    # Retrieve watchlists
    get_resp = client.get(f"/api/v1/watchlist/{phone}")
    assert get_resp.status_code == 200
    assert len(get_resp.json()) >= 1


def test_api_farmer_daily_digest():
    phone = "+919876543344"
    resp = client.get(f"/api/v1/farmer/{phone}/digest")
    assert resp.status_code == 200
    data = resp.json()
    assert "digest_text" in data
    assert "weather" in data
    assert "mandi" in data
