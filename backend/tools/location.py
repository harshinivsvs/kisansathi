"""
Location management and geocoding module for KisanSathi.
Provides district master data, Haversine distance calculations,
WhatsApp location pin parsing, and consent tracking.
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class GeoLocation(BaseModel):
    """Geographical coordinate and administrative context."""
    latitude: float
    longitude: float
    village: Optional[str] = None
    taluka: Optional[str] = None
    district: Optional[str] = None
    state: str = "Maharashtra"
    source: str = "profile"  # "whatsapp_pin", "typed_name", "district_dropdown", "profile"
    consent_given: bool = True


class LocationManager:
    """Handles location resolution, distance computation, and consent guardrails."""

    # Maharashtra district centroids and prominent agricultural clusters
    DISTRICT_CENTROIDS: Dict[str, Tuple[float, float]] = {
        "nashik": (19.9975, 73.7898),
        "pune": (18.5204, 73.8567),
        "ahmednagar": (19.0952, 74.7496),
        "nagpur": (21.1458, 79.0882),
        "akola": (20.7002, 77.0082),
        "amravati": (20.9374, 77.7796),
        "jalgaon": (21.0077, 75.5626),
        "jalna": (19.8347, 75.8816),
        "aurangabad": (19.8762, 75.3433),
        "chhatrapati sambhajinagar": (19.8762, 75.3433),
        "solapur": (17.6599, 75.9064),
        "kolhapur": (16.7050, 74.2433),
        "sangli": (16.8524, 74.5815),
        "satara": (17.6805, 74.0183),
        "dhule": (20.9042, 74.7749),
        "nandurbar": (21.3684, 74.2389),
        "wardha": (20.7453, 78.6022),
        "yavatmal": (20.3888, 78.1204),
        "nanded": (19.1383, 77.3210),
        "buldhana": (20.5293, 76.1843),
        "parbhani": (19.2608, 76.7748),
        "beed": (18.9891, 75.7601),
        "latur": (18.4088, 76.5604),
        "osmanabad": (18.1861, 76.0419),
        "dharashiv": (18.1861, 76.0419),
        "chandrapur": (19.9615, 79.2961),
        "gadchiroli": (20.1849, 80.0035),
        "gondia": (21.4598, 80.1961),
        "bhandara": (21.1714, 79.6542),
        "raigad": (18.5158, 73.1822),
        "ratnagiri": (16.9902, 73.3120),
        "sindhudurg": (16.1182, 73.6931),
        "thane": (19.2183, 72.9781),
        "palghar": (19.6967, 72.7699)
    }

    # Prominent agricultural town geocodes
    AGRI_TOWN_COORDINATES: Dict[str, Tuple[float, float, str]] = {
        "lasalgaon": (20.1478, 74.2251, "Nashik"),
        "pimpalgaon": (20.1706, 73.9856, "Nashik"),
        "baramati": (18.1517, 74.5772, "Pune"),
        "rahata": (19.6738, 74.4842, "Ahmednagar"),
        "kalamna": (21.1458, 79.0882, "Nagpur"),
        "yeotmal": (20.3888, 78.1204, "Yavatmal"),
        "sangamner": (19.5762, 74.2144, "Ahmednagar"),
        "shrirampur": (19.6190, 74.6558, "Ahmednagar"),
        "malegaon": (20.5539, 74.5288, "Nashik"),
        "dindori": (20.2014, 73.8347, "Nashik"),
        "shegaon": (20.7937, 76.6935, "Buldhana"),
        "pandharpur": (17.6775, 75.3262, "Solapur"),
    }

    # Cross-lingual place aliases (Hindi Devanagari, Marathi, Telugu)
    MULTILINGUAL_PLACE_ALIASES: Dict[str, str] = {
        # Hindi Devanagari & Marathi
        "नासिक": "Nashik",
        "नाशिक": "Nashik",
        "पुणे": "Pune",
        "लासलगाव": "Lasalgaon",
        "लासलगांव": "Lasalgaon",
        "पिंपलगांव": "Pimpalgaon",
        "पिंपलगाव": "Pimpalgaon",
        "अहमदनगर": "Ahmednagar",
        "राहाता": "Rahata",
        "सोलापूर": "Solapur",
        "सोलापुर": "Solapur",
        "नागपुर": "Nagpur",
        "नागपूर": "Nagpur",
        "अकोला": "Akola",
        "यवतमाळ": "Yavatmal",
        "यवतमाल": "Yavatmal",
        "बारामती": "Baramati",
        "कळमना": "Kalamna",
        "कलमना": "Kalamna",
        # Telugu Script
        "నాసిక్": "Nashik",
        "నాసిక్‌": "Nashik",
        "పుణె": "Pune",
        "పూణే": "Pune",
        "లాసల్‌గావ్": "Lasalgaon",
        "లాసల్‌గావ్‌": "Lasalgaon",
        "లాసలగావ్": "Lasalgaon",
        "పింపల్‌గావ్": "Pimpalgaon",
        "పింపల్‌గావ్‌": "Pimpalgaon",
        "అహ్మద్‌నగర్": "Ahmednagar",
        "రాహతా": "Rahata",
        "సోలాపూర్": "Solapur",
        "నాగపూర్": "Nagpur",
        "కలమ్నా": "Kalamna",
        "బారామతి": "Baramati",
        "యవత్మాల్": "Yavatmal",
        "అకోలా": "Akola"
    }

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculates great-circle distance between two coordinates in kilometers.
        """
        R = 6371.0  # Earth's radius in km
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = (
            math.sin(delta_phi / 2.0) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
        )
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return round(R * c, 2)

    @classmethod
    def resolve_location(
        cls,
        query_place: Optional[str] = None,
        default_district: str = "Nashik"
    ) -> GeoLocation:
        """
        Resolves a typed place name, mandi name, or district to coordinates.
        Defaults to configured cluster (Nashik) if unprovided.
        """
        if not query_place:
            target = default_district.lower().strip()
            coords = cls.DISTRICT_CENTROIDS.get(target, cls.DISTRICT_CENTROIDS["nashik"])
            return GeoLocation(
                latitude=coords[0],
                longitude=coords[1],
                district=default_district.title(),
                state="Maharashtra",
                source="district_dropdown"
            )

        p_raw = query_place.strip()
        canonical_alias = cls.MULTILINGUAL_PLACE_ALIASES.get(p_raw)
        if not canonical_alias:
            for alias, canon in cls.MULTILINGUAL_PLACE_ALIASES.items():
                if alias.lower() in p_raw.lower():
                    canonical_alias = canon
                    break

        p_clean = (canonical_alias or p_raw).lower().strip()

        # Check town coordinates first
        for town, (lat, lon, dist) in cls.AGRI_TOWN_COORDINATES.items():
            if town in p_clean:
                return GeoLocation(
                    latitude=lat,
                    longitude=lon,
                    village=town.title(),
                    district=dist,
                    state="Maharashtra",
                    source="typed_name"
                )

        # Check district centroids
        for dist, (lat, lon) in cls.DISTRICT_CENTROIDS.items():
            if dist in p_clean:
                return GeoLocation(
                    latitude=lat,
                    longitude=lon,
                    district=dist.title(),
                    state="Maharashtra",
                    source="typed_name"
                )

        # Fallback to default district
        fallback_coords = cls.DISTRICT_CENTROIDS.get(default_district.lower(), cls.DISTRICT_CENTROIDS["nashik"])
        return GeoLocation(
            latitude=fallback_coords[0],
            longitude=fallback_coords[1],
            village=query_place.title(),
            district=default_district.title(),
            state="Maharashtra",
            source="typed_name"
        )

    @staticmethod
    def parse_whatsapp_location(payload: Dict[str, Any]) -> GeoLocation:
        """Parses WhatsApp location message payload."""
        lat = float(payload.get("latitude", 19.9975))
        lon = float(payload.get("longitude", 73.7898))
        name = payload.get("name")
        address = payload.get("address")

        return GeoLocation(
            latitude=round(lat, 4),
            longitude=round(lon, 4),
            village=name,
            district=address,
            state="Maharashtra",
            source="whatsapp_pin",
            consent_given=True
        )
