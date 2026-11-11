"""
Location-aware Mandi Price Tool for KisanSathi.
Finds nearest agricultural markets using Agmarknet master data, computes net-realizable
prices accounting for transport costs, handles reporting-date fallbacks, and supports
specific named market inquiries.
"""

from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.config import settings
from backend.tools.location import LocationManager


class MandiMarket(BaseModel):
    """Mandi metadata representation."""
    mandi_id: str
    name: str
    district: str
    state: str
    latitude: float
    longitude: float
    major_crops: List[str]
    is_major_terminal: bool = False


class MandiPriceRecord(BaseModel):
    """Price, arrival, and net-realizable metrics for a mandi."""
    mandi_id: str
    mandi_name: str
    district: str
    commodity: str
    min_price: float
    max_price: float
    modal_price: float
    arrival_tonnes: float
    report_date: str
    distance_km: float
    transport_cost_per_quintal: float
    net_realizable_price: float
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    is_date_fallback: bool = False


class MandiQueryResult(BaseModel):
    """Comprehensive mandi comparison result."""
    crop: str
    reference_location: str
    queried_mandi_name: Optional[str] = None
    prices: List[MandiPriceRecord] = Field(default_factory=list)
    best_net_mandi: Optional[MandiPriceRecord] = None
    transport_rate_per_km_quintal: float = 2.50
    summary_en: str
    summary_hi: str
    summary_te: str
    is_cached: bool = False


class MandiTool:
    """Orchestrates location-based mandi price searches, transport deduction, and fallbacks."""

    MASTER_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "maharashtra_mandis_master.json"
    PRICES_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "mandi_prices_seed.json"
    CACHE_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "mandi_cache.json"

    def __init__(
        self,
        transport_rate_per_km_quintal: float = 2.50,
        cache_ttl_seconds: int = 7200
    ):
        self.transport_rate = transport_rate_per_km_quintal
        self.cache_ttl = cache_ttl_seconds
        self.mandis: List[MandiMarket] = []
        self.price_db: List[Dict[str, Any]] = []
        self._load_master_data()
        self._load_price_data()

    def _load_master_data(self) -> None:
        if self.MASTER_FILE.exists():
            try:
                data = json.loads(self.MASTER_FILE.read_text(encoding="utf-8"))
                self.mandis = [MandiMarket(**m) for m in data]
            except Exception:
                self.mandis = []

    def _load_price_data(self) -> None:
        if self.PRICES_FILE.exists():
            try:
                self.price_db = json.loads(self.PRICES_FILE.read_text(encoding="utf-8"))
            except Exception:
                self.price_db = []

    def _get_prices_for_mandi_crop(self, mandi_id: str, crop: str) -> Optional[Dict[str, Any]]:
        """Looks up the most recent price record for a mandi and crop."""
        norm_crop = crop.lower().strip()
        matches = [
            p for p in self.price_db
            if p["mandi_id"] == mandi_id and p["commodity"].lower() == norm_crop
        ]
        if not matches:
            return None
        # Sort by report_date descending
        matches.sort(key=lambda x: x["report_date"], reverse=True)
        return matches[0]

    def query_prices(
        self,
        crop: str,
        user_lat: float,
        user_lon: float,
        specific_mandi_name: Optional[str] = None,
        max_results: int = 5,
        location_label: Optional[str] = None
    ) -> MandiQueryResult:
        """
        Calculates distances to mandis, computes net-realizable prices,
        and ranks by profitability minus freight cost.
        """
        norm_crop = crop.lower().strip()
        loc_label = location_label or f"Coordinates ({round(user_lat, 2)}, {round(user_lon, 2)})"

        # Calculate distances for all mandis
        ranked_markets = []
        for m in self.mandis:
            dist = LocationManager.haversine_distance(user_lat, user_lon, m.latitude, m.longitude)
            ranked_markets.append((dist, m))

        ranked_markets.sort(key=lambda x: x[0])

        # Filter candidate price records
        price_records: List[MandiPriceRecord] = []
        specific_match_record: Optional[MandiPriceRecord] = None

        for dist, m in ranked_markets:
            price_info = self._get_prices_for_mandi_crop(m.mandi_id, norm_crop)
            if not price_info:
                continue

            trans_cost = round(dist * self.transport_rate, 1)
            net_price = round(price_info["modal_price"] - trans_cost, 1)

            rec = MandiPriceRecord(
                mandi_id=m.mandi_id,
                mandi_name=m.name,
                district=m.district,
                commodity=norm_crop,
                min_price=float(price_info["min_price"]),
                max_price=float(price_info["max_price"]),
                modal_price=float(price_info["modal_price"]),
                arrival_tonnes=float(price_info["arrival_tonnes"]),
                report_date=price_info["report_date"],
                distance_km=dist,
                transport_cost_per_quintal=trans_cost,
                net_realizable_price=net_price,
                latitude=m.latitude,
                longitude=m.longitude,
                is_date_fallback=(price_info["report_date"] < "2026-11-10")
            )

            if specific_mandi_name and specific_mandi_name.lower() in m.name.lower():
                specific_match_record = rec
            else:
                price_records.append(rec)

        # Sort nearby alternatives by net-realizable price descending
        price_records.sort(key=lambda x: x.net_realizable_price, reverse=True)

        final_records = []
        if specific_match_record:
            final_records.append(specific_match_record)
            for r in price_records:
                if r.mandi_id != specific_match_record.mandi_id:
                    final_records.append(r)
        else:
            final_records = price_records

        final_records = final_records[:max_results]
        best_net = max(final_records, key=lambda x: x.net_realizable_price) if final_records else None

        # Generate localized explanations
        if best_net:
            summary_en = (
                f"For {norm_crop.title()}, {best_net.mandi_name} ({best_net.distance_km} km) offers the highest net return "
                f"of ₹{best_net.net_realizable_price:.0f}/q after estimated transport cost (₹{best_net.transport_cost_per_quintal:.0f}/q at ₹{self.transport_rate}/km). "
                f"Modal price: ₹{best_net.modal_price:.0f}/q (Date: {best_net.report_date})."
            )
            summary_hi = (
                f"{norm_crop.title()} के लिए, {best_net.mandi_name} ({best_net.distance_km} किमी) में परिवहन खर्च (₹{best_net.transport_cost_per_quintal:.0f}/क्विंटल) "
                f"घटाने के बाद सबसे अधिक शुद्ध भाव ₹{best_net.net_realizable_price:.0f}/क्विंटल मिल रहा है। "
                f"मंडी भाव: ₹{best_net.modal_price:.0f}/क्विंटल (दिनांक: {best_net.report_date})।"
            )
            summary_te = (
                f"{norm_crop.title()} పంటకు, రవాణా ఖర్చు (₹{best_net.transport_cost_per_quintal:.0f}/క్వింటాల్) మినహాయించిన తర్వాత "
                f"{best_net.mandi_name} మార్కెట్ ({best_net.distance_km} కి.మీ) అత్యధిక నికర రాబడి ₹{best_net.net_realizable_price:.0f}/క్వింటాల్ అందిస్తోంది. "
                f"మార్కెట్ మోడల్ ధర: ₹{best_net.modal_price:.0f}/క్వింటాల్ (తేదీ: {best_net.report_date})."
            )
        else:
            summary_en = f"No recent mandi price reports found within coverage for {norm_crop.title()}."
            summary_hi = f"{norm_crop.title()} के लिए आसपास की मंडियों में हालिया भाव उपलब्ध नहीं हैं।"
            summary_te = f"{norm_crop.title()} పంటకు సమీప మార్కెట్లలో తాజా ధరల నివేదికలు లభించలేదు."

        return MandiQueryResult(
            crop=norm_crop,
            reference_location=loc_label,
            queried_mandi_name=specific_mandi_name,
            prices=final_records,
            best_net_mandi=best_net,
            transport_rate_per_km_quintal=self.transport_rate,
            summary_en=summary_en,
            summary_hi=summary_hi,
            summary_te=summary_te,
            is_cached=False
        )
