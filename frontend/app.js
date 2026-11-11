/**
 * KisanSathi Interactive Web Demo & Telemetry Debugger
 */

const API_BASE = window.location.origin.includes("localhost") || window.location.origin.includes("127.0.0.1")
  ? ""
  : "";

let currentLanguage = "te";
let currentPhone = "+919876543210";
let currentLat = 19.9975;
let currentLon = 73.7898;
let currentVillage = "Lasalgaon";
let currentDistrict = "Nashik";
let currentCrop = "onion";

// Leaflet Map & Markers
let map = null;
let farmerMarker = null;
let mandiMarkersLayer = null;

// Audio Recording State
let mediaRecorder = null;
let audioChunks = [];
let recordTimerInterval = null;
let recordSeconds = 0;

// Coordinate map for dropdown
const DISTRICT_COORDS = {
  "nashik": { lat: 19.9975, lon: 73.7898, name: "Nashik", village: "Nashik" },
  "pimpalgaon": { lat: 20.1706, lon: 73.9856, name: "Nashik", village: "Pimpalgaon Baswant" },
  "lasalgaon": { lat: 20.1478, lon: 74.2251, name: "Nashik", village: "Lasalgaon" },
  "pune": { lat: 18.5204, lon: 73.8567, name: "Pune", village: "Baramati" },
  "ahmednagar": { lat: 19.0952, lon: 74.7496, name: "Ahmednagar", village: "Rahata" },
  "solapur": { lat: 17.6599, lon: 75.9064, name: "Solapur", village: "Pandharpur" },
  "nagpur": { lat: 21.1458, lon: 79.0882, name: "Nagpur", village: "Kalamna" },
  "yavatmal": { lat: 20.3888, lon: 78.1204, name: "Yavatmal", village: "Yavatmal" },
  "akola": { lat: 20.7002, lon: 77.0082, name: "Akola", village: "Akola" }
};

document.addEventListener("DOMContentLoaded", () => {
  initMap();
  setupEventListeners();
  loadFarmerProfile(currentPhone);
  loadNearbyMandis(currentCrop, currentLat, currentLon);
});

/* Map Initialization */
function initMap() {
  const mapContainer = document.getElementById("mandi-map");
  if (!mapContainer || typeof L === "undefined") return;

  map = L.map("mandi-map").setView([currentLat, currentLon], 9);

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18,
    attribution: "© OpenStreetMap contributors"
  }).addTo(map);

  mandiMarkersLayer = L.layerGroup().addTo(map);
  updateFarmerMarker();
}

function updateFarmerMarker() {
  if (!map) return;
  if (farmerMarker) map.removeLayer(farmerMarker);

  const farmerIcon = L.divIcon({
    html: '<div style="background:#15803d;color:white;width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;border:2px solid white;box-shadow:0 2px 4px rgba(0,0,0,0.3);"><i class="fa-solid fa-house"></i></div>',
    className: "farmer-pin",
    iconSize: [28, 28],
    iconAnchor: [14, 14]
  });

  farmerMarker = L.marker([currentLat, currentLon], { icon: farmerIcon })
    .addTo(map)
    .bindPopup(`<b>Farmer Location</b><br>${currentVillage}, ${currentDistrict}`)
    .openPopup();

  map.panTo([currentLat, currentLon]);
}

/* Event Listeners */
function setupEventListeners() {
  // Language Switcher
  document.querySelectorAll(".lang-btn").forEach(btn => {
    btn.addEventListener("click", (e) => {
      document.querySelectorAll(".lang-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentLanguage = btn.dataset.lang;
      document.getElementById("current-rec-lang").innerText = btn.innerText;
    });
  });

  // Quick prompt chips
  document.querySelectorAll(".prompt-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      document.getElementById("query-input").value = chip.dataset.query;
      sendQuery();
    });
  });

  // Send button & enter key
  document.getElementById("send-btn").addEventListener("click", sendQuery);
  document.getElementById("query-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendQuery();
  });

  // Location Selector
  document.getElementById("btn-update-loc").addEventListener("click", () => {
    const sel = document.getElementById("district-picker").value;
    const target = DISTRICT_COORDS[sel];
    if (target) {
      updateLocationOnServer(target.lat, target.lon, target.village, target.name);
    }
  });

  // Delete Location Consent
  document.getElementById("btn-clear-loc").addEventListener("click", async () => {
    if (confirm("Clear location consent and reset to default?")) {
      await fetch(`/api/v1/farmer/${encodeURIComponent(currentPhone)}/location`, { method: "DELETE" });
      document.getElementById("consent-badge").innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Consent Cleared (Defaulting to Nashik)`;
      document.getElementById("consent-badge").style.borderColor = "#fca5a5";
      document.getElementById("consent-badge").style.background = "#fee2e2";
      document.getElementById("consent-badge").style.color = "#991b1b";
    }
  });

  // Daily Digest Button
  document.getElementById("btn-daily-digest").addEventListener("click", fetchDailyDigest);

  // Mic Button (Audio Recording)
  const micBtn = document.getElementById("mic-btn");
  micBtn.addEventListener("click", toggleAudioRecording);
  document.getElementById("btn-stop-rec").addEventListener("click", stopAudioRecording);

  // Phone Change Listener
  document.getElementById("profile-phone").addEventListener("change", (e) => {
    currentPhone = e.target.value.trim();
    loadFarmerProfile(currentPhone);
  });
}

/* Sending Farm Advisory Query */
async function sendQuery() {
  const input = document.getElementById("query-input");
  const query = input.value.trim();
  if (!query) return;

  input.value = "";
  appendUserMessage(query);

  const returnAudio = document.getElementById("return-audio-toggle").checked;

  try {
    const res = await fetch("/api/v1/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: query,
        phone_number: currentPhone,
        language: currentLanguage,
        return_audio: returnAudio
      })
    });

    if (!res.ok) {
      throw new Error(`HTTP error! status: ${res.status}`);
    }

    const data = await res.json();
    appendAssistantMessage(data);
    updateTelemetry(data);

    // Refresh profile state
    if (data.farmer_profile) {
      renderProfile(data.farmer_profile);
      if (data.farmer_profile.primary_crop) {
        currentCrop = data.farmer_profile.primary_crop;
      }
      loadNearbyMandis(currentCrop, currentLat, currentLon);
    }
  } catch (err) {
    appendAssistantMessage({
      answer: "⚠️ Server connection error. Please verify the backend is active.",
      intent: "error",
      citations: []
    });
  }
}

/* Appending Messages */
function appendUserMessage(text) {
  const container = document.getElementById("chat-messages");
  const msgDiv = document.createElement("div");
  msgDiv.className = "message user-message";
  msgDiv.innerHTML = `
    <div class="message-avatar"><i class="fa-solid fa-user"></i></div>
    <div class="message-content">
      <div class="message-header">
        <span class="sender-name">Farmer</span>
        <span class="intent-badge" style="background:#bae6fd;color:#0369a1;">User Inquiry</span>
      </div>
      <div class="message-body">${escapeHtml(text)}</div>
    </div>
  `;
  container.appendChild(msgDiv);
  container.scrollTop = container.scrollHeight;
}

function appendAssistantMessage(data) {
  const container = document.getElementById("chat-messages");
  const msgDiv = document.createElement("div");
  msgDiv.className = "message assistant-message";

  let badgeClass = "badge-rag";
  let badgeIcon = "fa-shield-halved";
  let badgeLabel = "Verified Advisory";

  if (data.intent === "weather") {
    badgeClass = "badge-weather";
    badgeIcon = "fa-cloud-sun-rain";
    badgeLabel = "Local Weather Tool";
  } else if (data.intent === "mandi") {
    badgeClass = "badge-mandi";
    badgeIcon = "fa-money-bill-wave";
    badgeLabel = "Agmarknet Mandi Tool";
  } else if (data.intent === "watchlist") {
    badgeClass = "badge-watchlist";
    badgeIcon = "fa-bell";
    badgeLabel = "Price Alert Watchlist";
  }

  let audioHtml = "";
  if (data.audio_url) {
    audioHtml = `
      <div class="message-audio">
        <audio controls src="${data.audio_url}" class="audio-player"></audio>
      </div>
    `;
  }

  let citationsHtml = "";
  if (data.citations && data.citations.length > 0) {
    citationsHtml = `
      <div class="message-citations">
        <strong>Sources:</strong> ${data.citations.join(" • ")}
      </div>
    `;
  }

  msgDiv.innerHTML = `
    <div class="message-avatar"><i class="fa-solid fa-robot"></i></div>
    <div class="message-content">
      <div class="message-header">
        <span class="sender-name">KisanSathi AI</span>
        <span class="intent-badge ${badgeClass}"><i class="fa-solid ${badgeIcon}"></i> ${badgeLabel}</span>
      </div>
      <div class="message-body">${data.answer}</div>
      ${audioHtml}
      ${citationsHtml}
    </div>
  `;
  container.appendChild(msgDiv);
  container.scrollTop = container.scrollHeight;
}

/* Updating Telemetry & Side Panel */
function updateTelemetry(data) {
  document.getElementById("tele-intent").innerText = (data.intent || "rag").toUpperCase();
  document.getElementById("tele-confidence").innerText = `${((data.confidence || 0.95) * 100).toFixed(1)}%`;
  document.getElementById("tele-latency").innerText = `${data.latency_ms || 45} ms`;
  document.getElementById("tele-cost").innerText = `$${(data.estimated_cost_usd || 0.0001).toFixed(4)}`;

  const sourcesContainer = document.getElementById("tele-sources");
  if (data.citations && data.citations.length > 0) {
    sourcesContainer.innerText = data.citations.join(" | ");
  } else {
    sourcesContainer.innerText = "Institutional Agricultural Knowledge";
  }
}

/* Loading Farmer Profile */
async function loadFarmerProfile(phone) {
  try {
    const res = await fetch(`/api/v1/farmer/${encodeURIComponent(phone)}`);
    if (res.ok) {
      const data = await res.json();
      renderProfile(data);
    }
  } catch (err) {
    console.error("Failed to fetch farmer profile:", err);
  }
}

function renderProfile(profile) {
  if (profile.name) document.getElementById("profile-name").innerText = profile.name;
  if (profile.primary_crop) document.getElementById("profile-crop").innerText = profile.primary_crop.toUpperCase();
  if (profile.sowing_date) document.getElementById("profile-sowing").innerText = profile.sowing_date;

  currentLat = profile.latitude || 19.9975;
  currentLon = profile.longitude || 73.7898;
  currentVillage = profile.village || "Lasalgaon";
  currentDistrict = profile.district || "Nashik";

  document.getElementById("profile-location-text").innerText =
    `${currentVillage}, ${currentDistrict} (${currentLat.toFixed(4)}, ${currentLon.toFixed(4)})`;

  updateFarmerMarker();
}

/* Updating Location on Server */
async function updateLocationOnServer(lat, lon, village, district) {
  try {
    const res = await fetch("/api/v1/farmer/location", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        phone_number: currentPhone,
        latitude: lat,
        longitude: lon,
        village: village,
        district: district,
        consent_given: true
      })
    });

    if (res.ok) {
      const updated = await res.json();
      renderProfile(updated);
      loadNearbyMandis(currentCrop, lat, lon);

      appendAssistantMessage({
        answer: `📍 Location pin updated to **${village}, ${district}** (${lat.toFixed(4)}, ${lon.toFixed(4)}).\nAll future weather advisories and nearest mandi prices will automatically use this location.`,
        intent: "weather",
        citations: ["Location Intelligence Module"]
      });
    }
  } catch (err) {
    console.error("Location update failed:", err);
  }
}

/* Loading Nearby Mandis & Price Table */
async function loadNearbyMandis(crop, lat, lon) {
  try {
    const res = await fetch(`/api/v1/mandis/nearby?crop=${encodeURIComponent(crop)}&latitude=${lat}&longitude=${lon}`);
    if (!res.ok) return;

    const data = await res.json();
    const tbody = document.getElementById("mandi-table-body");
    tbody.innerHTML = "";

    if (mandiMarkersLayer) mandiMarkersLayer.clearLayers();

    if (!data.prices || data.prices.length === 0) {
      tbody.innerHTML = `<tr><td colspan="5" class="loading-cell">No market reports found for ${crop}</td></tr>`;
      return;
    }

    data.prices.forEach((m, idx) => {
      const isBest = idx === 0;
      const tr = document.createElement("tr");
      if (isBest) tr.className = "row-best";

      tr.innerHTML = `
        <td>${m.mandi_name} ${isBest ? '<span class="badge-top-net">Top Net</span>' : ''}</td>
        <td>${m.distance_km} km</td>
        <td>₹${m.modal_price}</td>
        <td>-₹${m.transport_cost_per_quintal}</td>
        <td><strong>₹${m.net_realizable_price}</strong></td>
      `;
      tbody.appendChild(tr);

      // Add to Map
      if (map) {
        const marker = L.circleMarker([m.latitude || (currentLat + (idx * 0.05)), m.longitude || (currentLon + (idx * 0.05))], {
          radius: isBest ? 8 : 6,
          fillColor: isBest ? "#16a34a" : "#eab308",
          color: "#ffffff",
          weight: 2,
          opacity: 1,
          fillOpacity: 0.8
        });
        marker.bindPopup(`<b>${m.mandi_name}</b><br>Modal: ₹${m.modal_price}/q<br>Net Return: ₹${m.net_realizable_price}/q<br>Distance: ${m.distance_km} km`);
        mandiMarkersLayer.addLayer(marker);
      }
    });
  } catch (err) {
    console.error("Failed to load nearby mandis:", err);
  }
}

/* Daily Digest */
async function fetchDailyDigest() {
  try {
    const res = await fetch(`/api/v1/farmer/${encodeURIComponent(currentPhone)}/digest`);
    if (res.ok) {
      const data = await res.json();
      appendAssistantMessage({
        answer: data.digest_text,
        intent: "weather",
        citations: ["IMD Weather Station", "Agmarknet Price Reports", "Crop Calendar Engine"]
      });
    }
  } catch (err) {
    console.error("Digest retrieval failed:", err);
  }
}

/* Audio Voice Recording */
function toggleAudioRecording() {
  const banner = document.getElementById("recording-banner");
  if (banner.style.display === "none") {
    startAudioRecording();
  } else {
    stopAudioRecording();
  }
}

function startAudioRecording() {
  const banner = document.getElementById("recording-banner");
  banner.style.display = "flex";
  recordSeconds = 0;
  document.getElementById("recording-timer").innerText = "00:00";

  recordTimerInterval = setInterval(() => {
    recordSeconds++;
    const mins = String(Math.floor(recordSeconds / 60)).padStart(2, "0");
    const secs = String(recordSeconds % 60).padStart(2, "0");
    document.getElementById("recording-timer").innerText = `${mins}:${secs}`;
  }, 1000);

  // If MediaRecorder API is supported
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];
      mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
      mediaRecorder.start();
    }).catch(() => {
      console.log("Mic access simulated for demo environment");
    });
  }
}

function stopAudioRecording() {
  const banner = document.getElementById("recording-banner");
  banner.style.display = "none";
  clearInterval(recordTimerInterval);

  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }

  // Simulated voice inquiry
  const sampleVoiceQueries = {
    "te": "ఉల్లిపాయ తామర పురుగుల నివారణకు ఏ మందు పిచికారీ చేయాలి?",
    "hi": "प्याज में थ्रिप्स और बैंगनी धब्बा रोग के लिए कौन सी दवा डालें?",
    "en": "What is the recommended spray for onion tripes and purple blotch?"
  };

  const transcribedText = sampleVoiceQueries[currentLanguage] || sampleVoiceQueries["en"];
  document.getElementById("query-input").value = transcribedText;
  sendQuery();
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.innerText = text;
  return div.innerHTML;
}
