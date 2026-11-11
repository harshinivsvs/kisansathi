# KisanSathi: Free API Keys Setup Guide

This guide provides step-by-step instructions for acquiring the **100% free-tier API keys** required to run KisanSathi in live production or development mode.

> [!NOTE]
> KisanSathi is designed with **zero-cost offline fallbacks**. The system will run completely and pass all benchmarks and tests without any internet connection or API keys. Adding these keys connects the system to live satellite feeds, live mandi arrivals, WhatsApp messaging, and Google Gemini LLMs.

---

## 1. Google Gemini API Key (LLM Generation)

**Tier:** Free (up to 15 Requests Per Minute / 1 million tokens/min on `gemini-1.5-flash`).  
**Purpose:** Answers grounded agronomic questions, extracts entities, and formats personalized farmer advice.

### Step-by-Step Instructions:
1. Navigate to [Google AI Studio](https://aistudio.google.com/).
2. Sign in with your Google Account.
3. Click on the blue button **"Get API key"** in the top left navigation menu.
4. Click **"Create API key"** in a new project (or select an existing Google Cloud project).
5. Copy the generated key (starts with `AIzaSy...`).
6. Paste into your `.env` file:
   ```env
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=AIzaSyYourGeneratedGeminiKeyHere
   LLM_MODEL_NAME=gemini-1.5-flash
   ```

---

## 2. OpenWeatherMap API Key (Location-Based Weather)

**Tier:** Free ("Free" plan includes 1,000 API calls/day or 60 calls/minute, Current Weather & 5-day / 3-hour forecast).  
**Purpose:** Fetches current temperature, humidity, wind speed, and rain probability using farmer coordinates with 30-min grid caching.

### Step-by-Step Instructions:
1. Visit [OpenWeather Sign Up](https://home.openweathermap.org/users/sign_up).
2. Enter your email and create a password.
3. Verify your email address by clicking the link sent to your inbox.
4. Log into [OpenWeather Dashboard](https://home.openweathermap.org/).
5. Click on your profile name at the top right and select **"My API keys"**.
6. Give your key a name (e.g. `KisanSathi-Weather`) and click **"Generate"**.
7. *Note:* New OpenWeather keys typically activate within 10 to 15 minutes.
8. Copy the 32-character hexadecimal key and paste into `.env`:
   ```env
   OPENWEATHER_API_KEY=your_32_character_openweather_key
   ```

---

## 3. Data.gov.in API Key (Agmarknet Mandi Prices)

**Tier:** Free (Open Government Data Platform India).  
**Purpose:** Accesses real-time daily modal, min, and max agricultural commodity prices across Indian APMCs (e.g., Lasalgaon, Pimpalgaon, Rahata).

### Step-by-Step Instructions:
1. Navigate to the [data.gov.in Registration Portal](https://www.data.gov.in/user/register).
2. Enter your name, mobile number, and email address, and complete OTP verification.
3. Log in and navigate to **"My Account"** -> **"API Key"**.
4. Accept the National Data Sharing and Accessibility Policy (NDSAP) terms.
5. Click **"Generate API Key"**.
6. Copy the API key and paste into `.env`:
   ```env
   DATA_GOV_IN_API_KEY=your_data_gov_in_api_key
   ```
*(KisanSathi comes pre-seeded with `data/raw/maharashtra_mandis_master.json` and `data/raw/mandi_prices_seed.json` for all 20 Maharashtra APMCs, so local testing works even without this key).*

---

## 4. Meta WhatsApp Cloud API (WhatsApp Webhook & Messaging)

**Tier:** Free (First 1,000 service conversations every month are completely free; free sandbox test numbers included).  
**Purpose:** Receives incoming farmer audio voice notes, location pins, and text messages, and returns audio voice notes and interactive button responses.

### Step-by-Step Instructions:
1. Go to the [Meta for Developers Portal](https://developers.facebook.com/).
2. Log in with your Facebook account and click **"My Apps"** -> **"Create App"**.
3. Select **"Other"** -> **"Business"** as the app type.
4. Name your app (e.g. `KisanSathi-FarmBot`) and provide your email.
5. In the App Dashboard, scroll down to **"WhatsApp"** and click **"Set up"**.
6. In the left sidebar under WhatsApp, click **"API Setup"**:
   - You will see a temporary **Access Token** and a test **Phone number ID**.
   - Under "To", add your personal WhatsApp phone number and verify it with the 6-digit code.
   - Click "Send message" to verify that test messaging works.
7. Set up the Webhook:
   - In WhatsApp -> **"Configuration"**, click **"Edit"** next to Webhook.
   - Set **Callback URL**: `https://<your-ngrok-or-domain>/api/v1/whatsapp/webhook`
   - Set **Verify Token**: `kisansathi_verify_token` (must match `WHATSAPP_VERIFY_TOKEN` in `.env`).
   - Under Webhook fields, click **"Manage"** and subscribe to `messages`.
8. Copy the credentials into `.env`:
   ```env
   WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id
   WHATSAPP_ACCESS_TOKEN=your_access_token
   WHATSAPP_VERIFY_TOKEN=kisansathi_verify_token
   WHATSAPP_APP_SECRET=your_app_secret
   ```

---

## Verification Checklist

Run the following command in terminal to verify that your environment variables and fallbacks are functional:

```powershell
.\venv\Scripts\python -c "from backend.config import settings; print('App:', settings.APP_NAME); print('LLM:', settings.LLM_PROVIDER); print('OpenWeather configured:', bool(settings.OPENWEATHER_API_KEY))"
```

To run the automated test suite verifying both live and mocked fallback modes:

```powershell
.\venv\Scripts\pytest tests/ -v
```
