# KisanSathi Phase 3: AI Intent & Location-Aware Tool Routing Evaluation

**Date:** 2026-11-11 22:22:54  
**Evaluation Scope:** Location-based weather, mandi prices, price watchlists, and agronomic advisory.  
**Network Independence:** 100% offline executable using deterministic geographic centroids and Agmarknet seed records.

## 1. Executive Performance Metrics

| Benchmark Metric              | Value                                 |
|-------------------------------|---------------------------------------|
| Total Benchmark Queries       | 30                                    |
| Supported Languages           | English, Telugu, Hindi                |
| Tool Selection Accuracy       | 100.0%                                |
| Parameter Extraction Accuracy | 100.0%                                |
| Average Routing Latency       | 0.04 ms                               |
| Evaluation Network Mode       | 100% Offline (Deterministic & Mocked) |

## 2. Intent Classification Breakdown

| Intent Domain   |   Total Queries |   Correctly Classified | Accuracy Rate   |
|-----------------|-----------------|------------------------|-----------------|
| WEATHER         |              11 |                     11 | 100.0%          |
| MANDI           |              11 |                     11 | 100.0%          |
| WATCHLIST       |               5 |                      5 | 100.0%          |
| RAG             |               3 |                      3 | 100.0%          |

## 3. Detailed Query Breakdown (30 Test Cases)

|   # | Lang   | Inquiry Excerpt                             | Expected Intent   | Detected Intent   | Resolved Place     | Extracted Crop   | Intent Status   | Param Acc   |
|-----|--------|---------------------------------------------|-------------------|-------------------|--------------------|------------------|-----------------|-------------|
|   1 | EN     | Will it rain tomorrow in Nashik?            | weather           | weather           | Nashik             | onion            | PASS            | 100%        |
|   2 | EN     | What is the temperature and humidity in ... | weather           | weather           | Pune               | onion            | PASS            | 100%        |
|   3 | EN     | Is the wind speed safe for spraying pest... | weather           | weather           | Rahata             | onion            | PASS            | 100%        |
|   4 | EN     | What is the 7-day rainfall outlook for m... | weather           | weather           | (Profile Fallback) | onion            | PASS            | 100%        |
|   5 | EN     | Will there be heavy showers in Ahmednaga... | weather           | weather           | Ahmednagar         | onion            | PASS            | 100%        |
|   6 | TE     | నాసిక్‌లో రేపు వర్షం పడుతుందా? పిచికారీ ... | weather           | weather           | Nashik             | onion            | PASS            | 100%        |
|   7 | TE     | మా ఊరిలో రాబోయే రెండు రోజుల్లో వాతావరణం ... | weather           | weather           | (Profile Fallback) | onion            | PASS            | 100%        |
|   8 | TE     | పుణెలో వర్షపాతం మరియు ఉష్ణోగ్రత ఎంత?        | weather           | weather           | Pune               | onion            | PASS            | 100%        |
|   9 | HI     | कल नासिक में बारिश होगी क्या? क्या दवा क... | weather           | weather           | Nashik             | onion            | PASS            | 100%        |
|  10 | HI     | पुणे का मौसम कैसा रहेगा कल?                 | weather           | weather           | Pune               | onion            | PASS            | 100%        |
|  11 | HI     | क्या आज लासलगाव में तेज हवा चलने की संभा... | weather           | weather           | Lasalgaon          | onion            | PASS            | 100%        |
|  12 | EN     | What is the modal price of onion in Lasa... | mandi             | mandi             | Lasalgaon          | onion            | PASS            | 100%        |
|  13 | EN     | Give me the latest tomato market rates i... | mandi             | mandi             | Pimpalgaon         | tomato           | PASS            | 100%        |
|  14 | EN     | Which nearby mandi gives the highest net... | mandi             | mandi             | (Profile Fallback) | soybean          | PASS            | 100%        |
|  15 | EN     | What is the price of cotton per quintal ... | mandi             | mandi             | Kalamna            | cotton           | PASS            | 100%        |
|  16 | EN     | Show me nearest market rates for onion      | mandi             | mandi             | (Profile Fallback) | onion            | PASS            | 100%        |
|  17 | TE     | లాసల్‌గావ్‌లో ఉల్లిపాయ క్వింటాల్ ధర ఎంత?    | mandi             | mandi             | Lasalgaon          | onion            | PASS            | 100%        |
|  18 | TE     | నా టమాటా పంటకు సమీప మార్కెట్లలో రేటు ఎంత... | mandi             | mandi             | (Profile Fallback) | tomato           | PASS            | 100%        |
|  19 | TE     | పింపల్‌గావ్‌లో టమాటా మోడల్ ధర ఎంత?          | mandi             | mandi             | Pimpalgaon         | tomato           | PASS            | 100%        |
|  20 | HI     | लासलगाव मंडी में आज प्याज का भाव क्या है... | mandi             | mandi             | Lasalgaon          | onion            | PASS            | 100%        |
|  21 | HI     | पिंपलगांव में टमाटर का ताजा मंडी भाव बता... | mandi             | mandi             | Pimpalgaon         | tomato           | PASS            | 100%        |
|  22 | HI     | सोयाबीन का नजदीकी मंडियों में क्या रेट म... | mandi             | mandi             | (Profile Fallback) | soybean          | PASS            | 100%        |
|  23 | EN     | Alert me when onion price crosses 2200 i... | watchlist         | watchlist         | (Profile Fallback) | onion            | PASS            | 100%        |
|  24 | EN     | Tell me when tomato crosses 1800 in Pimp... | watchlist         | watchlist         | Pimpalgaon         | tomato           | PASS            | 100%        |
|  25 | TE     | ఉల్లిపాయ ధర 2500 దాటినప్పుడు నాకు వాట్సా... | watchlist         | watchlist         | (Profile Fallback) | onion            | PASS            | 100%        |
|  26 | TE     | టమాటా ధర 1500 దాటితే నాకు తెలపండి           | watchlist         | watchlist         | (Profile Fallback) | tomato           | PASS            | 100%        |
|  27 | HI     | प्याज का भाव 2000 पार होते ही मुझे अलर्ट... | watchlist         | watchlist         | (Profile Fallback) | onion            | PASS            | 100%        |
|  28 | EN     | How do I control purple blotch in my oni... | rag               | rag               | (Profile Fallback) | onion            | PASS            | 100%        |
|  29 | TE     | టమాటా ఆకుమచ్చ తెగులు నివారణకు ఏ మందు పిచ... | rag               | rag               | (Profile Fallback) | tomato           | PASS            | 100%        |
|  30 | HI     | सोयाबीन में पीला मोजेक रोग नियंत्रण के क... | rag               | rag               | (Profile Fallback) | soybean          | PASS            | 100%        |

## 4. Key Architectural Findings

- **Zero-Query Profiling:** When farmers ask queries without specifying location or crop (e.g. *'Show me nearest market rates for onion'*), the router automatically defaults to the stored profile village and crop coordinates.
- **Named Location Overrides:** Inquiries explicitly referencing agricultural hubs (e.g. *'Lasalgaon'*, *'पिंपलगांव'*, *'పుణె'*) accurately override stored defaults for that single query without modifying the permanent farmer profile.
- **Multilingual Script Resilience:** The hybrid regex and thesaurus parser robustly resolves Devanagari (*लासलगाव*, *पिंपलगांव*) and Telugu script (*లాసల్‌గావ్*, *పింపల్‌గావ్*) without external geocoding latency.
- **Watchlist Triggering:** Multi-directional monetary threshold extraction extracts price targets regardless of whether the verb precedes (*'crosses 2200'*) or succeeds (*'2500 దాటినప్పుడు'*, *'2000 पार'*) the number.
