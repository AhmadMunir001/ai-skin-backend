import os
import json
from typing import Any, Dict, List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
)

MODEL = "openai/gpt-3.5-turbo"


# ──────────────────────────────────────────────
# DEFAULTS & HELPERS
# ──────────────────────────────────────────────

def _default_routine(note: str = "") -> Dict[str, Any]:
    return {
        "skin_type_summary": "Combination skin — gentle balanced care recommended.",
        "morning_routine": [
            "Cleanse with a gentle, pH-balanced foaming cleanser",
            "Apply a hydrating toner (avoid alcohol-based)",
            "Use a lightweight moisturizer with niacinamide",
            "Finish with broad-spectrum SPF 30+ sunscreen"
        ],
        "night_routine": [
            "Double-cleanse: micellar water first, then gentle cleanser",
            "Apply a nourishing serum (Vitamin C or hyaluronic acid)",
            "Use a slightly richer night moisturizer or sleeping mask"
        ],
        "weekly_treatments": [
            "Exfoliate gently 1–2x per week (BHA for oily areas, AHA for dry)",
            "Apply a clay mask on the T-zone once a week"
        ],
        "natural_remedy": "Mix 1 tsp raw honey + 2 drops tea tree oil; apply for 10 min, rinse.",
        "food": "Include turmeric milk, papaya, and leafy greens daily for skin health.",
        "hydration": "Drink 8–10 glasses of water. Add sabza (basil seeds) to water for extra hydration.",
        "ingredients_to_use": ["niacinamide", "hyaluronic acid", "SPF 30+"],
        "ingredients_to_avoid": ["alcohol denat.", "heavy mineral oil", "harsh sulfates"],
        "note": note
    }


def _normalize_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(i).strip() for i in value if str(i).strip()]
    return []


def _normalize_text(value: Any, default: str = "") -> str:
    if not value:
        return default
    t = str(value).strip()
    return t if t else default


def _validate_routine(data: Dict[str, Any]) -> Dict[str, Any]:
    d = _default_routine()
    return {
        "skin_type_summary":  _normalize_text(data.get("skin_type_summary"), d["skin_type_summary"]),
        "morning_routine":    _normalize_list(data.get("morning_routine")) or d["morning_routine"],
        "night_routine":      _normalize_list(data.get("night_routine")) or d["night_routine"],
        "weekly_treatments":  _normalize_list(data.get("weekly_treatments")) or d["weekly_treatments"],
        "natural_remedy":     _normalize_text(data.get("natural_remedy"), d["natural_remedy"]),
        "food":               _normalize_text(data.get("food"), d["food"]),
        "hydration":          _normalize_text(data.get("hydration"), d["hydration"]),
        "ingredients_to_use": _normalize_list(data.get("ingredients_to_use")) or d["ingredients_to_use"],
        "ingredients_to_avoid": _normalize_list(data.get("ingredients_to_avoid")) or d["ingredients_to_avoid"],
        "note":               _normalize_text(data.get("note"), ""),
    }


# ──────────────────────────────────────────────
# PROMPT BUILDER
# ──────────────────────────────────────────────

def _build_prompt(skin_data: dict) -> str:
    acne   = skin_data.get("acne", "medium")
    oil    = skin_data.get("oiliness", "medium")
    dry    = skin_data.get("dryness", "medium")
    pig    = skin_data.get("pigmentation", "medium")
    sens   = skin_data.get("sensitivity", "medium")
    conf   = skin_data.get("confidence", 1.0)
    warns  = skin_data.get("warnings", [])
    scores = skin_data.get("scores", {})
    regions = skin_data.get("regions_analyzed", [])

    # Determine dominant concern for extra focus
    concern_map = {"acne": acne, "oiliness": oil, "dryness": dry, "pigmentation": pig}
    severity_order = {"high": 3, "medium": 2, "low": 1}
    dominant = max(concern_map, key=lambda k: severity_order.get(concern_map[k], 0))

    return f"""
You are a certified K-beauty dermatologist and skincare expert specializing in South Asian skin, \
particularly Pakistani women living in hot, humid, and polluted climates (cities like Karachi, Lahore, Faisalabad).

== PATIENT SKIN ANALYSIS ==
- Acne: {acne}
- Oiliness: {oil}
- Dryness: {dry}
- Pigmentation: {pig}
- Sensitivity: {sens}
- Dominant concern: {dominant}
- Confidence: {conf}
- Regions analyzed: {', '.join(regions) if regions else 'full face'}
- Raw scores: {json.dumps(scores)}
- Warnings: {warns}
- Skin Tone (ITA): {skin_data.get('skin_tone', 'unknown')}
- Redness Level: {skin_data.get('redness', 'medium')}  
- Pore Size: {skin_data.get('pore_size', 'moderate')}
- Skin Zone Type: {skin_data.get('skin_zone_type', 'unknown')}

== INSTRUCTIONS ==

1. Write a 1-sentence skin_type_summary (e.g., "You have oily-acne prone skin with mild pigmentation").

2. morning_routine: exactly 4 steps. Each step = one action sentence with a specific product type and active ingredient where useful.
   - For high acne: include salicylic acid or benzoyl peroxide cleanser
   - For high oiliness: include niacinamide serum, lightweight gel moisturizer
   - For high dryness: include hyaluronic acid, ceramide moisturizer
   - For high pigmentation: include Vitamin C serum, SPF 50
   - Always end with SPF (Pakistani sun is intense)

3. night_routine: exactly 3 steps.
   - For high acne: include retinol or BHA at night
   - For sensitivity: avoid retinol, use centella asiatica or cica

4. weekly_treatments: 2 items. Specify frequency (e.g., "1x per week").

5. natural_remedy: 1 remedy using easily available Pakistani ingredients \
(e.g., multani mitti, rose water, aloe vera, turmeric, honey, besan, neem).

6. food: 1-2 sentences. Mention specific Pakistani foods (e.g., amla, methi, sabja, papaya, dahi, haldi doodh).

7. hydration: 1 sentence. Mention specific Pakistani drinks/tips (e.g., nimbu pani, coconut water, sabza).

8. ingredients_to_use: list 3–5 key ingredients as short strings.

9. ingredients_to_avoid: list 2–4 ingredients to avoid based on skin type.

10. note: empty string "" unless confidence is between 0.4–0.7 (then add a one-line caution).

== RULES ==
- Max 4 morning steps, max 3 night steps
- No brand names — only ingredient/product type
- Simple English, no medical jargon
- Climate context: hot humid summers, dry winters, high UV, pollution

Return ONLY valid JSON, no markdown, no explanation:
{{
  "skin_type_summary": "",
  "morning_routine": [],
  "night_routine": [],
  "weekly_treatments": [],
  "natural_remedy": "",
  "food": "",
  "hydration": "",
  "ingredients_to_use": [],
  "ingredients_to_avoid": [],
  "note": ""
}}
"""


# ──────────────────────────────────────────────
# MAIN FUNCTION
# ──────────────────────────────────────────────

def generate_routine(skin_data: dict) -> Dict[str, Any]:
    confidence = skin_data.get("confidence", 1.0)

    # Too blurry / bad image — skip LLM entirely
    if confidence < 0.4:
        return {
            **_default_routine(),
            "note": "Image quality too low for accurate analysis. Please retake in clear, natural lighting facing the camera directly."
        }

    prompt = _build_prompt(skin_data)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise skincare JSON generator. "
                        "Return ONLY valid JSON with no extra text, no markdown code fences."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.35,
            max_tokens=900,
        )

        raw = response.choices[0].message.content.strip()

        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)

        # Inject confidence caution
        if 0.4 <= confidence < 0.7:
            parsed["note"] = "Results may vary due to moderate image quality. For best accuracy, use clear natural lighting."

        return _validate_routine(parsed)

    except json.JSONDecodeError:
        return _default_routine(
            note="Routine generated from safe defaults — AI response format was invalid."
        )
    except Exception:
        return _default_routine(
            note="Routine generated from safe defaults — AI service was temporarily unavailable."
        )