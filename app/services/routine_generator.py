import os
import json
from typing import Any, Dict, List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)


def _default_routine(note: str = "") -> Dict[str, Any]:
    return {
        "morning_routine": [
            "Use a gentle cleanser",
            "Apply a lightweight moisturizer",
            "Use sunscreen SPF 30 or above"
        ],
        "night_routine": [
            "Wash face with a gentle cleanser",
            "Apply a simple moisturizer"
        ],
        "natural_remedy": "Apply fresh aloe vera gel for 10 minutes, then rinse.",
        "food": "Add fruits and vegetables rich in antioxidants to your diet.",
        "hydration": "Drink 8 to 10 glasses of water daily.",
        "note": note
    }


def _normalize_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _normalize_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _validate_routine(data: Dict[str, Any]) -> Dict[str, Any]:
    validated = {
        "morning_routine": _normalize_list(data.get("morning_routine")),
        "night_routine": _normalize_list(data.get("night_routine")),
        "natural_remedy": _normalize_text(data.get("natural_remedy")),
        "food": _normalize_text(data.get("food")),
        "hydration": _normalize_text(data.get("hydration")),
        "note": _normalize_text(data.get("note"))
    }

    if not validated["morning_routine"]:
        validated["morning_routine"] = _default_routine()["morning_routine"]

    if not validated["night_routine"]:
        validated["night_routine"] = _default_routine()["night_routine"]

    if not validated["natural_remedy"]:
        validated["natural_remedy"] = _default_routine()["natural_remedy"]

    if not validated["food"]:
        validated["food"] = _default_routine()["food"]

    if not validated["hydration"]:
        validated["hydration"] = _default_routine()["hydration"]

    return validated


def generate_routine(skin_data: dict):
    confidence = skin_data.get("confidence", 1.0)
    warnings = skin_data.get("warnings", [])
    prompt = f"""
You are an expert skincare consultant.

User skin:
- Acne: {skin_data.get('acne')}
- Oiliness: {skin_data.get('oiliness')}
- Dryness: {skin_data.get('dryness')}
- Pigmentation: {skin_data.get('pigmentation')}

Confidence level: {confidence}

Warnings:
{warnings}

Instructions:

1. PERSONALIZE based on severity:
   - High acne → acne control
   - High dryness → hydration
   - High oiliness → oil control
   - High pigmentation → brightening

2. If confidence is between 0.4 and 0.7:
   - Be cautious in recommendations
   - Add a note that results may vary

3. Keep routines:
   - Max 4 steps morning
   - Max 4 steps night
   - Simple language

4. Add:
   - 1 natural remedy
   - 1 food suggestion
   - 1 hydration tip

IMPORTANT:
Return ONLY valid JSON.

Format:
{{
  "morning_routine": [],
  "night_routine": [],
  "natural_remedy": "",
  "food": "",
  "hydration": "",
  "note": ""
}}
"""

    # LOW CONFIDENCE → DO NOT GENERATE ROUTINE
    if confidence < 0.4:
        return {
            "morning_routine": [],
            "night_routine": [],
            "natural_remedy": "",
            "food": "",
            "hydration": "",
            "note": "Your image quality is too low for accurate analysis. Please retake a clearer photo in good lighting."
        }

    try:
        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise skincare routine generator that returns only valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.4
        )

        raw_output = response.choices[0].message.content.strip()

        # Remove accidental markdown fences if model adds them
        if raw_output.startswith("```"):
            raw_output = raw_output.replace("```json", "").replace("```", "").strip()

        try:
            parsed_output = json.loads(raw_output)

            if 0.4 <= confidence < 0.7:
                parsed_output["note"] = "Results may vary due to moderate image quality. For best results, use clear lighting."

            return _validate_routine(parsed_output)

        except json.JSONDecodeError:
            return _default_routine(
                note="We generated a safe fallback routine because the AI response format was invalid."
            )    

    except Exception:
        return _default_routine(
            note="We generated a safe fallback routine because the AI service was unavailable."
        )