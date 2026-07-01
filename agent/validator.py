# agent/validator.py

import json
import re


def _try_parse(text: str):
    """Try to parse JSON, return None on failure."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_json(text: str) -> dict:
    """
    Robustly extract a JSON object from an LLM response.
    Handles:
    - Markdown code fences
    - Leading/trailing prose
    - Nested JSON (reply field contains escaped JSON string)
    - Truncated nested JSON strings
    """
    text = text.strip()

    # 1. Strip markdown code fences
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)

    # 2. Find the outermost { ... } block
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        candidate = brace_match.group(0)
    else:
        candidate = text

    # 3. Try direct parse
    data = _try_parse(candidate)
    if data and isinstance(data, dict):
        # 4. Handle nested JSON: if 'reply' itself is a JSON string
        reply_val = data.get("reply", "")
        if isinstance(reply_val, str) and reply_val.strip().startswith("{"):
            inner = _try_parse(reply_val.strip())
            if inner and isinstance(inner, dict) and "reply" in inner:
                return inner
        return data

    # 5. Last resort: return raw text as reply
    return {
        "reply": text,
        "recommendations": [],
        "end_of_conversation": False,
    }


def _normalize_name(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


def _normalize_url(url: str) -> str:
    url = str(url).strip().lower()
    url = re.sub(r'^https?://(www\.)?', '', url)
    return url.rstrip('/')


def validate_response(data: dict, catalog: list[dict]) -> dict:
    """
    Enforce response schema, map raw recommendation names/URLs to canonical
    catalog items, and resolve test types to single-character codes.
    """
    reply = data.get("reply", "")
    if not isinstance(reply, str) or not reply.strip():
        reply = "I am sorry, I could not generate a response. Please try again."

    raw_recs = data.get("recommendations", [])
    if not isinstance(raw_recs, list):
        raw_recs = []

    # Build lookup dictionaries from catalog
    url_to_item = {}
    name_to_item = {}
    
    for item in catalog:
        c_url = item.get("link", "")
        c_name = item.get("name", "")
        if c_url:
            url_to_item[_normalize_url(c_url)] = item
        if c_name:
            name_to_item[_normalize_name(c_name)] = item

    # Test type map from prompt
    test_type_map = {
        "knowledge & skills": "K",
        "knowledge and skills": "K",
        "k": "K",
        "personality & behavior": "P",
        "personality and behavior": "P",
        "p": "P",
        "ability & aptitude": "A",
        "ability and aptitude": "A",
        "a": "A",
        "simulations": "S",
        "s": "S",
        "competencies": "C",
        "c": "C",
        "development & 360": "D",
        "development and 360": "D",
        "d": "D",
        "biodata & situational judgment": "B",
        "biodata and situational judgment": "B",
        "b": "B",
        "assessment exercises": "E",
        "e": "E"
    }

    clean_recs = []
    for item in raw_recs:
        if not isinstance(item, dict):
            continue

        raw_url = item.get("url", "")
        raw_name = item.get("name", "")
        raw_type = str(item.get("test_type", "")).strip().lower()

        matched_item = None

        # 1. Match by normalized URL
        norm_url = _normalize_url(raw_url) if raw_url else ""
        if norm_url and norm_url in url_to_item:
            matched_item = url_to_item[norm_url]

        # 2. Match by normalized name
        if not matched_item:
            norm_name = _normalize_name(raw_name) if raw_name else ""
            if norm_name and norm_name in name_to_item:
                matched_item = name_to_item[norm_name]

        # 3. Substring match on name
        if not matched_item and norm_name:
            for c_norm_name, c_item in name_to_item.items():
                if norm_name in c_norm_name or c_norm_name in norm_name:
                    matched_item = c_item
                    break

        if matched_item:
            # Map test type
            c_keys = matched_item.get("keys", [])
            mapped_type = ""
            if c_keys:
                primary_key = c_keys[0].strip().lower()
                mapped_type = test_type_map.get(primary_key, "")

            if not mapped_type:
                mapped_type = test_type_map.get(raw_type, raw_type.upper()[:1])

            clean_recs.append({
                "name": matched_item["name"].strip(),
                "url": matched_item["link"].strip(),
                "test_type": mapped_type if mapped_type else "K"
            })

    # Cap at 10 per spec
    clean_recs = clean_recs[:10]

    eoc = data.get("end_of_conversation", False)
    if not isinstance(eoc, bool):
        eoc = bool(eoc)

    return {
        "reply": reply.strip(),
        "recommendations": clean_recs,
        "end_of_conversation": eoc,
    }
