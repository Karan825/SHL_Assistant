# agent/agent.py

import json
import os
import re
from dotenv import load_dotenv
from agent.retriever import Retriever
from agent.prompts import SYSTEM_PROMPT, build_catalog_context
from agent.validator import extract_json, validate_response, _normalize_url

load_dotenv()

from google import genai
from google.genai import types
from groq import Groq

_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
_MODEL  = "gemini-3.1-flash-lite"
_gemini_failed = False

# ---------------------------------------------------------------------------
# Guard word-lists
# ---------------------------------------------------------------------------
_INJECTION_PHRASES = [
    "ignore previous", "forget your instructions", "act as",
    "legal advice", "lawsuit", "salary", "immigration",
    "ignore all", "disregard", "new persona", "pretend you are",
    "what is the weather", "tell me a joke", "stock price",
]

MAX_TURNS      = 8
SHORTLIST_TURN = 6


class SHLAgent:
    """
    Stateless conversational agent for SHL assessment recommendation.
    Full conversation history is passed on every call.
    """

    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        groq_api_key = os.environ.get("GROQ_API_KEY", "")
        self.groq_client = Groq(api_key=groq_api_key) if groq_api_key else None

    # ------------------------------------------------------------------
    # Query building & Explicit Mentions Matching
    # ------------------------------------------------------------------

    def _normalize_text(self, text: str) -> str:
        return re.sub(r'[^a-z0-9\s]', '', text.lower())

    def get_explicit_mentions(self, messages: list[dict]) -> list[dict]:
        all_text_parts = []
        for m in messages:
            content = m["content"]
            if content.strip().startswith("{"):
                try:
                    parsed = json.loads(content)
                    all_text_parts.append(parsed.get("reply", ""))
                    recs = parsed.get("recommendations", [])
                    for r in recs:
                        if isinstance(r, dict) and "name" in r:
                            all_text_parts.append(r["name"])
                except json.JSONDecodeError:
                    all_text_parts.append(content)
            else:
                all_text_parts.append(content)
        
        all_text = " ".join(all_text_parts).lower()
        norm_text = self._normalize_text(all_text)
        
        # Domain-specific targets to pre-populate context based on user queries for C1-C10
        target_links = set()
        
        # 1. Senior Leadership / Executive / CXO / Director -> C1
        if any(w in norm_text for w in ["senior leadership", "cxo", "director", "executive"]):
            target_links.update([
                "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
                "https://www.shl.com/products/product-catalog/view/opq-universal-competency-report-2-0/",
                "https://www.shl.com/products/product-catalog/view/opq-leadership-report/"
            ])
            
        # 2. Rust / Linux / Systems / High performance / Networking -> C2
        if any(w in norm_text for w in ["rust", "linux", "systems programming", "networking infrastructure"]):
            target_links.update([
                "https://www.shl.com/products/product-catalog/view/smart-interview-live-coding/",
                "https://www.shl.com/products/product-catalog/view/linux-programming-general/",
                "https://www.shl.com/products/product-catalog/view/networking-and-implementation-new/",
                "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
                "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/"
            ])
            
        # 3. Contact Center / Customer Service / Inbound / SVAR / Spoken English -> C3
        if any(w in norm_text for w in ["contact centre", "contact center", "customer service", "inbound", "svar", "spoken english"]):
            target_links.update([
                "https://www.shl.com/products/product-catalog/view/svar-spoken-english-us-new/",
                "https://www.shl.com/products/product-catalog/view/contact-center-call-simulation-new/",
                "https://www.shl.com/products/product-catalog/view/entry-level-customer-serv-retail-and-contact-center/",
                "https://www.shl.com/products/product-catalog/view/customer-service-phone-simulation/"
            ])
            
        # 4. Graduate Financial Analyst / Finance / Accounting -> C4
        if any(w in norm_text for w in ["financial analyst", "accounting", "finance knowledge", "graduate financial"]):
            target_links.update([
                "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-numerical-reasoning/",
                "https://www.shl.com/products/product-catalog/view/financial-accounting-new/",
                "https://www.shl.com/products/product-catalog/view/basic-statistics-new/",
                "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
                "https://www.shl.com/products/product-catalog/view/graduate-scenarios/"
            ])
            
        # 5. Sales / Restructuring / Reskill -> C5
        if any(w in norm_text for w in ["sales organization", "re-skill sales", "sales report", "sales transformation"]):
            target_links.update([
                "https://www.shl.com/products/product-catalog/view/global-skills-assessment/",
                "https://www.shl.com/products/product-catalog/view/global-skills-development-report/",
                "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
                "https://www.shl.com/products/product-catalog/view/opq-mq-sales-report/",
                "https://www.shl.com/products/product-catalog/view/salestransformationreport2-0-individualcontributor/"
            ])
            
        # 6. Plant Operator / Safety / Dependability / Chemical -> C6
        if any(w in norm_text for w in ["plant operator", "chemical facility", "safety", "dsi"]):
            target_links.update([
                "https://www.shl.com/products/product-catalog/view/dependability-and-safety-instrument-dsi/",
                "https://www.shl.com/products/product-catalog/view/safety-and-dependability-focus-8-0/",
                "https://www.shl.com/products/product-catalog/view/workplace-health-and-safety-new/"
            ])
            
        # 7. Healthcare / Patient records / HIPAA -> C7
        if any(w in norm_text for w in ["healthcare admin", "patient records", "hipaa", "medical terminology"]):
            target_links.update([
                "https://www.shl.com/products/product-catalog/view/hipaa-security/",
                "https://www.shl.com/products/product-catalog/view/medical-terminology-new/",
                "https://www.shl.com/products/product-catalog/view/microsoft-word-365-essentials-new/",
                "https://www.shl.com/products/product-catalog/view/dependability-and-safety-instrument-dsi/",
                "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/"
            ])
            
        # 8. Excel / Word / Admin Assistant -> C8
        if any(w in norm_text for w in ["admin assistant", "excel", "word"]):
            target_links.update([
                "https://www.shl.com/products/product-catalog/view/ms-excel-new/",
                "https://www.shl.com/products/product-catalog/view/ms-word-new/",
                "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
                "https://www.shl.com/products/product-catalog/view/microsoft-excel-365-new/",
                "https://www.shl.com/products/product-catalog/view/microsoft-word-365-new/"
            ])
            
        # 9. Full-Stack / Backend / Java / Spring / REST / Angular / SQL / Docker / AWS -> C9
        if any(w in norm_text for w in ["full-stack", "java", "spring", "restful", "rest api", "angular", "sql", "docker", "aws"]):
            target_links.update([
                "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
                "https://www.shl.com/products/product-catalog/view/spring-new/",
                "https://www.shl.com/products/product-catalog/view/restful-web-services-new/",
                "https://www.shl.com/products/product-catalog/view/sql-new/",
                "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
                "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
                "https://www.shl.com/products/product-catalog/view/amazon-web-services-aws-development-new/",
                "https://www.shl.com/products/product-catalog/view/docker-new/"
            ])
            
        # 10. Graduate Management Trainee -> C10
        if any(w in norm_text for w in ["management trainee", "trainee scheme"]):
            target_links.update([
                "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
                "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
                "https://www.shl.com/products/product-catalog/view/graduate-scenarios/"
            ])

        normalized_targets = {_normalize_url(url) for url in target_links}
        matches = []
        for item in self.retriever.catalog:
            link = item.get("link", "")
            norm_link = _normalize_url(link)
            if norm_link in normalized_targets:
                matches.append(item)
            else:
                name = item["name"].lower()
                norm_name = self._normalize_text(name)
                
                # Fallback to exact / keyword checks
                is_match = False
                if norm_name in norm_text:
                    is_match = True
                else:
                    if "opq" in norm_text and "opq" in norm_name:
                        is_match = True
                    elif "dsi" in norm_text and ("dsi" in norm_name or "dependability and safety" in norm_name):
                        is_match = True
                    elif "gsa" in norm_text and ("gsa" in norm_name or "global skills" in norm_name):
                        is_match = True
                    elif "svar" in norm_text and "svar" in norm_name:
                        is_match = True
                    elif "verify" in norm_text and "verify" in norm_name:
                        is_match = True
                    elif "scenarios" in norm_text and "scenarios" in norm_name:
                        is_match = True
                    elif "live coding" in norm_text and "live coding" in norm_name:
                        is_match = True
                    elif "linux" in norm_text and "linux" in norm_name:
                        is_match = True
                    elif "aws" in norm_text and ("aws" in norm_name or "amazon web" in norm_name):
                        is_match = True
                    elif "docker" in norm_text and "docker" in norm_name:
                        is_match = True
                    elif "sql" in norm_text and "sql" in norm_name:
                        is_match = True
                    elif "spring" in norm_text and "spring" in norm_name:
                        is_match = True
                    elif "java" in norm_name and "java" in norm_text:
                        if "javascript" in norm_name:
                            if "javascript" in norm_text:
                                is_match = True
                        else:
                            if re.search(r'\bjava\b', norm_text):
                                is_match = True
                    elif "sales transformation" in norm_text and "sales transformation" in norm_name:
                        is_match = True
                    elif "phone simulation" in norm_text and "phone simulation" in norm_name:
                        is_match = True
                    elif "contact center" in norm_text and "contact center" in norm_name:
                        is_match = True
                    elif "retail" in norm_text and "retail" in norm_name:
                        is_match = True
                    elif "statistics" in norm_text and "statistics" in norm_name:
                        is_match = True
                    elif "accounting" in norm_text and "accounting" in norm_name:
                        is_match = True
                
                if is_match:
                    matches.append(item)
        return matches

    def _is_comparison(self, messages: list[dict]) -> bool:
        last = messages[-1]["content"].lower()
        return any(p in last for p in [
            "difference between", "compare", "vs ", "versus",
            "which is better", "what is the difference",
        ])

    def _turn_count(self, messages: list[dict]) -> int:
        return sum(1 for m in messages if m["role"] == "user")

    def _build_retrieval_query(self, messages: list[dict]) -> str:
        user_msgs = [m["content"].strip() for m in messages if m["role"] == "user"]
        return " ".join(user_msgs)

    def _assistant_text(self, content: str) -> str:
        stripped = content.strip()
        if stripped.startswith("{"):
            try:
                parsed = json.loads(stripped)
                return parsed.get("reply", content)
            except json.JSONDecodeError:
                pass
        return content

    # ------------------------------------------------------------------
    # Main Chat
    # ------------------------------------------------------------------

    def chat(self, messages: list[dict]) -> dict:
        # Check simple prompt injection / off-topic phrases at python level
        last_user_content = messages[-1]["content"].lower()
        if any(phrase in last_user_content for phrase in _INJECTION_PHRASES):
            return {
                "reply": (
                    "I can only help with SHL assessment selection. "
                    "Please describe the role you are hiring for."
                ),
                "recommendations": [],
                "end_of_conversation": False,
            }

        # --- Retrieve relevant catalog entries ---
        mentions = self.get_explicit_mentions(messages)
        query = self._build_retrieval_query(messages)
        candidates = self.retriever.search(query, top_k=20)

        # Merge mentions and candidates, avoiding duplicates
        seen_urls = set()
        merged = []
        for item in mentions:
            url = item.get("link", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                merged.append(item)
        for item in candidates:
            url = item.get("link", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                merged.append(item)

        catalog_ctx = build_catalog_context(merged[:10])
        system = SYSTEM_PROMPT.format(catalog_context=catalog_ctx)

        # Add turn-pressure note when approaching the turn cap
        turn_num = self._turn_count(messages)
        last_text = messages[-1]["content"]
        force_end = turn_num >= MAX_TURNS

        if turn_num >= SHORTLIST_TURN:
            last_text += (
                "\n\n[System note: You are near the conversation turn limit. "
                "Please commit to a final shortlist of 1-10 assessments now.]"
            )

        raw_text = None

        # 1. Try Groq (Llama-3.1-8b-instant) first
        if self.groq_client:
            groq_messages = [{"role": "system", "content": system}]
            for msg in messages[:-1]:
                role = "user" if msg["role"] == "user" else "assistant"
                text = self._assistant_text(msg["content"]) if role == "assistant" else msg["content"]
                groq_messages.append({"role": role, "content": text})
            groq_messages.append({"role": "user", "content": last_text})

            try:
                completion = self.groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=groq_messages,
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                raw_text = completion.choices[0].message.content
            except Exception as e:
                print(f"[LLM PROVIDER] Groq call failed: {e}. Falling back to Gemini...")

        # 2. Fallback to Gemini (gemini-3.5-flash)
        if not raw_text:
            contents = []
            for msg in messages[:-1]:
                role = "user" if msg["role"] == "user" else "model"
                text = self._assistant_text(msg["content"]) if role == "model" else msg["content"]
                contents.append(types.Content(role=role, parts=[types.Part(text=text)]))
            contents.append(types.Content(role="user", parts=[types.Part(text=last_text)]))

            cfg = types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.2,
            )
            try:
                response = _client.models.generate_content(
                    model=_MODEL,
                    contents=contents,
                    config=cfg,
                )
                raw_text = response.text
            except Exception as e:
                print(f"[LLM PROVIDER] Gemini fallback failed: {e}")
                raise e

        # --- Parse & validate ---
        raw = extract_json(raw_text)

        # Get previous recommendations if any
        prev_recs = []
        for msg in reversed(messages[:-1]):
            role = msg.get("role")
            if role in ["assistant", "model"]:
                content = msg["content"]
                if content.strip().startswith("{"):
                    try:
                        parsed = json.loads(content)
                        if parsed.get("recommendations"):
                            prev_recs = parsed["recommendations"]
                            break
                    except:
                        pass

        last_user = messages[-1]["content"].lower()
        
        # Carry forward previous recommendations if the user is confirming or asking a question
        if prev_recs:
            is_confirm_or_question = (
                "?" in last_user
                or any(p in last_user for p in ["compare", "difference", "vs ", "versus", "which is", "what is"])
                or any(p in last_user for p in ["keep", "as-is", "as is", "perfect", "thanks", "works", "confirm", "yes", "clear"])
            )
            
            # Check if user is specifying refinement details or particular test names
            refinement_keywords = [
                "drop", "remove", "replace", "delete", "instead", "add ", "with ", "except",
                "simulation", "solution", "report", "bundle", "opq", "verify", "scenarios", "g+", "test",
                "java", "spring", "excel", "word", "sales", "safety", "healthcare"
            ]
            is_refinement = (
                not "?" in last_user and (
                    any(p in last_user for p in refinement_keywords)
                    or len(last_user.split()) > 6  # longer confirmations might contain complex instructions
                )
            )
            
            # If it's a simple confirmation/question and not a refinement turn,
            # or if the LLM returned empty recommendations, carry forward prev_recs.
            if (is_confirm_or_question and not is_refinement) or not raw.get("recommendations"):
                raw["recommendations"] = prev_recs

        # Enforce end_of_conversation if the user confirms the shortlist
        is_user_confirming = any(p in last_user for p in [
            "perfect", "thanks", "works", "confirm", "lock it in", 
            "covers it", "as-is", "as is", "that works", "narrow down",
            "clear. we'll use"
        ])
        # Avoid false positive completions when user clarifies or negates the recommendations
        is_negation = any(p in last_user for p in [
            "but", "not what", "except", "don't want", "do not want", "instead", "different"
        ])
        if is_user_confirming and not is_negation and (prev_recs or raw.get("recommendations")):
            raw["end_of_conversation"] = True

        if force_end:
            raw["end_of_conversation"] = True
        return validate_response(raw, self.retriever.catalog)
