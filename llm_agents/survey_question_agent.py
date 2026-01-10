from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from rag_system.retrieval import retrieve_guidance
from llm_agents.openrouter_client import OpenRouterClient


# -----------------------------
# Deterministic quality checks
# -----------------------------
LEADING_WORDS = {"obviously", "clearly", "surely", "best", "amazing", "terrible", "game-changing"}
ABSOLUTES = {"always", "never", "all", "none", "everyone", "no one"}
VAGUE_FREQ = {"often", "regularly", "frequently", "rarely", "sometimes"}
NEGATION_WORDS = {"not", "never", "no", "none", "without"}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def flesch_reading_ease(text: str) -> float:
    sentences = max(1, len(re.findall(r"[.!?]", text)) or 1)
    words = re.findall(r"[A-Za-z']+", text)
    n_words = max(1, len(words))

    def syllables(w: str) -> int:
        w = w.lower()
        w = re.sub(r"[^a-z]", "", w)
        if not w:
            return 0
        groups = re.findall(r"[aeiouy]+", w)
        count = len(groups)
        if w.endswith("e") and count > 1:
            count -= 1
        return max(1, count)

    n_syllables = sum(syllables(w) for w in words)
    score = 206.835 - 1.015 * (n_words / sentences) - 84.6 * (n_syllables / n_words)
    return float(score)


def evaluate_question(q: str) -> Dict[str, Any]:
    toks = _tokenize(q)
    issues: list[dict] = []

    if any(w in toks for w in LEADING_WORDS):
        issues.append({"type": "leading_language", "severity": "high", "detail": "Leading/emotionally loaded wording."})

    if any(w in toks for w in ABSOLUTES):
        issues.append({"type": "absolutes", "severity": "medium", "detail": "Contains absolute terms; may reduce validity."})

    if any(w in toks for w in VAGUE_FREQ) and not re.search(r"(past|last)\s+\d+\s+(day|week|month)s?", q.lower()):
        issues.append({"type": "vague_frequency", "severity": "medium", "detail": "Frequency wording without timeframe/anchors."})

    if any(w in toks for w in NEGATION_WORDS):
        issues.append({"type": "negation", "severity": "low", "detail": "Negation increases cognitive load; consider rewriting positively."})

    if re.search(r"\b(useful and easy|easy and useful|quality and speed|speed and quality)\b", q.lower()):
        issues.append({"type": "double_barreled", "severity": "high", "detail": "Asks about two constructs in one item."})

    fre = flesch_reading_ease(q)
    if fre < 50:
        issues.append({"type": "readability", "severity": "medium", "detail": f"Hard to read (Flesch={fre:.1f}). Shorten/simplify."})

    return {
        "question": q,
        "flesch_reading_ease": round(fre, 1),
        "issues": issues,
        "issue_count": len(issues),
    }


def propose_rewrite(q: str) -> str:
    s = q.strip()

    if any(w in _tokenize(s) for w in VAGUE_FREQ) and "past" not in s.lower() and "last" not in s.lower():
        s = s.rstrip("?") + " in the past 30 days?"

    for w in LEADING_WORDS:
        s = re.sub(rf"\b{w}\b", "", s, flags=re.IGNORECASE).replace("  ", " ").strip()

    s = re.sub(r"\b(useful and easy)\b", "useful", s, flags=re.IGNORECASE)

    return s


# -----------------------------
# OpenRouter-backed LLM wrapper
# -----------------------------
class LLMClient:
    def __init__(self):
        self.or_client = OpenRouterClient()

    def is_configured(self) -> bool:
        return self.or_client.is_configured()

    def generate(self, prompt: str) -> Optional[str]:
        if not self.is_configured():
            return None
        model = os.getenv("OPENROUTER_MODEL", "mistral/devstral-2-2512:free")
        out = self.or_client.chat(prompt=prompt, model=model, temperature=0.2)
        return out if out.strip() else None


# -----------------------------
# Agent output structures
# -----------------------------
@dataclass
class GeneratedItem:
    construct: str
    question: str
    response_scale: str
    rationale: str


@dataclass
class AgentResult:
    objectives: str
    constructs: list[str]
    generated: list[GeneratedItem]
    evaluations: list[dict]
    recommended_rewrites: list[dict]
    rag_guidance: list[dict]


DEFAULT_CONSTRUCTS = [
    "Adoption / usage",
    "Perceived usefulness",
    "Ease of use",
    "Trust & privacy concerns",
    "Reliability",
    "Productivity impact",
]


def extract_constructs(objective: str, llm: LLMClient) -> list[str]:
    if llm.is_configured():
        prompt = f"""
Extract a short list of measurable survey constructs for this research objective.
Return ONLY a JSON array of strings.

Objective:
{objective}

Rules:
- 5 to 8 constructs max
- Use survey-research language (e.g., adoption, usability, trust)
- Avoid duplicates
"""
        out = llm.generate(prompt)
        if out:
            import json
            try:
                constructs = json.loads(out.strip())
                if isinstance(constructs, list) and all(isinstance(x, str) for x in constructs):
                    return constructs
            except Exception:
                pass

    constructs = DEFAULT_CONSTRUCTS.copy()
    if re.search(r"\b(satisfaction|nps|recommend)\b", objective.lower()):
        constructs.append("Overall satisfaction / advocacy")
    if re.search(r"\b(cost|price|value)\b", objective.lower()):
        constructs.append("Value perception")

    seen = set()
    out2 = []
    for c in constructs:
        if c not in seen:
            seen.add(c)
            out2.append(c)
    return out2


def generate_questions(constructs: list[str], objective: str, llm: LLMClient) -> list[GeneratedItem]:
    if llm.is_configured():
        prompt = f"""
You are designing survey questions for a product research study.

Objective:
{objective}

Constructs:
{constructs}

Return ONLY valid JSON: an array of objects with keys:
- construct
- question
- response_scale
- rationale

Rules:
- No leading language
- One construct per question (avoid double-barreled)
- Include timeframes where appropriate (e.g., "past 30 days")
- Scales must be clearly labeled and consistent
"""
        out = llm.generate(prompt)
        if out:
            import json
            try:
                arr = json.loads(out.strip())
                items: list[GeneratedItem] = []
                if isinstance(arr, list):
                    for obj in arr:
                        if not isinstance(obj, dict):
                            continue
                        items.append(GeneratedItem(
                            construct=str(obj.get("construct", "")).strip(),
                            question=str(obj.get("question", "")).strip(),
                            response_scale=str(obj.get("response_scale", "")).strip(),
                            rationale=str(obj.get("rationale", "")).strip(),
                        ))
                    items = [i for i in items if i.construct and i.question and i.response_scale]
                    if items:
                        return items
            except Exception:
                pass

    # Deterministic fallback
    items: list[GeneratedItem] = []
    for c in constructs:
        if c == "Adoption / usage":
            items.append(GeneratedItem(c, "How often do you use Copilot at work?",
                                      "0=Never, 1=Monthly, 2=Weekly, 3=Daily",
                                      "Measures adoption frequency with anchored categories."))
        elif c == "Perceived usefulness":
            items.append(GeneratedItem(c, "To what extent does Copilot help you complete your work tasks more effectively?",
                                      "Likert 1–5 (Strongly disagree → Strongly agree)",
                                      "Captures perceived value without assuming improvement."))
        elif c == "Ease of use":
            items.append(GeneratedItem(c, "How easy is it for you to use Copilot for your typical work tasks?",
                                      "Likert 1–5 (Very difficult → Very easy)",
                                      "Measures usability with clear endpoints."))
        elif c == "Trust & privacy concerns":
            items.append(GeneratedItem(c, "How concerned are you about data privacy or security when using Copilot?",
                                      "Likert 1–5 (Not at all concerned → Extremely concerned)",
                                      "Measures barrier using neutral language."))
        elif c == "Reliability":
            items.append(GeneratedItem(c, "How reliable are Copilot’s responses for your work (e.g., accuracy and consistency)?",
                                      "Likert 1–5 (Not at all reliable → Extremely reliable)",
                                      "Defines reliability with examples."))
        elif c == "Productivity impact":
            items.append(GeneratedItem(c, "In the past 30 days, how much has Copilot affected your productivity?",
                                      "1=Decreased a lot … 3=No change … 5=Increased a lot",
                                      "Time-bounded, allows negative/neutral/positive effects."))
        else:
            items.append(GeneratedItem(c, f"Please rate: {c}", "Likert 1–5", "Generic fallback item."))
    return items


def run_survey_question_agent(objective: str) -> AgentResult:
    llm = LLMClient()
    constructs = extract_constructs(objective, llm)
    generated = generate_questions(constructs, objective, llm)

    evaluations = [evaluate_question(item.question) for item in generated]

    rewrites = []
    for ev in evaluations:
        if ev["issue_count"] > 0:
            rewrites.append({
                "original": ev["question"],
                "suggested": propose_rewrite(ev["question"]),
                "issues": ev["issues"]
            })

    query = objective + " " + " ".join([iss["type"] for ev in evaluations for iss in ev["issues"]])
    hits = retrieve_guidance(query)
    rag_guidance = [{"source": h.source, "score": round(h.score, 3), "snippet": h.snippet} for h in hits]

    return AgentResult(
        objectives=objective,
        constructs=constructs,
        generated=generated,
        evaluations=evaluations,
        recommended_rewrites=rewrites,
        rag_guidance=rag_guidance
    )
