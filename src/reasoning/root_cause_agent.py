import json
import os
from typing import Dict, Any, Optional


def _confidence_level(confidence: float) -> str:
    if confidence >= 0.7:
        return "high"
    if confidence >= 0.4:
        return "moderate"
    return "low"


def _fallback_rules(defect_class: str, confidence: float) -> Dict[str, Any]:
    confidence_level = _confidence_level(confidence)

    defect_knowledge = {
        "center": {
            "pattern_reasoning": (
                "The defect pattern is concentrated near the wafer center, "
                "which typically correlates with non-uniform thermal or pressure conditions "
                "during central process phases."
            ),
            "cause": (
                "Instability in center-zone temperature or chamber pressure "
                "during deposition or etching steps."
            ),
            "action": (
                "Review thermal uniformity maps, recalibrate center-zone pressure control, "
                "and inspect recent process drift logs."
            ),
            "severity": "Medium",
        },
        "edge_ring": {
            "pattern_reasoning": (
                "Defects forming a ring near the wafer edge often indicate mechanical or alignment-related issues, "
                "especially during polishing or edge exclusion steps."
            ),
            "cause": "Wafer misalignment, edge over-polishing, or uneven edge pressure application.",
            "action": (
                "Verify wafer centering calibration, inspect edge exclusion parameters, "
                "and check polishing pad wear."
            ),
            "severity": "High",
        },
        "local": {
            "pattern_reasoning": (
                "Localized defect clusters suggest isolated contamination or transient process disturbances "
                "affecting a small wafer region."
            ),
            "cause": "Localized particle contamination or brief chamber instability.",
            "action": "Inspect recent chamber cleaning cycles and analyze tool logs for localized anomalies.",
            "severity": "Medium",
        },
        "scratch": {
            "pattern_reasoning": (
                "Linear or directional defect patterns are characteristic of mechanical contact "
                "during wafer handling or transport."
            ),
            "cause": "Physical contact from handling equipment or misaligned wafer transport mechanisms.",
            "action": "Audit wafer handling robots, cassette alignment, and transport paths.",
            "severity": "High",
        },
        "random": {
            "pattern_reasoning": (
                "Scattered, non-uniform defects across the wafer are often associated with airborne particles "
                "or sporadic contamination events."
            ),
            "cause": "Random particle deposition from cleanroom airflow disturbances or filter degradation.",
            "action": "Review cleanroom airflow data and particle counter logs.",
            "severity": "Low",
        },
        "clean": {
            "pattern_reasoning": "No significant defect patterns were detected across the wafer surface.",
            "cause": "Wafer appears within normal process limits.",
            "action": "No corrective action required. Continue routine monitoring.",
            "severity": "None",
        },
    }

    info = defect_knowledge.get(defect_class)
    if info is None:
        return {
            "summary": "The detected pattern does not match known defect categories.",
            "confidence": round(confidence, 3),
            "recommendation": "Manual inspection and expert review are advised.",
        }

    return {
        "defect_class": defect_class,
        "model_confidence": round(confidence, 3),
        "confidence_interpretation": (
            f"The model confidence is {confidence_level}; treat this hypothesis accordingly."
        ),
        "pattern_analysis": info["pattern_reasoning"],
        "probable_root_cause": info["cause"],
        "recommended_action": info["action"],
        "severity_assessment": info["severity"],
        "genai_note": (
            "Fallback rules were used. Connect a GenAI provider for dynamic reasoning."
        ),
    }


def _gemini_response(
    defect_class: str,
    confidence: float,
    context_hint: str,
) -> Optional[Dict[str, Any]]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        import google.generativeai as genai
    except Exception:
        return None

    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    model = genai.GenerativeModel(model_name)

    confidence_level = _confidence_level(confidence)
    prompt = f"""
You are a semiconductor process engineer assistant. Produce a concise, actionable root-cause analysis.

Return ONLY valid JSON with these keys:
summary, confidence_interpretation, pattern_analysis, probable_root_cause,
recommended_action, severity_assessment

Inputs:
- defect_class: {defect_class}
- model_confidence: {confidence:.3f}
- confidence_level: {confidence_level}
- context_hint: {context_hint}

Guidelines:
- Keep each field under 2 sentences.
- Avoid generic phrases; be specific and practical.
- severity_assessment should be one of: None, Low, Medium, High, Critical.
"""

    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 400,
                "response_mime_type": "application/json",
            },
        )
        text = response.text or ""
        data = json.loads(text)
        if not isinstance(data, dict):
            return None

        required = [
            "summary",
            "confidence_interpretation",
            "pattern_analysis",
            "probable_root_cause",
            "recommended_action",
            "severity_assessment",
        ]
        if not all(key in data for key in required):
            return None

        data["defect_class"] = defect_class
        data["model_confidence"] = round(confidence, 3)
        data["genai_note"] = "Generated by Gemini."
        return data
    except Exception:
        return None


def analyze_defect(defect_class: str, confidence: float) -> Dict[str, Any]:
    """
    Uses Gemini if available; otherwise falls back to deterministic rules.
    Set GEMINI_API_KEY and (optionally) GEMINI_MODEL in your environment.
    """
    hint_map = {
        "center": "Defects cluster near wafer center.",
        "edge_ring": "Defects form a ring near the wafer edge.",
        "local": "Defects appear in a localized cluster.",
        "scratch": "Defects appear as linear scratches.",
        "random": "Defects are scattered across the wafer.",
        "clean": "No significant defects detected.",
    }
    context_hint = hint_map.get(defect_class, "Pattern does not match known categories.")

    gemini_payload = _gemini_response(defect_class, confidence, context_hint)
    payload = gemini_payload if gemini_payload is not None else _fallback_rules(defect_class, confidence)

    summary = payload.get("summary") or payload.get("probable_root_cause") or ""
    payload["cause_summary"] = summary[:160] if summary else "Cause analysis unavailable."
    payload["risk_level"] = payload.get("severity_assessment", "Unknown")
    return payload
