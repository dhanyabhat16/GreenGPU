"""
AI Explanation module using Google Gemini API for GreenGPU Optimizer.

Generates natural language explanations for:
1. CPU shift decision (compute resource analysis)
2. Removed test cases (redundant test case reduction)
"""

from typing import Any, Optional

try:
    from config import GEMINI_API_KEY, GEMINI_MODEL  # noqa: F401
except ImportError:
    import os
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")


def _get_client():
    """Lazy import and return GenerativeModel."""
    if not GEMINI_API_KEY:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        return genai.GenerativeModel(GEMINI_MODEL)
    except Exception:
        return None


def _extract_text(response) -> str:
    """
    Safely extract text from Gemini response.
    Handles blocked content, multiple parts, and response.text ValueError.
    """
    if not response:
        return ""
    try:
        return (response.text or "").strip()
    except (ValueError, AttributeError):
        pass
    # Fallback: get text from candidates/parts
    try:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return ""
        parts = getattr(candidates[0], "content", None) and getattr(
            candidates[0].content, "parts", None
        ) or []
        texts = []
        for p in parts:
            if getattr(p, "text", None):
                texts.append(p.text)
        return " ".join(texts).strip() if texts else ""
    except Exception:
        return ""


def explain_cpu_shift_decision(summary: dict) -> str:
    """
    Use Gemini to generate a short explanation of why CPU was recommended (or not).

    summary: ComputeDecision.summary_for_ai from compute_analyzer.
    """
    client = _get_client()
    if not client:
        return (
            "AI explanation unavailable due to API quota limits. "
            "Rule-based explanation: " + str(summary.get("reasons", []))
        )

    prompt = f"""You are a green computing assistant. In one short paragraph (2-4 sentences), explain to a developer why the system recommended or did not recommend shifting their ML workload from GPU to CPU. Be clear and technical but concise.

Analysis summary from the system:
{summary}

Do not use bullet points. Write in prose."""

    try:
        response = client.generate_content(prompt)
        text = _extract_text(response)
        if text:
            return text
        # Check if blocked by safety
        if response and getattr(response, "candidates", None):
            c = response.candidates[0]
            if getattr(c, "finish_reason", None) and "blocked" in str(c.finish_reason).lower():
                return (
                    "AI explanation was blocked by content filters. "
                    "Reason from rules: " + str(summary.get("reasons", []))
                )
    except Exception as e:
        return f"AI explanation could not be generated: {e}. Rule-based reason: {summary.get('reasons', [])}"
    return str(summary.get("reasons", []))


def explain_removed_test_cases(summary: dict) -> str:
    """
    Use Gemini to explain why certain test cases were removed as redundant.

    summary: ReductionResult.summary_for_ai from test_case_reducer.
    """
    client = _get_client()
    if not client:
        return (
            "AI explanation unavailable (GEMINI_API_KEY not set). "
            f"Removed {summary.get('removed_count', 0)} test cases due to high similarity (threshold {summary.get('similarity_threshold', 0.97)})."
        )

    prompt = f"""You are a green computing assistant. In one short paragraph (2-4 sentences), explain to a developer why those specific test cases were removed from their validation set by giving examples. Mention semantic/similarity reasons and that they do not add validation benefit. Be concise.

Reduction summary:
{summary}

Do not use bullet points. Write in prose."""

    try:
        response = client.generate_content(prompt)
        text = _extract_text(response)
        if text:
            return text
        if response and getattr(response, "candidates", None):
            c = response.candidates[0]
            if getattr(c, "finish_reason", None) and "blocked" in str(c.finish_reason).lower():
                return (
                    "AI explanation was blocked by content filters. "
                    f"Removed {summary.get('removed_count', 0)} cases (similarity threshold {summary.get('similarity_threshold')})."
                )
    except Exception as e:
        return (
            f"AI explanation could not be generated: {e}. "
            f"Removed {summary.get('removed_count', 0)} cases (similarity threshold {summary.get('similarity_threshold')})."
        )
    return str(summary)


def explain_impact_summary(data: dict) -> str:
    """
    Use Gemini to generate a sustainability impact summary from profile + dedup + eval data.
    """
    client = _get_client()
    if not client:
        return (
            "AI summary unavailable (GEMINI_API_KEY not set). "
            "Run profiling and deduplication to see your impact metrics above."
        )

    prompt = f"""You are a green computing assistant. In one short paragraph (3-5 sentences), summarize the sustainability impact of this ML workload optimization. Be specific about energy saved, CO2 avoided, and cost. Mention both compute (GPU/CPU) and test case deduplication impacts if relevant. Use the data provided. Be concise and impactful.

Data:
{data}

Do not use bullet points. Write in prose."""

    try:
        response = client.generate_content(prompt)
        text = _extract_text(response)
        if text:
            return text
        if response and getattr(response, "candidates", None):
            c = response.candidates[0]
            if getattr(c, "finish_reason", None) and "blocked" in str(c.finish_reason).lower():
                return "AI summary was blocked by content filters. Run profiling and deduplication to see your impact metrics."
    except Exception as e:
        return f"AI summary could not be generated: {e}. Run profiling and deduplication to see your impact metrics."
    return str(data)
