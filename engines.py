# engines.py

import os
from backend.config import DEMO_MODE, DEMO_WATERMARK

def get_llm_response(prompt, engine="dummy"):
    if engine == "openai":
        return _openai_response(prompt)
    elif engine == "mistral":
        return _mistral_response(prompt)
    elif engine == "dummy":
        return _dummy_response(prompt)
    else:
        return "LLM engine not configured. Please check config.py"

def _openai_response(prompt):
    try:
        import openai
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return _dummy_response(prompt)
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert business advisor with deep CXO experience."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=0.6,
        )
        result = response['choices'][0]['message']['content'].strip()
        if DEMO_MODE:
            result += DEMO_WATERMARK
        return result
    except Exception as e:
        return f"Error from OpenAI: {e}\n\n{_dummy_response(prompt)}"

def _mistral_response(prompt):
    # Placeholder for Mistral API/local. For now, fallback to dummy.
    return "[Mistral LLM integration not yet implemented. Showing demo output:]\n" + _dummy_response(prompt)

def _dummy_response(prompt):
    cxos = {
        "CFO": "CFO Summary: Key cost centers identified. Monitor cash flow, reduce discretionary spending, and seek new revenue streams.",
        "COO": "COO Insights: Address operational bottlenecks, streamline logistics, and automate routine processes.",
        "CMO": "CMO Recommendations: Focus on digital outreach, leverage partnerships, and track campaign ROI closely.",
        "CHRO": "CHRO Actions: Prioritize employee engagement, develop retention strategies, and ensure compliance with new regulations."
    }
    for k, v in cxos.items():
        if k in prompt:
            return v + DEMO_WATERMARK
    return "\n\n".join([v for v in cxos.values()]) + DEMO_WATERMARK
