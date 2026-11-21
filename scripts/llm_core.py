import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 120


def ask_llm(prompt, model="llama3"):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 200}
        },
        timeout=OLLAMA_TIMEOUT
    )
    return response.json()["response"]


def ask_llm_json(prompt, model="llama3"):
    system = """
You are an MSA alignment controller.

You MUST respond ONLY in valid JSON.

Required format:
{
  "action": "accept | refine | trim | change_tool | change_strategy | stop",
  "tool": "mafft | muscle | esm2",
  "msa_mode": "auto | localpair | globalpair | fast | align | progressive | super5 | local | global",
  "extra_args": "string of flags (empty if none)",
  "reason": "short explanation",
  "confidence": "low | medium | high",
  "metrics_echo": {
    "gap_fraction": <number you saw in prompt>,
    "sp_score": <number you saw in prompt>,
    "identity": <number you saw in prompt>
  }
}

RULES:
- NEVER accept if gap_fraction > 0.15
- If gap_fraction > 0.15 and current tool has been tried 2+ times, use action="change_tool" to switch tools
- You MUST echo the exact metric values you see in the prompt under metrics_echo
- Do NOT estimate or round metric values
"""

    full_prompt = system + "\n\nINPUT:\n" + prompt

    text = ask_llm(full_prompt, model)

    try:
        return json.loads(text)
    except:
        return {
            "error": "invalid_json",
            "raw_output": text
        }
