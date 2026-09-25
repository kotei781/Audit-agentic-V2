import os
import json
import logging
import traceback
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from openai import OpenAI
import requests

# --- Configuration & Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Supervisor")

# --- System Prompt for Red-Teaming ---
SUPERVISOR_SYSTEM_PROMPT = """You are an Independent AI Supervisor and Red-Teamer.
Your sole purpose is to critically audit the output of an Audit AI.
You are skeptical, rigorous, and obsessed with accuracy.

TASK:
Compare the 'raw_content' (the source of truth) with the 'audit_json' (the report produced by the Audit AI).
Identify any of the following errors:
1. HALLUCINATION: The Audit AI reported data that does not exist in the raw_content.
2. CALCULATION_ERROR: Mathematical errors in the audit report.
3. MISMATCH_DATA: The Audit AI correctly found the data but reported the wrong value.
4. LOGIC_ERROR: The Audit AI's conclusion contradicts the evidence in the raw_content.

OUTPUT FORMAT:
You MUST respond with a valid JSON object following this schema:
{
  "status": "PASSED" | "REJECTED",
  "confidence_score": float, // 0.0 to 1.0
  "discrepancies": [
    {
      "type": "HALLUCINATION" | "CALCULATION_ERROR" | "MISMATCH_DATA" | "LOGIC_ERROR",
      "field": "the field name that is incorrect",
      "audit_reported": "the value reported by the Audit AI",
      "actual_raw_data": "the actual value found in the source data",
      "explanation": "detailed explanation of why this is an error"
    }
  ],
  "supervisor_comment": "overall summary of the evaluation",
  "evaluated_by_tier": "The tier/model name that performed this evaluation"
}

If no errors are found, 'status' should be 'PASSED' and 'discrepancies' should be an empty list [].
"""

class FlexibleSupervisor:
    def __init__(self):
        # Load keys and models from environment
        self.keys = {
            "GEMINI_1": os.getenv("GEMINI_API_KEY_1") or os.getenv("GEMINI_API_KEY"),
            "GEMINI_2": os.getenv("GEMINI_API_KEY_2"),
            "GROQ_DEEPSEEK": os.getenv("GROQ_API_KEY") or os.getenv("DEEPSEEK_API_KEY"),
            "OPENAI": os.getenv("OPENAI_API_KEY"),
        }

        # Define the Fallback Chain (Multi-Tier Matrix)
        # Model names are loaded from env to support dynamic config via UI
        self.fallback_chain = [
            {
                "tier": "Tier 1 - Gemini Primary",
                "provider": "gemini",
                "model": os.getenv("GEMINI_MODEL_TIER1", "gemini-1.5-pro"),
                "key_id": "GEMINI_1"
            },
            {
                "tier": "Tier 2 - Gemini Secondary",
                "provider": "gemini",
                "model": os.getenv("GEMINI_MODEL_TIER2", "gemini-1.5-flash"),
                "key_id": "GEMINI_2"
            },
            {
                "tier": "Tier 3 - Groq/DeepSeek",
                "provider": "openai_compat",
                "model": os.getenv("GROQ_MODEL", "llama-3.3-70b"),
                "key_id": "GROQ_DEEPSEEK"
            },
            {
                "tier": "Tier 4 - OpenAI",
                "provider": "openai",
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "key_id": "OPENAI"
            },
            {
                "tier": "Tier 5 - Ollama Local",
                "provider": "ollama",
                "model": os.getenv("OLLAMA_MODEL", "llama3"),
                "key_id": None
            },
        ]

    def _call_gemini(self, model_name: str, api_key: str, prompt: str) -> str:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text

    def _call_openai_compat(self, model_name: str, api_key: str, prompt: str) -> str:
        # Handles Groq, DeepSeek, or OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1" if (api_key and "GROQ" in api_key) else None)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

    def _call_ollama(self, model_name: str, prompt: str) -> str:
        # Local fallback
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        url = f"{base_url}/api/generate"
        payload = {
            "model": model_name,
            "prompt": f"{SUPERVISOR_SYSTEM_PROMPT}\n\n{prompt}",
            "stream": False,
            "format": "json"
        }
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json().get("response", "")

    def supervise(self, raw_content: str, audit_json: Dict[str, Any]) -> Dict[str, Any]:
        user_prompt = f"RAW CONTENT:\n{raw_content}\n\nAUDIT JSON:\n{json.dumps(audit_json, indent=2)}"

        for config_item in self.fallback_chain:
            tier_name = config_item["tier"]
            provider = config_item["provider"]
            model = config_item["model"]
            key_id = config_item["key_id"]
            api_key = self.keys.get(key_id) if key_id else None

            try:
                logger.info(f"[Supervisor] Attempting evaluation with {tier_name}...")

                if provider == "gemini":
                    if not api_key: raise ValueError("Missing API Key")
                    result_text = self._call_gemini(model, api_key, f"{SUPERVISOR_SYSTEM_PROMPT}\n\n{user_prompt}")

                elif provider == "openai_compat":
                    if not api_key: raise ValueError("Missing API Key")
                    result_text = self._call_openai_compat(model, api_key, user_prompt)

                elif provider == "openai":
                    if not api_key: raise ValueError("Missing API Key")
                    client = OpenAI(api_key=api_key)
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
                        response_format={"type": "json_object"}
                    )
                    result_text = response.choices[0].message.content

                elif provider == "ollama":
                    result_text = self._call_ollama(model, user_prompt)

                else:
                    raise ValueError(f"Unknown provider {provider}")

                # Parse JSON result
                # Try to clean markdown blocks if present
                clean_json = result_text.strip()
                if clean_json.startswith("```json"):
                    clean_json = clean_json.split("```json")[1].split("```")[0].strip()
                elif clean_json.startswith("```"):
                    clean_json = clean_json.split("```")[1].split("```")[0].strip()

                final_result = json.loads(clean_json)
                final_result["evaluated_by_tier"] = tier_name
                logger.info(f"[Supervisor] Success! Evaluation completed by {tier_name}.")
                return final_result

            except Exception as e:
                # SILENT AUTO-FAILOVER
                logger.warning(f"[Supervisor] {tier_name} thất bại (Lỗi: {type(e).__name__}: {str(e)}), lập tức chuyển sang cấp tiếp theo...")
                continue

        # All tiers failed
        return {
            "status": "ERROR",
            "confidence_score": 0.0,
            "discrepancies": [],
            "supervisor_comment": "Tất cả các cấp dự phòng (Fallback Chain) đều thất bại. Vui lòng kiểm tra lại API Key hoặc kết nối mạng/Ollama.",
            "evaluated_by_tier": "NONE"
        }

# --- Standalone Main Block for Testing ---
if __name__ == "__main__":
    # Giả lập môi trường: Thiết lập một Key sai cho Tier 1 để test failover
    os.environ["GEMINI_API_KEY_1"] = "INVALID_KEY_FOR_TESTING"
    # Bạn có thể thiết lập các key thật ở đây hoặc trong .env để test đến Tier 4/5
    # os.environ["GEMINI_API_KEY_2"] = "YOUR_ACTUAL_KEY"

    supervisor = FlexibleSupervisor()

    # Test Case: AI Kiểm toán báo cáo sai giá trị (Hallucination/Mismatch)
    sample_raw_content = "Company: TechCorp, Revenue: 1,000,000 USD, Employees: 50, Location: New York"
    sample_audit_json = {
        "company_name": "TechCorp",
        "revenue": "5,000,000 USD", # Sai giá trị -> Mismatch/Hallucination
        "employees": 50,
        "conclusion": "Company is doing well"
    }

    print("\n--- BẮT ĐẦU KIỂM THỬ SUPERVISOR AGENT ---")
    print("Kịch bản: Tier 1 dùng Key sai -> Hệ thống phải tự động nhảy sang các Tier tiếp theo mà không văng lỗi đỏ.")

    result = supervisor.supervise(sample_raw_content, sample_audit_json)

    print("\n--- KẾT QUẢ CUỐI CÙNG ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\n--- KẾT THÚC KIỂM THỬ ---")
