import os

from dotenv import load_dotenv

load_dotenv()


SYSTEM_SKILL = os.getenv("SYSTEM_SKILL", "system")

TOKEN_THRESHOLD_PREPARE = int(os.getenv("TOKEN_THRESHOLD_PREPARE", 50_000))
TOKEN_THRESHOLD_OBSERVE = int(os.getenv("TOKEN_THRESHOLD_PREPARE", 25_000))
TOKEN_THRESHOLD_REFLECT = int(os.getenv("TOKEN_THRESHOLD_PREPARE", 10_000))

SUPERVISOR_MODEL = os.getenv("SUPERVISOR_MODEL", "gpt-5-mini")
LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL", "https://openrouter.ai/api/v1/chat/completions"
)
