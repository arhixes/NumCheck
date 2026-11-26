import os
import time
import uuid

import httpx
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# GigaChat
AUTH_KEY = os.environ["GIGACHAT_AUTH_KEY"] 
SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
MODEL = os.getenv("LLM_MODEL", "GigaChat-2-Max")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.giga.chat/v1")
OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
CA_BUNDLE = os.getenv("GIGACHAT_CA_BUNDLE")
VERIFY = CA_BUNDLE if CA_BUNDLE else False
http_client = httpx.Client(verify=VERIFY, timeout=60)
_token = {"value": None, "expires_at": 0.0}

SYSTEM_PROMPT = (
    "Ты финансовый аналитик. Проанализируй переданный текст с финансовыми данными. "
    "Для каждого ключевого числа или расчёта пересчитай значение самостоятельно "
    "и сравни с заявленным. Отвечай только JSON по заданной схеме."
)

app = FastAPI(title="Financial Analyst with Number Verification")
def get_token() -> str:
    """Access token GigaChat живёт ~30 минут, поэтому обновляем его по необходимости."""
    if _token["value"] and time.time() < _token["expires_at"] - 60:
        return _token["value"]
    r = http_client.post(
        OAUTH_URL,
        headers={
            "Authorization": f"Basic {AUTH_KEY}",
            "RqUID": str(uuid.uuid4()),
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        data={"scope": SCOPE},
    )
    r.raise_for_status()
    data = r.json()
    _token["value"] = data["access_token"]
    _token["expires_at"] = data["expires_at"] / 1000
    return _token["value"]

class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Финансовый текст или данные для анализа")
    temperature: float = Field(0.2, ge=0.0, le=2.0, description="Температура генерации")


class NumberCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str
    stated_value: float
    recalculated_value: float
    is_correct: bool
    comment: str


class AnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str
    checks: list[NumberCheck]
    all_numbers_correct: bool


def inline_refs(schema: dict) -> dict:
    """Разворачивает $defs/$ref, чтобы схема была плоской."""
    defs = schema.pop("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(defs[node["$ref"].split("/")[-1]])
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(i) for i in node]
        return node

    return walk(schema)


def strip_keys(node, keys):
    if isinstance(node, dict):
        return {k: strip_keys(v, keys) for k, v in node.items() if k not in keys}
    if isinstance(node, list):
        return [strip_keys(i, keys) for i in node]
    return node


SCHEMA = strip_keys(inline_refs(AnalysisResult.model_json_schema()), {"title", "additionalProperties"})


# Эндпоинт
@app.post("/analyze", response_model=AnalysisResult)
def analyze(req: AnalyzeRequest) -> AnalysisResult:
    try:
        client = OpenAI(
            base_url=BASE_URL,
            api_key=get_token(),
            http_client=http_client,
            timeout=45,
            max_retries=0,
        )
        response = client.chat.completions.create(
            model=MODEL,
            temperature=req.temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": req.text},
            ],
            response_format={"type": "json_schema", "schema": SCHEMA, "strict": True},
        )
        return AnalysisResult.model_validate_json(response.choices[0].message.content)
    except ValidationError as e:
        raise HTTPException(status_code=502, detail=f"Модель вернула невалидный JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ошибка обращения к LLM: {e}")
