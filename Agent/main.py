import os
import traceback
import asyncio
import time
from fastapi import FastAPI, Query, Header, HTTPException, Response
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from chromadb import PersistentClient

# 반드시 PYTHONPATH가 맞아야 아래 import가 정상작동
from Agent.agent_b.model import build_collection as build_b, get_answer as answer_b
from Agent.agent_a.model import get_answer as answer_a
from Agent.agent_b.config import DB_DIR, COLLECTION_NAME

app = FastAPI(
    title="배당이당 Agent",
    version="0.1.0",
    description="한국 상장사 배당정보 + 배당 이후 수익률 + 상세공시 요약 제공",
)

# === CORS (API 사용/테스트 편의) ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 환경에선 전체 허용, 운영에선 도메인 제한 권장
    allow_methods=["*"],
    allow_headers=["*"],
)

# === in-memory 메트릭 ===
_metrics = {
    "total_requests": 0,
    "failures": 0,
    "cumulative_latency_sec": 0.0,
}

class AnswerResponse(BaseModel):
    answer: str

# 배당 관련 키워드 (둘 중 한글/영문 등 확장해도 됨)
DIVIDEND_KEYWORDS = [
    "배당", "지급일", "총배당금", "현금배당", "배당금" , "기준일", "배당정보", "배당금이", "배당금은"
]
DIVIDEND_HINTS_IN_ANSWER = ["배당", "지급일", "총배당금", "예측", "기대치"]

def flatten_answer(raw) -> str:
    if isinstance(raw, dict):
        a = raw.get("answer", "")
        if isinstance(a, dict):
            a = a.get("answer", a)
        if isinstance(a, list):
            paragraphs = [str(item).strip() for item in a if item is not None and str(item).strip()]
            return "\n\n".join(paragraphs) if paragraphs else ""
        if isinstance(a, str):
            return a.strip()
        return str(a)
    if isinstance(raw, list):
        return "\n\n".join(str(item).strip() for item in raw if item is not None)
    return str(raw)

def sanitize_answer_text(text: str) -> str:
    if not text or text.strip() == "":
        return "질문은 이해했지만, 현재 사용 가능한 데이터가 부족하여 정확한 답변을 제공하기 어렵습니다."
    lowered = text.lower()
    if "nan" in lowered or "none" in lowered or "정보 없음" in text:
        return "질문은 이해했지만, 현재 사용 가능한 데이터가 부족하여 정확한 답변을 제공하기 어렵습니다."
    if "0원" in text:
        import re
        has_other_info = bool(re.search(r"\d+%|지급일|기준일|예측|수익률", text))
        if not has_other_info:
            return "질문은 이해했지만, 현재 사용 가능한 데이터가 부족하여 정확한 답변을 제공하기 어렵습니다."
    return text

@app.on_event("startup")
async def startup_event():
    print("▶ startup_event 시작")
    async def build_bg():
        try:
            await asyncio.to_thread(build_b)  # agent_b collection 백그라운드 구축
            print("✅ ChromaDB collection built successfully (background).")
        except Exception as e:
            print("⚠️ build_collection 실패(백그라운드):", e)
            traceback.print_exc()
    asyncio.create_task(build_bg())
    print("▶ startup_event 종료")

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

@app.get(
    "/agent",
    summary="Query Agent",
    description="예: 삼성전자 배당 정보 알려줘 / 이번년도 배당금이 가장 많았던 기업은?",
    response_model=AnswerResponse,
)
async def query_agent(
    response: Response,
    question: str = Query(..., description="사용자 질문"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    request_id: Optional[str] = Header(None, alias="X-NCP-CLOVASTUDIO-REQUEST-ID"),
):
    start = time.perf_counter()
    _metrics["total_requests"] += 1

    # 헤더 누락 경고 로그
    if authorization is None:
        print("[WARN] Authorization header missing.")
    if request_id is None:
        print("[WARN] X-NCP-CLOVASTUDIO-REQUEST-ID header missing.")

    lowered_q = question.lower()
    has_div_keyword = any(kw in lowered_q for kw in DIVIDEND_KEYWORDS)

    try:
        print(f"[DEBUG] incoming question: {question}")

        # 1) 배당 키워드가 있으면 agent_b → (실패시) agent_a fallback
        if has_div_keyword:
            try:
                raw_resp = answer_b(question)
                print(f"[DEBUG] [agent_b] raw response: {raw_resp}")
                flattened = flatten_answer(raw_resp)
                final_answer = sanitize_answer_text(flattened)
                if final_answer and "질문은 이해했지만" not in final_answer:
                    # 배당에 명확한 답이 나오면 바로 반환
                    elapsed = time.perf_counter() - start
                    _metrics["cumulative_latency_sec"] += elapsed
                    response.headers["X-Processing-Time"] = f"{elapsed:.3f}s"
                    return {"answer": final_answer}
            except Exception as e:
                print(f"[WARN] agent_b 실패, agent_a로 fallback: {e}")
                traceback.print_exc()
            # agent_b가 에러거나 무응답이면 agent_a로 fallback
            try:
                raw_resp = answer_a(question)
                flattened = flatten_answer(raw_resp)
                final_answer = sanitize_answer_text(flattened)
                elapsed = time.perf_counter() - start
                _metrics["cumulative_latency_sec"] += elapsed
                response.headers["X-Processing-Time"] = f"{elapsed:.3f}s"
                return {"answer": final_answer}
            except Exception as e:
                print(f"[ERROR] agent_a fallback도 실패: {e}")
                traceback.print_exc()
                raise HTTPException(status_code=500, detail="배당 공시 검색 중 오류가 발생했습니다.")

        # 2) 배당 관련 키워드가 없으면 무조건 agent_a로만!
        else:
            try:
                raw_resp = answer_a(question)
                flattened = flatten_answer(raw_resp)
                final_answer = sanitize_answer_text(flattened)
                elapsed = time.perf_counter() - start
                _metrics["cumulative_latency_sec"] += elapsed
                response.headers["X-Processing-Time"] = f"{elapsed:.3f}s"
                return {"answer": final_answer}
            except Exception as e:
                print(f"[ERROR] agent_a(non-dividend) 실패: {e}")
                traceback.print_exc()
                raise HTTPException(status_code=500, detail="요청 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")

    except Exception as e:
        _metrics["failures"] += 1
        print(f"[ERROR] exception in query_agent: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="에이전트 처리 중 내부 오류가 발생했습니다.")

# ---- Health, metrics, debug API ----
@app.get("/debug/db_dir", summary="Debug: DB 경로 확인")
def debug_db_dir():
    return {
        "config_db_dir": DB_DIR,
        "collection_name": COLLECTION_NAME,
    }

@app.get("/debug/collections", summary="Debug: 컬렉션 목록")
def debug_collections():
    client = PersistentClient(path=DB_DIR)
    try:
        cols = [c.name for c in client.list_collections()]
    except Exception as e:
        return {"error": f"컬렉션 불러오기 실패: {e}"}
    return {"collections": cols}

@app.get("/health", summary="Health check")
async def health():
    return {"status": "ok"}

@app.get("/metrics", summary="Simple in-memory metrics")
def metrics():
    total = _metrics["total_requests"]
    failures = _metrics["failures"]
    avg_latency = (_metrics["cumulative_latency_sec"] / total) if total > 0 else 0.0
    return {
        "total_requests": total,
        "failures": failures,
        "average_latency_sec": round(avg_latency, 4),
    }