# agent_a/model.py

import os
import sys
from typing import Optional
from fastapi import HTTPException

# 프로젝트 루트 및 모듈 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

# 핵심 모듈 import
from agent_a.parser import parse_question
from agent_a.tasks.task import run as run_task          # 통합 핸들러
from agent_a.tasks.task4_llm import llm_fallback        # LLM fallback 핸들러
from agent_a.fetcher import get_history
from agent_a.stock_api import get_listed_tickers

# 로그 설정
LOG_PATH = os.path.join(ROOT_DIR, "agent_a", "log", "cli_interaction.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
def log_interaction(question: str, answer: str):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"Q: {question}\nA: {answer}\n{'-'*40}\n")

# ─── 도메인 체크 & 역질문 헬퍼 ───────────────────────────────────────────
def is_dividend_question(text: str) -> bool:
    """‘배당’ 키워드가 포함되어야 배당 에이전트로 처리"""
    for kw in ("배당", "배당금", "배당일", "배당수익률"):
        if kw in text:
            return True
    return False

def extract_subject(text: str) -> str:
    """질문에서 첫 번째 명사(토큰)만 간단하게 추출"""
    tokens = text.replace("?", "").split()
    return tokens[0] if tokens else text

# ──────────────────────────────────────────────────────────────────────

def get_answer(question: str) -> str:
    """
    agent_a 메인 응답 함수
     1) 배당 도메인 여부 체크 → 비배당이면 역질문
     2) 파싱된 Task 실행
     3) LLM fallback
    """
    params    = parse_question(question)
    task_type = params.get("type")

    # 1) 핸들러 처리 가능한 타입이면 무조건 run_task로 실행!
    if task_type not in ("llm_fallback", "unknown"):
        # 종목명/날짜 등 필수 파라미터 없을 때만 진짜 clarifying 문구 반환
        if task_type == "single_metric":
            if not params.get("ticker"):
                # 진짜로 종목명이 필요할 때만!
                return "종목명이 모호합니다. 어떤 회사의 정보를 원하시나요?"
            if not params.get("date"):
                return "언제(날짜) 기준으로 보시길 원하시나요? 예: 2025-07-24"
        if task_type in ("recent_rally", "recent_crash"):
            if not params.get("window"):
                return "‘최근’이라고 하셨는데, 구체적으로 몇 일 기준을 원하시나요? 예: 최근 7일, 1개월"
        # 나머지는 무조건 run_task로 처리 (리스트, dict 반환)
        answer = run_task(question)
        log_interaction(question, str(answer))
        return answer

    # 2) 파싱/핸들러 모두 실패(unknown)이면 catch-all 문구
    answer = llm_fallback(question, get_history, get_listed_tickers)
    log_interaction(question, str(answer))
    return answer