import os
import sys
import pandas as pd
from agent_a.parser import parse_question
from agent_a.tasks.task1_simple import run as run_task1
from agent_a.tasks.task2_conditional import run as run_task2
from agent_a.tasks.task3_signal import run as run_task3
from Agent.agent_a.tasks.task4_parser import run as run_task4  # LLM fallback (비공개 자유문답)
from agent_a.fetcher import get_history
from agent_a.stock_api import get_listed_tickers as get_all_tickers

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

AGENT_NAME = "배당이당_agent"

LOG_PATH = os.path.join(ROOT, "agent_a", "log", "cli_interaction.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

def log_interaction(question: str, answer: str):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"Q: {question}\nA: {answer}\n{'-'*40}\n")

def dispatch(q: str) -> str:
    params = parse_question(q)
    task_type = params.get('type')

    # 1. 명확하지 않은 질의(clarify)
    if task_type in ('ambiguous', 'blank', 'date_only'):
        answer = params.get("message", "조금 더 구체적으로 질문해주시면 좋겠습니다.")
        log_interaction(q, answer)
        return answer

    # 2. 단일 메트릭/기초 조회
    if task_type in ('single_metric', 'single_metric_flex', 'market_index', 'market_count', 'total_value', 'top_n', 'updown_count'):
        answer = run_task1(q, get_history, get_all_tickers)
        log_interaction(q, answer)
        return answer

    # 3. 조건검색(조건 필터)
    if task_type in (
        'volume_change_pct', 'combined_pct_volume', 'volume_threshold',
        'pct_change', 'price_range', 'top_n', 'count'
    ):
        answer = run_task2(q, get_history, get_all_tickers)
        log_interaction(q, answer)
        return answer

    # 4. 시그널/신호(Task 3)
    if task_type in (
        'rsi_condition', 'ma_condition', 'bollinger_touch',
        'cross_count', 'cross_stock', 'vol_ma_spike'
    ):
        answer = run_task3(q, get_history, get_all_tickers)
        log_interaction(q, answer)
        return answer

    # 5. LLM fallback (비공개, 오픈/누락 catch-all)
    if task_type in ("llm_fallback", "unknown"):
        answer = run_task4(q, get_history, get_all_tickers)
        log_interaction(q, answer)
        return answer

    # 예외 케이스
    answer = "죄송해요, 이해하지 못한 질문입니다."
    log_interaction(q, answer)
    return answer

def main():
    print(f"🌱 {AGENT_NAME}에 오신 것을 환영합니다!")
    print("자유롭게 질문하세요! ('exit' 입력 시 종료)\n")
    try:
        while True:
            q = input("> ").strip()
            if not q or q.lower() in ('exit', 'quit'):
                print("종료합니다. Have a 배당!")
                break
            answer = dispatch(q)
            print(answer)
    except (KeyboardInterrupt, EOFError):
        print("\n종료합니다.")
        sys.exit()

if __name__ == "__main__":
    main()