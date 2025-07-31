Dividend Reaction Pipeline & Agents

⸻

📖 개요

본 프로젝트는 한국 기업의 배당 공시 정보를 자동으로 수집하고, 이를 기반으로 머신러닝 분석과 자연어 질의응답 서비스를 제공하는 완전 자동화 파이프라인입니다.

파이프라인은 두 개의 독립적 환경인 agent_a와 agent_b로 구성되며, 각각 별도의 가상환경에서 운영되어 패키지 충돌을 방지합니다.

⸻

🌟 주요 기능

✅ 자동화 파이프라인
	•	매일 19시 스케줄링 자동 실행으로 아래 프로세스 순차 수행
	1.	📡 DART 공시 증분 수집 및 CSV↔JSONL 동기화 (–skip-sync 옵션 지원)
	2.	📐 머신러닝용 데이터 전처리 (결측치/중복 제거, 0원 배당 필터링)
	3.	💰 주가 데이터 수집 및 병합 (FinanceDataReader, yfinance, pykrx 활용)
	4.	📊 머신러닝 모듈용 CSV 생성 (classification, regression, clustering)
	5.	⚙️ 피처 엔지니어링 수행
	6.	🧠 모델 학습 및 예측
	•	분류: 배당 후 1일 상승 확률 (p_up)
	•	회귀: 예측 수익률 (y_pred), 잔차 (residual)
	7.	🗃 마스터 CSV 통합 (all_stocks_master.csv에 결과 컬럼 합산)
	8.	📚 문서 임베딩 (Sentence-Transformers + ChromaDB 활용)

⸻

📂 디렉터리 구조
Agent/
├── agent_a/
│   ├── data/
│   │   ├── agent_db.sqlite           # Agent A의 SQLite DB, 공시 및 주가 데이터를 저장
│   │   ├── mappings/
│   │   │   └── ticker_name_market.csv # 티커-종목명-시장 매핑 파일
│   │   └── queries/                  # 사전 정의된 사용자 질의 예시 및 룰 저장소
│   ├── log/
│   │   └── cli_interaction.log      # CLI 인터랙션 로그 기록
│   ├── parser.py                    # 자연어 파서 모듈 (Task4)
│   ├── model.py                     # 머신러닝 모델 관련 코드
│   ├── tasks/
│   │   ├── task.py                  # Task1~3 통합 작업 스크립트
│   │   ├── task1_simple.py          # 단순 조건 필터링
│   │   ├── task2_conditional.py     # 복합 조건 처리
│   │   ├── task3_signal.py          # 시그널 탐지 및 분석
│   │   └── task4_llm.py             # LLM 기반 질의응답 처리
│   ├── fetcher.py                  # 공시 및 데이터 수집기
│   ├── stock_api.py                # 주식 데이터 API 래퍼
│   └── cache_manager.py            # 데이터 캐싱 및 관리 도구
│
├── agent_b/
│   ├── data/
│   │   └── chroma_db/
│   │       └── chroma.sqlite3       # Agent B의 ChromaDB 임베딩 DB
│   └── main.py                     # FastAPI 기반 API 서버 메인 파일
│
run_pipeline.py                    # 전체 파이프라인 일괄 실행 스크립트
crawl_dividend.py                 # DART 공시 크롤러
scheduler.py                     # 스케줄러 (자동화 실행용)
util/
├── dart_api.py                   # DART Open API 인터페이스
├── data_cleaning.py              # 데이터 전처리 함수 모음
├── feature_engineering.py        # 피처 생성 함수
├── price_fetcher.py              # 주가 데이터 수집
└── embed_utils.py                # 문서 임베딩 관련 유틸리티
data/                            # 원본 및 가공 데이터 저장소
models/                          # 학습된 ML 모델 저장소
embeddings_db/                   # 임베딩 벡터 DB 저장소
artifacts/                       # 기타 결과물 및 로그 저장소

⚠️ 환경 설정 및 패키지 관리

1. venv_agent (Agent A)
•	주요 패키지: beautifulsoup4, fastapi, numpy, pandas, peewee, yfinance, finance-datareader 등
•	상세 목록:
	annotated-types==0.5.0
	anyio==3.7.1
	beautifulsoup4==4.13.4
	certifi==2025.6.15
	charset-normalizer==3.4.2
	fastapi==0.103.2
	numpy>=1.21.6
	pandas>=1.3.5
	peewee==3.18.1
	uvicorn==0.22.0
	yfinance==0.2.57
	pyarrow==14.0.1
	fastparquet==2024.2.0
	finance-datareader==0.9.90
	python-Levenshtein==0.20.2
	rapidfuzz
	ta

2. venv_pipeline (Agent B)
•	주요 패키지: fastapi, lightgbm, chromadb, sentence-transformers, scikit-learn, yfinance, pykrx 등
•	상세 목록:
	python-dotenv>=1.0.0
	numpy>=1.23
	pandas>=2.0
	pyarrow>=14.0
	fastparquet>=2024.2.0
	apscheduler>=3.10.1
	chardet>=5.2.0
	requests>=2.31
	urllib3>=2.0
	beautifulsoup4>=4.12
	lxml>=4.9
	tqdm>=4.66
	joblib>=1.3
	finance-datareader>=0.9.90
	yfinance>=0.2.40
	pykrx>=1.0.45
	scikit-learn>=1.3
	lightgbm>=4.3
	sentence-transformers>=2.6
	chromadb>=0.4
	faiss-cpu>=1.7
	papermill>=2.5
	fastapi>=0.109.0
	uvicorn>=0.25.0
	openai>=1.13.3
	python-multipart>=0.0.9
	typing_extensions>=4.8.0

🐳 Docker 멀티 스테이지 빌드 
FROM python:3.9-slim AS builder
WORKDIR /app/Agent/agent_b
COPY requirements_a.txt .
RUN pip install --no-cache-dir -r requirements_a.txt
COPY requirements_b.txt .
RUN pip install --no-cache-dir --no-deps -r requirements_b.txt

FROM python:3.9-slim
WORKDIR /app/Agent/agent_b
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY Agent/agent_b .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


💾 데이터베이스 경로
•	Agent A DB: Agent/agent_a/data/agent_db.sqlite
•	Agent B DB (ChromaDB): Agent/agent_b/data/chroma_db/chroma.sqlite3

🛠 실행 방법
git clone https://github.com/king-gun1116/dividend-reaction-agent.git
cd dividend-reaction-agent
pip install -r requirements.txt

# 환경 변수 설정 (.env)
DART_API_KEY=YOUR_OPENDART_KEY

# 파이프라인 실행
python run_pipeline.py --start 20130101 --end 20250723

# FastAPI 서버 실행 (Agent B)
uvicorn main:app --host 0.0.0.0 --port 8000

🎯 타겟 유저
•	일반 투자자, 증권 애널리스트, 금융 데이터 분석가, 퀀트 개발자 등
•	배당에 관심 있는 모든 금융 사용자
•	머신러닝 기반 배당 수익률 예측과 자동화된 공시 정보 서비스가 필요한 전문가 및 개인


📈 배당을 중심으로 설계한 이유
•	배당은 투자자에게 확실한 현금 흐름 이벤트로, 투자 판단의 핵심 요소입니다.
•	특히 12월 결산 배당일 전후에는 주가가 상승하는 경향이 있어, 배당 공시 정보가 매우 중요합니다.
•	본 시스템은 최근 결산일 정보까지 실시간 확인 가능하며,
•	사용자가 관심 있는 기업의 배당 클러스터 집단을 머신러닝으로 분류하여,
•	배당 이후 1~30일 단위의 수익률 예측 및 최적 매도 시점 안내까지 제공합니다.
•	배당 데이터는 2013년 1월 1일부터 누적 수집하며, 완전 자동화된 파이프라인으로 계속 업데이트됩니다.


🔧 개발 및 기술 히스토리

Agent A (초기 데이터 수집 및 처리)
• 초기에는 티커 단위로 데이터 연동을 시도했으나 속도 및 확장성 한계가 있었습니다.
• 이후 SQLite DB 기반 구조로 개선하여 속도 및 안정성을 향상시켰습니다.
• 주가 정보는 Yahoo Finance뿐 아니라, KRX 데이터를 보완하여 완성도를 높였습니다.
• 파싱 작업을 기존 task1~task3으로 분산된 스크립트에서 task.py 하나로 통합하여 관리 효율을 개선했습니다.

Agent B (배당 ML 모델 및 서비스)
• DART에서 배당 원문 보고서와 Investing.com 배당 데이터를 수집해 DB를 구축했습니다.
• 머신러닝 모델을 돌려 배당 공시 이후 주가가 상승/하락하는 확률을 예측합니다.
• 예측 정확도에 따른 클러스터 분류를 4단계로 구분:
	0: 뛰어난 성과 - 예측을 크게 상회  
	1: 부진한 성과 - 예측을 상당히 하회  
	2: 안정적 흐름 - 예측치와 실제가 유사  
	3: 소폭 부진 - 예측보다 약간 낮음  
•	FastAPI 기반 API 서버로 배당 관련 자연어 질의응답 서비스를 제공합니다.

📝 제출 관련 안내
	•	주제 범위: 금융 전반, 주식 외 타 분야 자유롭게 구성 가능
	•	타겟 유저: 일반 투자자, 애널리스트 등 누구나 이용 가능 (README에 명시 필수)
	•	제출 방식: REST API 기반 평가 (Web UI는 선택사항)
	•	README 포함 내용: 기능 설명, 기술 스택, 주요 흐름, 테스트 예시 등
	•	엔드포인트 구조: 계층형 URL 권장 (/api/v1/...)

⸻

이상으로 배당 중심의 머신러닝 자동화 파이프라인 및 Agent 시스템의 전반적 구조와 운영 방식을 상세히 기술하였습니다.
본 README는 금융 데이터 전문가 및 개발자가 빠르게 이해하고 활용할 수 있도록 설계되었습니다.


<질문 리스트>


- Task5
삼성전자 배당 정보 알려줘
종목코드 005930 배당 정보
최근 배당 수익률 알려줘
2024년 배당 수익률 상위 5개 기업은?
{기업명} 배당수익률 알려줘"
"{기업명} 배당공시일 알려줘" 
"{기업명} 배정기준일 알려줘"
"{연도}년 배당금이 가장 많았던 기업은?"
삼성전자 배당금이랑 지급일 알려줘”
이번 년도 가장 높은 배당수익률 가진 회사


<로컬 서버 > : http://127.0.0.1:8000/docs#

<최종 서버> : http://49.50.133.5:8000/docs
