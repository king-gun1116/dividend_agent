# ─── 빌드 스테이지 ────────────────────────────────────────
FROM python:3.9-slim AS builder
WORKDIR /app/Agent/agent_b

# A 전용 deps
COPY requirements_a.txt .
RUN pip install --no-cache-dir -r requirements_a.txt

# B 전용 deps
COPY requirements_b.txt .
# 이미 A deps와 충돌 나는 부분(예: numpy, pandas)은 --no-deps 로 설치 제외
RUN pip install --no-cache-dir --no-deps -r requirements_b.txt

# ─── 런타임 스테이지 ────────────────────────────────────
FROM python:3.9-slim
WORKDIR /app/Agent/agent_b

# 빌드된 패키지 복사
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 소스 복사
COPY Agent/agent_b .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]