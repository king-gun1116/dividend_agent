# ---------------------------------------------------------------
# scripts/ingest.py
#!/usr/bin/env python3
import os
import sys

# 프로젝트 최상위 루트 인식
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

from Agent.agent_b.config import DB_DIR
from Agent.agent_b.model import build_collection

if __name__ == "__main__":
    print("▶ INGEST 시작 (증분 추가 모드)")
    build_collection()
    print("✅ 컬렉션 생성 완료")