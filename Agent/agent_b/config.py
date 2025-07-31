import os

# ─── 프로젝트 루트를 자동으로 잡기 ────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── 데이터베이스 설정 ───────────────────────────────────────
DB_DIR = os.path.join(BASE_DIR, "agent_b", "data", "chroma_db")
COLLECTION_NAME = "dividend"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")