#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime
import subprocess, sys, os

# 프로젝트 루트와 스크립트 경로
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
RUN_PIPELINE      = os.path.join(BASE_DIR, 'run_pipeline.py')
CRAWL_DIVIDEND    = os.path.join(BASE_DIR, 'crawl_dividend.py')
PYTHON            = sys.executable  # 현재 가상환경 파이썬

def job():
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{now}] ▶ 작업 시작")

    # 1) run_pipeline.py 실행
    try:
        subprocess.run(
            [PYTHON, RUN_PIPELINE,
             '--start', '20130101',
             '--end',   datetime.today().strftime('%Y%m%d'),
             '--workers','6'],
            check=True
        )
        print(f"[{now}] ✅ run_pipeline.py 완료")
    except subprocess.CalledProcessError as e:
        print(f"[{now}] ❌ run_pipeline.py 오류: {e}")

    # 2) crawl_dividend.py 실행
    try:
        subprocess.run(
            [PYTHON, CRAWL_DIVIDEND],
            check=True
        )
        print(f"[{now}] ✅ crawl_dividend.py 완료")
    except subprocess.CalledProcessError as e:
        print(f"[{now}] ❌ crawl_dividend.py 오류: {e}")

if __name__ == '__main__':
    sched = BlockingScheduler(timezone='Asia/Seoul')
    # 매일 19:00(Asia/Seoul) 에 job() 실행
    sched.add_job(job, 'cron', hour=19, minute=0)
    print("▶ 스케줄러 시작 — 매일 19시에 run_pipeline.py & crawl_dividend.py 실행합니다.")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        print("▶ 스케줄러 종료")