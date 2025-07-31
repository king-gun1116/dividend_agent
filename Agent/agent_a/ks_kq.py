import yfinance as yf
import pandas as pd

start = "2024-01-01"
end   = "2025-07-02"  # <- 2025-07-01까지 포함되게

kospi = yf.download('^KS11', start=start, end=end)
kosdaq = yf.download('^KQ11', start=start, end=end)

kospi = kospi.reset_index()[["Date", "Close"]].rename(columns={"Date":"date", "Close":"KOSPI"})
kosdaq = kosdaq.reset_index()[["Date", "Close"]].rename(columns={"Date":"date", "Close":"KOSDAQ"})

df = pd.merge(kospi, kosdaq, on="date", how="outer").sort_values("date").reset_index(drop=True)

save_path = "/Users/gun/Desktop/미래에셋 AI 공모전/Agent/agent_a/data/ks_kq.csv"
df.to_csv(save_path, index=False, encoding="utf-8-sig")
print("✅ 저장 완료:", save_path)