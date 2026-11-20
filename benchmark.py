"""SPWS benchmark: python benchmark.py   (env: BASE_URL, ADMIN_PASSWORD)
Requires: pip install httpx pandas openpyxl"""
import os, io, time, statistics as st, concurrent.futures as cf, httpx, pandas as pd, numpy as np
BASE = os.getenv("BASE_URL", "http://127.0.0.1:8000"); PW = os.environ["ADMIN_PASSWORD"]
c = httpx.Client(base_url=BASE, timeout=60); assert c.post("/login", data={"password": PW}).status_code in (200, 303)
ck = c.cookies
def timeit(fn, n):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
    ts.sort(); return dict(n=n, mean=round(st.mean(ts), 1), median=round(ts[n // 2], 1), p95=round(ts[int(n * .95) - 1], 1))
body = {"subject": "Linear", "features": [4, 5, 9, 3, 3, 8, 17]}
print("healthz", timeit(lambda: c.get("/healthz"), 200))
print("api/data", timeit(lambda: c.get("/api/data"), 50))
print("api/predict", timeit(lambda: c.post("/api/predict", json=body), 200))
rng = np.random.default_rng(1); n = 1458
df = pd.DataFrame(dict(quiz1_lec=rng.integers(0, 10, n), quiz2_lec=rng.integers(0, 10, n), total_lecture=rng.integers(0, 20, n),
    quiz1_section=rng.integers(0, 5, n), quiz2_section=rng.integers(0, 5, n), total_section=rng.integers(0, 20, n),
    midterm=rng.integers(8, 20, n), student_ID=range(n), name="x", Subject="Linear"))
b = io.BytesIO(); df.to_excel(b, index=False); xb = b.getvalue()
print("batch 1458 rows", timeit(lambda: c.post("/api/batch-predict", files={"file": ("t.xlsx", xb)}), 10))
def load(w, total):
    cl = [httpx.Client(base_url=BASE, cookies=ck, timeout=60) for _ in range(w)]; lat = []
    def job(i):
        t = time.perf_counter(); r = cl[i % w].post("/api/predict", json=body); lat.append((time.perf_counter() - t) * 1000); return r.status_code
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(w) as ex: codes = list(ex.map(job, range(total)))
    el = time.perf_counter() - t0; lat.sort()
    return dict(clients=w, requests=total, errors=sum(x != 200 for x in codes), rps=round(total / el, 1), mean=round(st.mean(lat), 1), p95=round(lat[int(len(lat) * .95) - 1], 1))
for w in (1, 5, 10, 25, 50): print(load(w, w * 40))
