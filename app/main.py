"""SPWS - web layer (FastAPI): login, REST API, file upload, alerts, static pages.
The machine-learning code lives in ridge_model.py."""
import os, io, re, json, html, hmac, time, smtplib, threading, hashlib, secrets
from email.message import EmailMessage
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, Request, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from itsdangerous import URLSafeTimedSerializer, BadSignature
from .ridge_model import FEAT, ridge_train, ridge_predict, predicted_total, predict_student, risk  # ML core

BASE = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
STORE = DATA_DIR / "store.json"
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
signer = URLSafeTimedSerializer(os.getenv("SECRET_KEY") or ADMIN_PASSWORD or "dev")
lock = threading.Lock()
MAX_UPLOAD = 10 * 1024 * 1024

app = FastAPI(title="Student Performance Warning System", docs_url=None, redoc_url=None)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ---------- storage ----------
def load():
    with lock:
        if not STORE.exists():
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            STORE.write_text((BASE / "seed.json").read_text("utf8"), "utf8")
        return json.loads(STORE.read_text("utf8"))

def save(d):
    with lock:
        tmp = STORE.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False), "utf8")
        tmp.replace(STORE)

def stats(preds):
    out = {}
    for s in dict.fromkeys(p["subject"] for p in preds):
        g = [p for p in preds if p["subject"] == s]
        c = lambda r: sum(p["risk"] == r for p in g)
        out[s] = dict(students=len(g), avg_total=round(sum(p["pred_total"] for p in g) / len(g), 1),
                      pass_rate=round(100 * sum(bool(p.get("passed")) for p in g) / len(g), 1),
                      high_risk=c("High Risk"), med_risk=c("Medium Risk"), low_risk=c("Low Risk"))
    return out

# ---------- auth ----------
AUTH = DATA_DIR / "auth.json"   # created when the password is changed from inside the app
PBKDF2_ROUNDS = 200_000
MIN_PW = 8

def _hash(pw: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, PBKDF2_ROUNDS).hex()

def _auth_record():
    try: return json.loads(AUTH.read_text("utf8")) if AUTH.exists() else None
    except Exception: return None

def check_password(pw: str) -> bool:
    rec = _auth_record()
    if rec: return hmac.compare_digest(_hash(pw, bytes.fromhex(rec["salt"])), rec["hash"])
    return bool(ADMIN_PASSWORD) and hmac.compare_digest(pw.encode(), ADMIN_PASSWORD.encode())

def fingerprint() -> str:
    """Identifies the current password; sessions made with an old password stop being valid."""
    rec = _auth_record()
    return rec["hash"][:16] if rec else "env"

def set_session(resp, request: Request):
    resp.set_cookie("spws", signer.dumps(fingerprint()), max_age=8 * 3600, httponly=True, samesite="lax",
                    secure=request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https")
    return resp

def authed(request: Request):
    try:
        ok = hmac.compare_digest(str(signer.loads(request.cookies.get("spws", ""), max_age=8 * 3600)), fingerprint())
    except BadSignature:
        ok = False
    if not ok: raise HTTPException(401, "Not logged in")

attempts = {}
def throttle(key: str):
    now = time.time(); attempts[key] = [t for t in attempts.get(key, []) if now - t < 60]
    if len(attempts[key]) >= 5: raise HTTPException(429, "Too many attempts, try again in a minute")
    attempts[key].append(now)

@app.post("/login")
def login(request: Request, password: str = Form(...)):
    if not ADMIN_PASSWORD and not AUTH.exists():
        raise HTTPException(500, "ADMIN_PASSWORD is not set on the server")
    throttle(request.client.host if request.client else "?")
    if not check_password(password):
        return RedirectResponse("/login?error=1", 303)
    return set_session(RedirectResponse("/", 303), request)

class PwIn(BaseModel):
    current: str
    new: str

@app.post("/api/change-password", dependencies=[Depends(authed)])
def change_password(b: PwIn, request: Request):
    throttle("pw:" + (request.client.host if request.client else "?"))
    if not check_password(b.current): raise HTTPException(403, "Current password is incorrect")
    if len(b.new) < MIN_PW: raise HTTPException(422, f"New password must be at least {MIN_PW} characters")
    if b.new == b.current: raise HTTPException(422, "New password must be different from the current one")
    salt = secrets.token_bytes(16)
    with lock:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = AUTH.with_suffix(".tmp"); tmp.write_text(json.dumps(dict(salt=salt.hex(), hash=_hash(b.new, salt))), "utf8"); tmp.replace(AUTH)
    return set_session(JSONResponse(dict(ok=True)), request)   # keeps this session; other sessions are signed out

@app.get("/login")
def login_page(): return FileResponse(BASE / "static/login.html")

@app.post("/logout")
def logout():
    r = RedirectResponse("/login", 303); r.delete_cookie("spws"); return r

@app.get("/")
def index(request: Request):
    try: authed(request)
    except HTTPException: return RedirectResponse("/login")
    return FileResponse(BASE / "static/index.html")

@app.get("/healthz")
def healthz(): return {"ok": True}

# ---------- API ----------
@app.get("/api/data", dependencies=[Depends(authed)])
def data():
    d = load(); p = d["predictions"]
    return dict(predictions=p, stats=stats(p), subjects=list(d["models"]))

class PredictIn(BaseModel):
    subject: str
    features: list[float]

@app.post("/api/predict", dependencies=[Depends(authed)])
def predict(b: PredictIn):
    m = load()["models"].get(b.subject)
    if not m: raise HTTPException(404, "Unknown subject")
    if len(b.features) != len(FEAT): raise HTTPException(422, f"Expected {len(FEAT)} features")
    return predict_student(m, b.features)

def read_table(up: UploadFile):
    raw = up.file.read(MAX_UPLOAD + 1)
    if len(raw) > MAX_UPLOAD: raise HTTPException(413, "File too large (max 10MB)")
    try:
        df = pd.read_csv(io.BytesIO(raw)) if (up.filename or "").lower().endswith(".csv") else pd.read_excel(io.BytesIO(raw))
    except Exception:
        raise HTTPException(422, "Could not read the file. Use .xlsx or .csv")
    df.columns = [str(c).strip() for c in df.columns]
    if df.empty: raise HTTPException(422, "Empty file")
    return df

def num(df, c): return pd.to_numeric(df[c], errors="coerce").fillna(0)

def build(df, subj, pf):
    out = []
    for i, (_, r) in enumerate(df.iterrows()):
        mid, tl, ts = float(r["midterm"] or 0), float(r["total_lecture"] or 0), float(r["total_section"] or 0)
        pt = predicted_total(mid, tl, ts, pf[i])
        out.append(dict(id=html.escape(str(r.get("student_ID", i))), name=html.escape(str(r.get("name", "—"))), subject=subj,
                        midterm=mid, total_lecture=tl, total_section=ts, pred_final=round(float(pf[i]), 1),
                        pred_total=round(pt, 1), risk=risk(pt)))
    return out

def need(df, cols):
    miss = [c for c in cols if c not in df.columns]
    if miss: raise HTTPException(422, "Missing columns: " + ", ".join(miss))

@app.post("/api/batch-predict", dependencies=[Depends(authed)])
def batch(file: UploadFile = File(...), subject: str = Form("")):
    models = load()["models"]; df = read_table(file); need(df, FEAT)
    for c in FEAT: df[c] = num(df, c)
    subj_col = df["Subject"].astype(str) if "Subject" in df.columns else pd.Series([subject] * len(df))
    results = []
    for s, g in df.groupby(subj_col):
        if s not in models: raise HTTPException(404, f"Unknown subject: {s}")
        results += build(g, s, ridge_predict(models[s], g[FEAT].values))
    return dict(results=results[:2000], total=len(results))

@app.post("/api/subjects", dependencies=[Depends(authed)])
def add_subject(file: UploadFile = File(...), name: str = Form(...), final_max: float = Form(40)):
    name = name.strip()
    if not re.fullmatch(r"[\w \-\u0600-\u06FF]{1,40}", name): raise HTTPException(422, "Invalid subject name")
    d = load()
    if name in d["models"]: raise HTTPException(409, "Subject already exists")
    df = read_table(file); need(df, FEAT + ["final", "final_total"])
    if len(df) < 20: raise HTTPException(422, "Need at least 20 students to train a model")
    for c in FEAT + ["final", "final_total"]: df[c] = num(df, c)
    model = ridge_train(df[FEAT].values, df["final"].values, hi=final_max)
    new = build(df, name, ridge_predict(model, df[FEAT].values))
    for p, (_, r) in zip(new, df.iterrows()):
        p["actual_final"], p["actual_total"], p["passed"] = float(r["final"]), float(r["final_total"]), bool(r["final_total"] >= 50)
    d["models"][name] = model; d["predictions"] += new; save(d)
    return dict(stats=stats(new)[name])

class AlertIn(BaseModel):
    subject: str
    risks: list[str]
    method: str = "email"
    sender: str = "Academic Affairs Office"
    reply_to: str = ""
    messages: dict[str, str]

@app.post("/api/alerts/send", dependencies=[Depends(authed)])
def send_alerts(b: AlertIn):
    targets = [p for p in load()["predictions"] if p["subject"] == b.subject and p["risk"] in b.risks][:500]
    host = os.getenv("SMTP_HOST"); log = []; smtp = None
    try:
        if host and b.method != "sms":
            smtp = smtplib.SMTP(host, int(os.getenv("SMTP_PORT", 587)), timeout=20); smtp.starttls()
            if os.getenv("SMTP_USER"): smtp.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASS", ""))
        for p in targets:
            st = "sent"
            if b.method == "sms": st = "skipped: SMS provider not configured"
            elif not p.get("email"): st = "skipped: no email on file"
            elif not smtp: st = "skipped: SMTP not configured"
            else:
                try:
                    m = EmailMessage(); m["From"] = os.getenv("SMTP_FROM") or os.getenv("SMTP_USER"); m["To"] = p["email"]
                    m["Subject"] = f"Academic notice — {b.subject}"
                    if b.reply_to: m["Reply-To"] = b.reply_to
                    m.set_content(b.messages.get(p["id"], "")); smtp.send_message(m)
                except Exception as e: st = f"failed: {type(e).__name__}"
            log.append(dict(id=p["id"], name=p["name"], risk=p["risk"], status=st))
    finally:
        if smtp: smtp.quit()
    return dict(sent=sum(l["status"] == "sent" for l in log), skipped=sum(l["status"] != "sent" for l in log), log=log)
