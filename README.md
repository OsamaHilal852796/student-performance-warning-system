# Student Performance Warning System (SPWS)

تطبيق FastAPI: الباك إند فيه الموديل والداتا، والواجهة صفحة واحدة، والدخول بباسورد.

## تشغيل محلي
```bash
pip install -r requirements.txt
export ADMIN_PASSWORD="باسورد-قوي" SECRET_KEY="نص-عشوائي-طويل"
uvicorn app.main:app --reload
# افتح http://127.0.0.1:8000
```

## الرفع على Render (أسهل طريقة)
1. ارفع المجلد على GitHub (repo **Private**).
2. Render -> New -> Web Service -> اختار الـ repo -> Runtime: **Docker**.
3. Environment Variables: `ADMIN_PASSWORD` و `SECRET_KEY` و `DATA_DIR=/data`.
4. **مهم:** Disks -> Add Disk -> Mount Path `/data`. من غيره أي مادة جديدة بتترفع بتضيع مع كل redeploy.
   الـ Disk محتاج خطة مدفوعة. على المجانية الداتا الأصلية بترجع بس الإضافات بتضيع.
5. افتح الرابط وسجل دخول.

## الرفع على Google Cloud Run
```bash
gcloud run deploy spws --source . --region europe-west1 --allow-unauthenticated \
  --set-env-vars ADMIN_PASSWORD=xxx,SECRET_KEY=yyy,DATA_DIR=/data
```
Cloud Run مفيهوش تخزين دائم افتراضي: اربط Cloud Storage volume على `/data` أو انقل التخزين لقاعدة بيانات.

## تغيير الباسورد
- من جوه التطبيق: زرار **🔑 Password** في الناف بار، اكتب الباسورد الحالي والجديد (8 حروف على الأقل).
- الباسورد الجديد بيتخزن مشفّر (PBKDF2 + salt) في `DATA_DIR/auth.json`، وأي جلسة تانية مفتوحة بتتقفل.
- **مهم:** لازم `DATA_DIR` يكون على disk دائم، وإلا الباسورد بيرجع لقيمة `ADMIN_PASSWORD` مع كل redeploy.
- لو نسيت الباسورد: امسح `auth.json` من الـ disk (أو غيّر `ADMIN_PASSWORD` وامسح الملف) وهيرجع للباسورد اللي في الـ environment variable.

## الإيميلات
الإرسال بيشتغل لو ضبطت `SMTP_*` **وكانت بيانات الطالب فيها حقل `email`**. الداتا الحالية مفيهاش إيميلات ولا تليفونات، فكل الطلاب بيظهروا `skipped: no email on file`. الـ SMS محتاج مزود (Twilio مثلا) ومش متوصل.

## ملفات الرفع (Excel/CSV)
الأعمدة: `quiz1_lec, quiz2_lec, total_lecture, quiz1_section, quiz2_section, total_section, midterm` + `student_ID, name` (+ `Subject` اختياري).
لإضافة مادة جديدة لازم كمان `final` و `final_total`، وعلى الأقل 20 طالب.

---
## Code layout
- `app/main.py` - web layer only (FastAPI: login, API, uploads, alerts, storage)
- `app/ridge_model.py` - machine-learning core (identical to `ml/ridge_model.py`), imported by `main.py`
- `ml/` - standalone training/evaluation script
- `app/static/` - frontend (HTML/CSS/JS)
