[README.md](https://github.com/user-attachments/files/32200858/README.md)
# Mo Dark AI Ultimate 4.0

نسخة موحدة تجمع أهم أجزاء النسخ السابقة في تطبيق Streamlit واحد.

## المزايا
- Multi-model text
- Vision
- Image generation
- Web search
- URL reader
- Persistent SQLite chat history
- TF-IDF memory
- Automatic agents
- CSV/XLSX/JSON analysis
- PDF/DOCX/text/code extraction
- Image analysis
- Project ZIP builder
- Premium RTL UI

## التشغيل محلياً

```bash
pip install -r requirements.txt
streamlit run app.py
```

ثم ضع مفتاح Hugging Face في:

`.streamlit/secrets.toml`

```toml
HF_TOKEN = "ضع_التوكن_هنا"
```

## Streamlit Community Cloud

ضع `app.py` و `requirements.txt` في GitHub.
بعدها من Streamlit Community Cloud اختر المستودع والملف `app.py`.

من إعدادات Secrets أضف:

```toml
HF_TOKEN = "..."
```

## ملاحظة مهمة

هذه النسخة لا تنفذ shell commands أو Python عشوائي على السيرفر من داخل الدردشة.
هذا مقصود حتى لا يتحول تطبيق Streamlit المنشور إلى remote code execution.

كذلك كلمة "Ultimate" لا تعني أن Hugging Face يضمن نماذج أو موارد غير محدودة؛
توفر النموذج، الـ provider، السرعة، الحصص، حجم الرفع، والوقت تعتمد على المنصة.
