import streamlit as st
from pypdf import PdfReader, PdfWriter
import fitz
from PIL import Image
import re, io, os, zipfile, base64
import anthropic

# ── 常數 ─────────────────────────────────────────────────
DOC_PRIORITY = ["RoHS", "聲明書", "申請書"]
DOC_TITLE_KW = {
    "申請書": ["商品驗證登錄申請書", "Registration of Product Certification"],
    "聲明書": ["符合型式聲明書", "Declaration of Conformity to Type"],
    "RoHS":   ["RoHS"],
}
NOT_NEW_DOC_KW = [
    "核備申請資料", "系列型號清單", "試驗報告清單", "附表",
    "系列型號單", "試驗報告單", "系列型式", "型號清單",
    "報告清單", "附件", "續頁",
]
DOC_FILENAMES = {
    "申請書": "00_01 商品驗證登錄申請書.pdf",
    "聲明書": "00_07 符合型式聲明書.pdf",
    "RoHS":   "07_99 RoHS切結書.pdf",
}

_cache: dict = {}

# ── 工具函式 ──────────────────────────────────────────────
def safe_name(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", s or "未識別")

def clean_title_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", "", text)
    text = safe_name(text)
    text = text.strip("_-—－：:，,。.")
    return text[:40]

# ── 圖片轉換 ─────────────────────────────────────────────
def _page_to_img(page, scale: int = 2) -> Image.Image:
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def _img_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ── Claude OCR ────────────────────────────────────────────
def claude_ocr(page, api_key: str) -> dict:
    key = id(page)
    if key in _cache:
        return _cache[key]

    # 若 PDF 有文字層直接用，不呼叫 API
    full_text = page.get_text().strip()
    if len(full_text) > 80:
        dtype = detect_type_from_text(full_text)
        custom_title = None
        if dtype is None:
            custom_title = extract_custom_title_from_page(page, full_text)
        out = {
            "dtype":        dtype,
            "ci":           extract_ci(full_text),
            "model":        extract_model(full_text),
            "custom_title": custom_title,
            "raw":          full_text[:500],
        }
        _cache[key] = out
        return out

    # 純掃描 → 用 Claude API 辨識
    client = anthropic.Anthropic(api_key=api_key)
    img = _page_to_img(page, scale=2)
    b64 = _img_to_base64(img)

    prompt = """請辨識這份台灣商品驗證登錄文件，提取以下資訊並以JSON格式回傳：
{
  "doc_type": "申請書 或 聲明書 或 RoHS 或 其他",
  "cert_no": "CI開頭的受理編號或證書號碼，例如 CI3A2261861837",
  "model": "室外機型號，例如 RXQ8AYLT 或 RXYQ8AYLT",
  "title": "若doc_type為其他，請填寫文件標題，否則填null"
}
如果找不到某個欄位，填 null。只回傳JSON，不要其他文字。"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64,
                    }
                },
                {"type": "text", "text": prompt}
            ]
        }]
    )

    text = response.content[0].text.strip()

    try:
        import json
        clean = re.sub(r'```json|```', '', text).strip()
        result = json.loads(clean)
        dtype = _map_dtype(result.get("doc_type", ""))
        custom_title = None
        if dtype is None:
            raw_title = result.get("title") or ""
            custom_title = clean_title_text(raw_title) if raw_title else None
        out = {
            "dtype":        dtype,
            "ci":           result.get("cert_no"),
            "model":        result.get("model"),
            "custom_title": custom_title,
            "raw":          text,
        }
        _cache[key] = out
        return out
    except Exception:
        out = {"dtype": None, "ci": None, "model": None, "custom_title": None, "raw": text}
        _cache[key] = out
        return out

def _map_dtype(s: str):
    if "申請" in s: return "申請書"
    if "聲明" in s: return "聲明書"
    if "RoHS" in s or "rohs" in s.lower(): return "RoHS"
    return None

# ── 文字版辨識 ────────────────────────────────────────────
def fix_ocr(text: str) -> str:
    def _f(m):
        s = m.group(0)
        for old, new in [
            ('IG','16'),('I6','16'),('IO','10'),('I0','10'),
            ('IB','18'),('I8','18'),('IS','15'),('I5','15'),
            ('I2','12'),('I4','14'),('I3','13'),('I9','19'),
        ]:
            s = s.replace(old, new)
        return s
    return re.sub(r'[A-Z]{2,}[A-Z0-9]+(?:LT|ET|VT)', _f, text)

def detect_type_from_text(text: str):
    if any(kw in text for kw in NOT_NEW_DOC_KW):
        return None
    for dt in DOC_PRIORITY:
        if any(kw in text for kw in DOC_TITLE_KW[dt]):
            return dt
    return None

def extract_ci(text: str):
    compact = re.sub(r"\s+", "", text)
    for pat in [
        re.compile(r'受理編號[：:﹕]?(CI\w+)'),
        re.compile(r'證書號碼[：:﹕]?(CI\w+)'),
        re.compile(r'(CI[3-9A-Z]\w{9,})'),
    ]:
        m = pat.search(compact)
        if m: return m.group(1)
    return None

def extract_model(text: str):
    fixed = fix_ocr(text)
    fixed = re.sub(r"[（(][^）)]{0,20}[）)]", "", fixed)
    compact = re.sub(r"\s+", "", fixed)
    for pat in [
        re.compile(r'申請主型式[：:﹕]?([A-Z][A-Z0-9]+(?:LT|ET|VT))', re.I),
        re.compile(r'型式[：:﹕]?([A-Z]{2,}[A-Z0-9]{3,}(?:LT|ET|VT))', re.I),
        re.compile(r'([A-Z]{2,}[A-Z0-9]{3,}(?:AYLT|BYLT|AVET|AVLT|ZVLT))'),
        re.compile(r'([A-Z]{2,}[A-Z0-9]{5,}(?:LT|ET|VT))'),
    ]:
        m = pat.search(compact)
        if m: return m.group(1).upper()
    return None

def extract_custom_title_from_page(page, fallback_text: str = "") -> str:
    try:
        d = page.get_text("dict")
        page_h = float(page.rect.height)
        spans = []
        for block in d.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    txt = clean_title_text(span.get("text", ""))
                    if not txt or len(txt) < 4:
                        continue
                    bbox = span.get("bbox", [0, 0, 0, 0])
                    if bbox[1] > page_h * 0.45:
                        continue
                    spans.append((float(span.get("size", 0)), bbox[1], txt))
        if spans:
            spans.sort(key=lambda x: (-x[0], x[1]))
            return spans[0][2]
    except Exception:
        pass

    ignore_words = ["頁", "第", "日期", "編號", "電話", "傳真", "地址", "申請人", "負責人"]
    for line in fallback_text.splitlines()[:8]:
        line_clean = clean_title_text(line.strip())
        if not line_clean or len(line_clean) < 4:
            continue
        if any(w in line_clean for w in ignore_words):
            continue
        return line_clean
    return ""

# ── 主解析流程 ────────────────────────────────────────────
def parse_pdf(uploaded_bytes: bytes, api_key: str, progress_cb=None) -> list:
    doc   = fitz.open(stream=uploaded_bytes, filetype="pdf")
    total = len(doc)
    segments = []

    for i in range(total):
        page = doc[i]
        r = claude_ocr(page, api_key)

        if progress_cb:
            progress_cb((i + 1) / total, f"已完成 {i+1}/{total} 頁…")

        dtype = r["dtype"]

        if dtype in ("申請書", "聲明書", "RoHS"):
            segments.append({
                "type":         dtype,
                "ci":           r["ci"],
                "model":        r["model"],
                "custom_title": None,
                "page_idxs":    [i],
                "raw":          r.get("raw", ""),
            })

        elif r.get("custom_title"):
            if (segments and segments[-1]["type"] == "自訂"
                    and segments[-1]["custom_title"] == r["custom_title"]):
                segments[-1]["page_idxs"].append(i)
            else:
                segments.append({
                    "type":         "自訂",
                    "ci":           None,
                    "model":        None,
                    "custom_title": r["custom_title"],
                    "page_idxs":    [i],
                    "raw":          r.get("raw", ""),
                })

        else:
            target = next((s for s in reversed(segments) if s["type"] == "申請書"), None)
            if target is None and segments:
                target = segments[-1]
            if target:
                target["page_idxs"].append(i)

    # CI ↔ model 互補
    ci_model = {s["ci"]: s["model"] for s in segments if s.get("ci") and s.get("model")}
    model_ci = {v: k for k, v in ci_model.items()}
    for s in segments:
        if not s.get("ci") and s.get("model") and s["model"] in model_ci:
            s["ci"] = model_ci[s["model"]]
        if not s.get("model") and s.get("ci") and s["ci"] in ci_model:
            s["model"] = ci_model[s["ci"]]

    # 合併同 CI 的重複申請書
    merged = []
    for s in segments:
        if (s["type"] == "申請書" and s.get("ci") and merged
                and merged[-1]["type"] == "申請書"
                and merged[-1].get("ci") == s.get("ci")):
            merged[-1]["page_idxs"].extend(s["page_idxs"])
        else:
            merged.append(s)

    doc.close()
    return merged

# ── 建立 ZIP ──────────────────────────────────────────────
def build_zip(uploaded_bytes: bytes, segments: list) -> bytes:
    pdf_reader = PdfReader(io.BytesIO(uploaded_bytes))

    model_folder = {}
    for s in segments:
        if s.get("ci") and s.get("model"):
            model_folder[s["model"]] = f"{safe_name(s['ci'])}-{safe_name(s['model'])}"

    used_paths = set()

    def unique_zip_path(path: str) -> str:
        path = path.replace("\\", "/")
        if path not in used_paths:
            used_paths.add(path)
            return path
        folder, filename = os.path.split(path)
        name, ext = os.path.splitext(filename)
        n = 1
        while True:
            new_filename = f"{name}({n}){ext}"
            new_path = f"{folder}/{new_filename}" if folder else new_filename
            if new_path not in used_paths:
                used_paths.add(new_path)
                return new_path
            n += 1

    def get_filename(seg: dict) -> str:
        dtype = seg.get("type")
        if dtype in DOC_FILENAMES:
            return DOC_FILENAMES[dtype]
        if dtype == "自訂" and seg.get("custom_title"):
            return f"{safe_name(seg['custom_title'])}.pdf"
        pages = seg.get("page_idxs") or []
        return f"未識別文件_p{pages[0]+1}.pdf" if pages else "未識別文件.pdf"

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for seg in segments:
            model = seg.get("model")
            dtype = seg.get("type")
            fname = get_filename(seg)

            writer = PdfWriter()
            for pidx in sorted(seg["page_idxs"]):
                writer.add_page(pdf_reader.pages[pidx])
            pdf_buf = io.BytesIO()
            writer.write(pdf_buf)

            folder = model_folder.get(model)
            if dtype in DOC_FILENAMES and folder:
                zip_path = f"{folder}/{fname}"
            else:
                zip_path = fname

            zip_path = unique_zip_path(zip_path)
            zf.writestr(zip_path, pdf_buf.getvalue())

    return zip_buf.getvalue()

# ════════════════════════════════════════════════════════
# Streamlit UI
# ════════════════════════════════════════════════════════
st.set_page_config(page_title="PDF 歸檔工具", page_icon="🗂️", layout="centered")
st.markdown("""
<style>
.stApp{background:#EAF4FB;color:#1A3A4F!important;}
.stApp h1,.stApp h2,.stApp h3,.stApp p,.stApp label{color:#1A3A4F!important;}
.block-container{max-width:680px!important;padding:1rem 1.5rem 3rem!important;margin:auto!important;}
.stApp .stButton>button{width:100%!important;min-height:46px!important;font-size:.95rem!important;
  font-weight:600!important;background:#fff!important;color:#2176AE!important;
  border:1.5px solid #2176AE!important;border-radius:10px!important;}
.stApp .stButton>button:hover{background:#2176AE!important;color:#fff!important;}
.stApp .stDownloadButton>button{width:100%!important;min-height:50px!important;
  font-size:1rem!important;font-weight:700!important;background:#2176AE!important;
  color:#fff!important;border:none!important;border-radius:10px!important;}
.stApp hr{border-color:#B3D4E8!important;margin:.8rem 0!important;}
#MainMenu,footer,header{visibility:hidden;}
</style>""", unsafe_allow_html=True)

st.title("🗂️ PDF 歸檔工具")
st.caption("上傳掃描合冊 PDF，自動切割並依證書編號/型號建立資料夾")
st.markdown("---")

api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
if not api_key:
    st.error("系統未設定 ANTHROPIC_API_KEY，請聯絡管理員")
    st.stop()

for k, v in [("zip_bytes",None),("zip_name",""),("segments",None),("last_file",None),("ready",False)]:
    if k not in st.session_state:
        st.session_state[k] = v

uploaded = st.file_uploader("請上傳掃描 PDF", type=["pdf"])

if uploaded and uploaded.name != st.session_state["last_file"]:
    st.session_state.update({"zip_bytes":None,"zip_name":"","segments":None,
                              "ready":False,"last_file":uploaded.name})
    _cache.clear()

if uploaded:
    uploaded_bytes = uploaded.read()
    total_pages = len(PdfReader(io.BytesIO(uploaded_bytes)).pages)

    if st.session_state["segments"] is None:
        st.info(f"共 {total_pages} 頁，AI 辨識中…")
        prog = st.progress(0)
        stat = st.empty()
        def _upd(pct, msg): prog.progress(pct); stat.caption(msg)

        with st.spinner("AI 辨識中…"):
            try:
                segs = parse_pdf(uploaded_bytes, api_key, progress_cb=_upd)
            except Exception as e:
                st.error(f"辨識失敗：{e}")
                st.stop()

        prog.progress(1.0); stat.caption("✅ 辨識完成")
        st.session_state["segments"]  = segs
        st.session_state["zip_bytes"] = build_zip(uploaded_bytes, segs)
        st.session_state["zip_name"]  = os.path.splitext(uploaded.name)[0] + "_歸檔.zip"
        st.session_state["ready"]     = True

    segs = st.session_state["segments"]
    model_folder = {}
    for s in segs:
        if s.get("ci") and s.get("model"):
            model_folder[s["model"]] = f"{s['ci']}-{s['model']}"

    ok  = sum(1 for s in segs if s.get("model") and s.get("model") in model_folder)
    bad = len(segs) - ok
    st.success(f"共 {total_pages} 頁，識別出 {len(segs)} 份文件（✅ {ok} 份可歸檔，⚠ {bad} 份放根目錄）")
    st.markdown("---")
    st.subheader("識別結果")

    show_debug = st.checkbox("顯示 AI 辨識原始輸出", value=False)

    for seg in segs:
        ci    = seg.get("ci")    or "⚠ 未識別CI"
        model = seg.get("model") or "⚠ 未識別型號"
        dtype = seg.get("type")

        if dtype == "自訂" and seg.get("custom_title"):
            fname = f"{safe_name(seg['custom_title'])}.pdf"
        else:
            fname = DOC_FILENAMES.get(dtype, dtype or "未識別")

        pages    = seg["page_idxs"]
        page_str = f"p{min(pages)+1}" if len(pages)==1 else f"p{min(pages)+1}–{max(pages)+1}"
        folder   = model_folder.get(seg.get("model"))

        if folder:
            st.write(f"✅ `{ci}-{model}` / **{fname}** ({page_str}, {len(pages)}頁) → `{folder}`")
        else:
            st.warning(f"`{ci}-{model}` / **{fname}** ({page_str}, {len(pages)}頁) → 放於 ZIP 根目錄")

        if show_debug and "raw" in seg:
            with st.expander(f"AI 原始輸出 — 頁{min(pages)+1} [{dtype}]", expanded=False):
                st.code(seg["raw"], language=None)

    st.markdown("---")
    st.subheader("下載")
    if st.session_state["ready"] and st.session_state["zip_bytes"]:
        st.download_button(
            label="📥 下載 ZIP",
            data=st.session_state["zip_bytes"],
            file_name=st.session_state["zip_name"],
            mime="application/zip",
            use_container_width=True,
        )
        st.caption("主要文件放入 CI-型號資料夾；其他文件放 ZIP 根目錄，同名自動加 (1)(2)。")
    else:
        st.button("📥 下載 ZIP（辨識完成後可用）", disabled=True, use_container_width=True)
