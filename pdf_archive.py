import streamlit as st
from pypdf import PdfReader, PdfWriter
import fitz
from PIL import Image
import re, io, os, zipfile, base64
import google.generativeai as genai

# ── 常數 ─────────────────────────────────────────────────
DOC_PRIORITY = ["RoHS", "聲明書", "申請書"]
DOC_TITLE_KW = {
    "申請書": ["商品驗證登錄申請書", "Registration of Product Certification"],
    "聲明書": ["符合型式聲明書", "Declaration of Conformity to Type"],
    "RoHS":   ["RoHS"],
}
NOT_NEW_DOC_KW = ["核備申請資料", "系列型號清單", "試驗報告清單", "附表"]
DOC_FILENAMES = {
    "申請書": "00_01 商品驗證登錄申請書.pdf",
    "聲明書": "00_07 符合型式聲明書.pdf",
    "RoHS":   "07_99 RoHS切結書.pdf",
}

_cache: dict = {}

# ── Gemini 辨識 ───────────────────────────────────────────
def _page_to_img(page, scale: int = 2) -> Image.Image:
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def _img_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def gemini_ocr(page, api_key: str) -> dict:
    key = id(page)
    if key in _cache:
        return _cache[key]

    full_text = page.get_text().strip()

    if len(full_text) > 80:
        out = {
            "dtype": detect_type_from_text(full_text),
            "ci":    extract_ci(full_text),
            "model": extract_model(full_text),
            "raw":   full_text,
        }
        _cache[key] = out
        return out

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    img = _page_to_img(page, scale=2)
    b64 = _img_to_base64(img)
    prompt = """請辨識這份台灣商品驗證登錄文件，提取以下資訊並以JSON格式回傳：
{
  "doc_type": "申請書 或 聲明書 或 RoHS 或 其他",
  "cert_no": "CI開頭的受理編號或證書號碼，例如 CI3A2261861837",
  "model": "室外機型號，例如 3MXM90PVLT"
}
如果找不到某個欄位，填 null。只回傳JSON，不要其他文字。"""
    response = model.generate_content([
        {"mime_type": "image/png", "data": b64},
        prompt
    ])
    text = response.text.strip()

    try:
        import json
        clean = re.sub(r'```json|```', '', text).strip()
        result = json.loads(clean)
        out = {
            "dtype": _map_dtype(result.get("doc_type", "")),
            "ci":    result.get("cert_no"),
            "model": result.get("model"),
            "raw":   text,
        }
        _cache[key] = out
        return out
    except Exception:
        out = {"dtype": None, "ci": None, "model": None, "raw": text}
        _cache[key] = out
        return out

def _map_dtype(s: str):
    if "申請" in s: return "申請書"
    if "聲明" in s: return "聲明書"
    if "RoHS" in s or "rohs" in s.lower(): return "RoHS"
    return None

# ── 文字版辨識（備用）────────────────────────────────────
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
    for pat in [
        re.compile(r'受理編號\s*[：:﹕]\s*(CI\w+)'),
        re.compile(r'證書號碼[：:﹕\s]*(CI\w+)'),
        re.compile(r'\b(CI[3-9A-Z]\w{9,})\b'),
    ]:
        m = pat.search(text)
        if m: return m.group(1)
    return None

def extract_model(text: str):
    fixed = fix_ocr(text)
    for pat in [
        re.compile(r'([A-Z]{2,}[A-Z0-9]+(?:AYLT|AVET|BYLT|BYMT|AVLT|AYMT|[A-Z]YLT))\s*[（(]?\s*室外', re.I),
        re.compile(r'申請主型式\s*([A-Z][A-Z0-9]+(?:LT|ET|VT))', re.I),
        re.compile(r'\b([A-Z]{2,}[A-Z0-9]{3,}(?:AYLT|BYLT|AVET|AVLT))\b'),
    ]:
        m = pat.search(fixed)
        if m: return m.group(1)
    return None

# ── 主解析流程 ────────────────────────────────────────────
def parse_pdf(uploaded_bytes: bytes, api_key: str, progress_cb=None) -> list[dict]:
    doc   = fitz.open(stream=uploaded_bytes, filetype="pdf")
    total = len(doc)

    segments = []
    for i in range(total):
        page = doc[i]
        r = gemini_ocr(page, api_key)

        if progress_cb:
            progress_cb((i + 1) / total, f"已完成 {i+1}/{total} 頁…")

        if r["dtype"]:
            segments.append({
                "type":      r["dtype"],
                "ci":        r["ci"],
                "model":     r["model"],
                "page_idxs": [i],
                "raw":       r.get("raw", ""),
            })
        else:
            target = next((s for s in reversed(segments) if s["type"] == "申請書"), None)
            if target is None and segments: target = segments[-1]
            if target: target["page_idxs"].append(i)

    ci_model = {s["ci"]: s["model"] for s in segments
                if s["type"] == "申請書" and s["ci"] and s["model"]}
    model_ci = {v: k for k, v in ci_model.items()}
    for s in segments:
        if not s["ci"]    and s["model"] and s["model"] in model_ci: s["ci"]    = model_ci[s["model"]]
        if not s["model"] and s["ci"]    and s["ci"] in ci_model:    s["model"] = ci_model[s["ci"]]

    merged = []
    for s in segments:
        if (s["type"] == "申請書" and s["ci"] and merged
                and merged[-1]["type"] == "申請書"
                and merged[-1]["ci"] == s["ci"]):
            merged[-1]["page_idxs"].extend(s["page_idxs"])
        else:
            merged.append(s)

    doc.close()
    return merged

def build_zip(uploaded_bytes: bytes, segments: list[dict]) -> bytes:
    pdf_reader = PdfReader(io.BytesIO(uploaded_bytes))
    ci_model = {s["ci"]: s["model"] for s in segments
                if s["type"] == "申請書" and s["ci"] and s["model"]}
    model_ci = {v: k for k, v in ci_model.items()}

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for seg in segments:
            ci    = seg["ci"]    or model_ci.get(seg["model"])
            model = seg["model"] or ci_model.get(seg["ci"])
            fname = DOC_FILENAMES.get(seg["type"], f"{seg['type']}.pdf")
            writer = PdfWriter()
            for pidx in sorted(seg["page_idxs"]):
                writer.add_page(pdf_reader.pages[pidx])
            pdf_buf = io.BytesIO()
            writer.write(pdf_buf)
            if ci and model:
                zf.writestr(f"{ci}-{model}/{fname}", pdf_buf.getvalue())
            else:
                pages = seg["page_idxs"]
                page_str = f"p{pages[0]+1}" if len(pages)==1 else f"p{min(pages)+1}-{max(pages)+1}"
                zf.writestr(f"未識別_{seg['type']}_{page_str}.pdf", pdf_buf.getvalue())
    return zip_buf.getvalue()

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

api_key = st.text_input("請輸入 Gemini API Key", type="password",
                         placeholder="AIza...")
if not api_key:
    st.warning("請先輸入 Gemini API Key 才能使用")
    st.stop()

for k, v in [("zip_bytes",None),("zip_name",""),("segments",None),("last_file",None),("ready",False)]:
    if k not in st.session_state: st.session_state[k] = v

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
        st.session_state["segments"] = segs
        st.session_state["zip_bytes"] = build_zip(uploaded_bytes, segs)
        st.session_state["zip_name"]  = os.path.splitext(uploaded.name)[0] + "_歸檔.zip"
        st.session_state["ready"]     = True

    segs = st.session_state["segments"]
    ok   = sum(1 for s in segs if s["ci"] and s["model"])
    bad  = len(segs) - ok
    st.success(f"共 {total_pages} 頁，識別出 {len(segs)} 份文件（✅ {ok} 份可歸檔，⚠ {bad} 份未完整識別）")
    st.markdown("---")

    st.subheader("識別結果")
    ci_model = {s["ci"]:s["model"] for s in segs if s["type"]=="申請書" and s["ci"] and s["model"]}
    model_ci = {v:k for k,v in ci_model.items()}
    show_debug = st.checkbox("顯示 AI 辨識原始輸出", value=False)

    for seg in segs:
        ci    = seg["ci"]    or model_ci.get(seg["model"]) or "⚠ 未識別"
        model = seg["model"] or ci_model.get(seg["ci"])    or "⚠ 未識別"
        fname = DOC_FILENAMES.get(seg["type"], seg["type"])
        pages = seg["page_idxs"]
        page_str = f"p{min(pages)+1}" if len(pages)==1 else f"p{min(pages)+1}–{max(pages)+1}"
        if "⚠" in ci or "⚠" in model:
            st.warning(f"`{ci}-{model}` / **{fname}** ({page_str}, {len(pages)}頁) → 放於 ZIP 根目錄")
        else:
            st.write(f"✅ `{ci}-{model}` / **{fname}** ({page_str}, {len(pages)}頁)")
        if show_debug and "raw" in seg:
            with st.expander(f"AI 原始輸出 — 頁{min(pages)+1} [{seg['type']}]", expanded=False):
                st.code(seg["raw"], language=None)

    st.markdown("---")
    st.subheader("下載")
    if st.session_state["ready"] and st.session_state["zip_bytes"]:
        st.download_button(
            label="📥 下載 ZIP",
            data=st.session_state["zip_bytes"],
            file_name=st.session_state["zip_name"],
            mime="application/zip", use_container_width=True,
        )
        st.caption("識別成功的檔案在各資料夾中；未識別的在 ZIP 根目錄，請手動處理")
    else:
        st.button("📥 下載 ZIP（辨識完成後可用）", disabled=True, use_container_width=True)
