import streamlit as st
from pypdf import PdfReader, PdfWriter
import fitz
from PIL import Image
import re, io, os, zipfile, tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── PaddleOCR（懶載入，單例）─────────────────────────────
_ocr_engine = None
_ocr_lock = __import__('threading').Lock()

def get_ocr():
    global _ocr_engine
    if _ocr_engine is None:
        with _ocr_lock:
            if _ocr_engine is None:
                from paddleocr import PaddleOCR
                _ocr_engine = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
    return _ocr_engine

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
_cache_lock = __import__('threading').Lock()

# ── OCR 核心 ─────────────────────────────────────────────
def _page_to_img(page, scale: int) -> Image.Image:
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def _ocr_image(img: Image.Image) -> str:
    """對 PIL Image 做 PaddleOCR，回傳文字。"""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp.name); path = tmp.name
    try:
        result = get_ocr().ocr(path, cls=True)
        lines = [l[1][0] for l in (result[0] or []) if l[1][1] > 0.4]
        return "\n".join(lines)
    finally:
        os.unlink(path)

def ocr_region(page, top_pct: float, bot_pct: float, scale: int = 2) -> str:
    key = f"{id(page)}_{top_pct}_{bot_pct}_{scale}"
    with _cache_lock:
        if key in _cache:
            return _cache[key]

    full_text = page.get_text().strip()
    if len(full_text) > 80:
        with _cache_lock: _cache[key] = full_text
        return full_text

    img = _page_to_img(page, scale)
    h = img.height
    strip = img.crop((0, int(h * top_pct), img.width, int(h * bot_pct)))
    text = _ocr_image(strip)

    with _cache_lock: _cache[key] = text
    return text

# ── 修正 / 辨識 ───────────────────────────────────────────
def fix_ocr(text: str) -> str:
    def _f(m):
        s = m.group(0)
        for old, new in [
            ('IG','16'),('I6','16'),('IO','10'),('I0','10'),
            ('IB','18'),('I8','18'),('IS','15'),('I5','15'),
            ('I2','12'),('I4','14'),('I3','13'),('I9','19'),
            ('2O','20'),('4O','40'),('6O','60'),('8O','80'),
            ('U2','12'),('U6','16'),('U8','18'),('U0','10'),
        ]:
            s = s.replace(old, new)
        return s
    return re.sub(r'[A-Z]{2,}[A-Z0-9]+(?:LT|ET|VT)', _f, text)

def detect_type(title: str, extra: str = ""):
    combined = title + "\n" + extra
    if any(kw in combined for kw in NOT_NEW_DOC_KW):
        return None
    for dt in DOC_PRIORITY:
        if any(kw in combined for kw in DOC_TITLE_KW[dt]):
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

# ── 單頁處理（可平行執行）────────────────────────────────
def process_page(args):
    """
    對單一頁面做 OCR 並回傳結果。
    回傳 dict：{"idx": i, "dtype": ..., "ci": ..., "model": ..., "debug": ...}
    """
    page, i = args

    title_text = ocr_region(page, 0.0, 0.15, scale=2)
    extra_text = ""
    if not detect_type(title_text):
        extra_text = ocr_region(page, 0.12, 0.22, scale=2)
    dtype = detect_type(title_text, extra_text)

    if not dtype:
        return {"idx": i, "dtype": None}

    ci_text = ocr_region(page, 0.12, 0.35, scale=2)

    if dtype == "申請書":
        model_text = ocr_region(page, 0.43, 0.58, scale=2)
    elif dtype == "聲明書":
        model_text = ocr_region(page, 0.25, 0.42, scale=3)
    else:
        model_text = ocr_region(page, 0.12, 0.30, scale=2)

    combined = fix_ocr(title_text + "\n" + extra_text + "\n" + ci_text + "\n" + model_text)
    return {
        "idx":   i,
        "dtype": dtype,
        "ci":    extract_ci(combined),
        "model": extract_model(combined),
        "debug": {
            "標題區": title_text.strip()[:200],
            "CI區":   ci_text.strip()[:200],
            "型號區": model_text.strip()[:400],
        }
    }

# ── 主解析流程（平行 OCR）────────────────────────────────
def parse_pdf(uploaded_bytes: bytes, progress_cb=None) -> list[dict]:
    doc   = fitz.open(stream=uploaded_bytes, filetype="pdf")
    total = len(doc)
    pages = [(doc[i], i) for i in range(total)]

    # 平行 OCR（4 執行緒；PaddleOCR 本身是 CPU bound，超過 4 效益遞減）
    results = [None] * total
    done = 0

    with ThreadPoolExecutor(max_workers=4) as ex:
        fut_map = {ex.submit(process_page, p): p[1] for p in pages}
        for fut in as_completed(fut_map):
            r = fut.result()
            results[r["idx"]] = r
            done += 1
            if progress_cb:
                progress_cb(done / total, f"已完成 {done}/{total} 頁…")

    # 依頁序重建 segments（平行結果需排序）
    segments = []
    for r in results:
        if r["dtype"]:
            segments.append({
                "type":      r["dtype"],
                "ci":        r["ci"],
                "model":     r["model"],
                "page_idxs": [r["idx"]],
                "debug":     r.get("debug", {}),
            })
        else:
            # 附表：歸入最近的申請書
            target = next((s for s in reversed(segments) if s["type"] == "申請書"), None)
            if target is None and segments: target = segments[-1]
            if target: target["page_idxs"].append(r["idx"])

    # 補齊聲明書 / RoHS 缺少的 CI 或型號
    ci_model = {s["ci"]: s["model"] for s in segments
                if s["type"] == "申請書" and s["ci"] and s["model"]}
    model_ci = {v: k for k, v in ci_model.items()}
    for s in segments:
        if not s["ci"]    and s["model"] and s["model"] in model_ci: s["ci"]    = model_ci[s["model"]]
        if not s["model"] and s["ci"]    and s["ci"] in ci_model:    s["model"] = ci_model[s["ci"]]

    # 合併同 CI 的重複申請書（避免同一份被切成兩份）
    merged = []
    for s in segments:
        if (s["type"] == "申請書" and s["ci"] and merged
                and merged[-1]["type"] == "申請書"
                and merged[-1]["ci"] == s["ci"]):
            # 同 CI → 合併頁碼
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
        for i, seg in enumerate(segments):
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
        st.info(f"共 {total_pages} 頁，平行分析中（4 執行緒）…")
        prog = st.progress(0)
        stat = st.empty()
        def _upd(pct, msg): prog.progress(pct); stat.caption(msg)

        with st.spinner("分析中…"):
            segs = parse_pdf(uploaded_bytes, progress_cb=_upd)

        prog.progress(1.0); stat.caption("✅ 分析完成")
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
    show_debug = st.checkbox("顯示 OCR 除錯資訊", value=False)

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
        if show_debug and "debug" in seg:
            with st.expander(f"OCR 原始輸出 — 頁{min(pages)+1} [{seg['type']}]", expanded=False):
                for zone, txt in seg["debug"].items():
                    st.markdown(f"**{zone}**")
                    st.code(txt, language=None)

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
        st.button("📥 下載 ZIP（分析完成後可用）", disabled=True, use_container_width=True)
