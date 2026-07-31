"""
PDF 掃描檔自動分割工具（Starbucks 風格）
=====================================
掃描機一次掃入多份文件（30~50 頁）時，依「標題行」（字型較大 / 上下留白的單獨一行）
自動判斷分割點，切成多份 PDF；分割線可用滑鼠拖曳調整（自訂元件，架構比照
floorplan_editor：純手刻 postMessage 協議，無需 npm/React build），
最後打包成 zip 供下載。

執行方式：
    pip install -r requirements.txt
    streamlit run app.py
"""

import base64
import io
import re
import zipfile
import statistics
import hashlib

import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
from pypdf import PdfReader, PdfWriter

from split_editor import split_editor

# ----------------------------------------------------------------------------
# 基本設定
# ----------------------------------------------------------------------------
st.set_page_config(page_title="PDF 自動分割工具", page_icon="☕", layout="wide")

FULL_ZOOM = 1.3              # 用來產生「最終分割」PDF 判讀時的縮圖解析度（不影響分割本身）
COMPONENT_THUMB_WIDTH = 260   # 傳給拖曳元件的縮圖寬度（px），越小 payload 越輕
TITLE_SIZE_RATIO = 1.15
TITLE_MAX_CHARS = 40

CERT_KEYWORDS = [
    "登錄證書編號", "驗證登錄證書編號", "商品驗證登錄證書編號", "證書編號", "證書字號",
    "登錄字號", "登錄編號", "認證編號", "認證字號", "許可字號", "核可字號",
]
MODEL_KEYWORDS = [
    "室外機型號", "室外機機型", "室內機型號", "產品型號", "商品型號", "型號", "機型",
]

# ----------------------------------------------------------------------------
# Starbucks 風格 CSS（主頁面）
# ----------------------------------------------------------------------------
STARBUCKS_CSS = """
<style>
:root {
    --sb-green: #00704A;
    --sb-green-dark: #063528;
    --sb-cream: #F2EFE4;
    --sb-gold: #C6A664;
    --sb-white: #FFFFFF;
}
.stApp { background-color: var(--sb-cream); }
.sb-header {
    background: linear-gradient(135deg, var(--sb-green-dark) 0%, var(--sb-green) 100%);
    padding: 28px 36px; border-radius: 14px; margin-bottom: 24px;
    box-shadow: 0 6px 18px rgba(6, 53, 40, 0.25);
}
.sb-header h1 { color: var(--sb-white); font-family: 'Georgia', serif; font-size: 30px; margin: 0 0 4px 0; letter-spacing: 1px; }
.sb-header p { color: var(--sb-gold); margin: 0; font-size: 14px; letter-spacing: 0.5px; }
.sb-section {
    color: var(--sb-green-dark); font-family: 'Georgia', serif; font-weight: 700; font-size: 20px;
    border-left: 6px solid var(--sb-gold); padding-left: 12px; margin: 22px 0 14px 0;
}
.sb-doc-card {
    background: var(--sb-white); border-left: 6px solid var(--sb-green); border-radius: 8px;
    padding: 12px 16px; margin-bottom: 10px; box-shadow: 0 2px 8px rgba(6,53,40,0.08);
}
.sb-doc-card .fname { color: var(--sb-green-dark); font-weight: 700; font-size: 15px; }
.sb-doc-card .pages { color: #7a7a7a; font-size: 12px; }
div.stButton > button, div.stDownloadButton > button {
    background-color: var(--sb-green) !important; color: var(--sb-white) !important;
    border-radius: 999px !important; border: none !important; padding: 8px 22px !important; font-weight: 700 !important;
}
div.stButton > button:hover, div.stDownloadButton > button:hover { background-color: var(--sb-green-dark) !important; }
</style>
"""
st.markdown(STARBUCKS_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# 核心邏輯：標題偵測、分件、命名
# ----------------------------------------------------------------------------
def sanitize_filename(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:80] if name else "未命名文件"


def extract_page_lines(page):
    lines_info = []
    d = page.get_text("dict")
    for block in d.get("blocks", []):
        block_lines = block.get("lines", [])
        n_lines_in_block = len(block_lines)
        for line in block_lines:
            spans = line.get("spans", [])
            if not spans:
                continue
            text = "".join(s.get("text", "") for s in spans).strip()
            if not text:
                continue
            max_size = max(s.get("size", 0) for s in spans)
            y0 = line.get("bbox", [0, 0, 0, 0])[1]
            lines_info.append({
                "text": text, "size": max_size, "y0": y0,
                "isolated": n_lines_in_block <= 2,
            })
    return lines_info


def detect_titles(doc):
    all_sizes = []
    per_page_lines = []
    for page in doc:
        lines = extract_page_lines(page)
        per_page_lines.append(lines)
        all_sizes.extend(l["size"] for l in lines if len(l["text"]) < 60)

    if not all_sizes:
        return {0: "文件"}

    median_size = statistics.median(all_sizes)
    size_threshold = median_size * TITLE_SIZE_RATIO

    titles = {}
    for page_idx, lines in enumerate(per_page_lines):
        if not lines:
            continue
        page_height = max((l["y0"] for l in lines), default=800) + 1
        best, best_score = None, 0
        for line in lines:
            text = line["text"]
            if len(text) == 0 or len(text) > TITLE_MAX_CHARS:
                continue
            score = 0
            if line["size"] >= size_threshold:
                score += 2
            if line["isolated"]:
                score += 1
            if line["y0"] < page_height * 0.35:
                score += 1
            if score > best_score:
                best_score = score
                best = text
        if best is not None and best_score >= 2:
            titles[page_idx] = best

    if 0 not in titles:
        fallback = per_page_lines[0][0]["text"] if per_page_lines and per_page_lines[0] else "文件"
        titles[0] = fallback

    return titles


def normalize_text(text: str) -> str:
    """把換行/多重空白壓成單一空白，避免標籤跟數值中間隔了換行就抓不到。"""
    return re.sub(r"\s+", " ", text)


def extract_code(normalized_text, keywords):
    for kw in keywords:
        pattern = re.escape(kw) + r"[\s:：\-—－\.、]{0,6}([A-Za-z0-9][A-Za-z0-9\-\/\.]{3,24})"
        for m in re.finditer(pattern, normalized_text):
            candidate = m.group(1).strip(" -/.")
            # 真正的型號/證書編號一定含數字；純英文字（如標籤的英文對照 "Serial"）一律跳過
            if len(candidate) >= 4 and any(ch.isdigit() for ch in candidate):
                return candidate
    return None


def get_full_text(doc, page_start, page_end):
    text = ""
    for p in range(page_start, page_end + 1):
        text += doc[p].get_text()
    return text


def guess_filename(doc, page_start, page_end, title_text):
    full_text = get_full_text(doc, page_start, page_end)
    normalized = normalize_text(full_text)

    code = extract_code(normalized, CERT_KEYWORDS) or extract_code(normalized, MODEL_KEYWORDS)
    if not code:
        for m in re.finditer(r"\b[A-Z0-9]{2,}(?:[-\/][A-Z0-9]+){1,4}\b", normalized):
            candidate = m.group(0)
            if any(ch.isdigit() for ch in candidate):
                code = candidate
                break

    title_clean = sanitize_filename(title_text)
    if code:
        return sanitize_filename(f"{title_clean}_{code}")
    return title_clean


def build_groups(splits_sorted, total_pages):
    groups = []
    for i, start in enumerate(splits_sorted):
        end = splits_sorted[i + 1] - 1 if i + 1 < len(splits_sorted) else total_pages - 1
        groups.append((start, end))
    return groups


def render_thumb_png(page, zoom):
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


def to_data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def resize_png_bytes(png_bytes: bytes, width: int) -> bytes:
    im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    w, h = im.size
    new_h = max(1, int(h * width / w))
    im = im.resize((width, new_h))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def split_pdf_to_zip(pdf_bytes, groups_with_names):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    zip_buf = io.BytesIO()
    used_names = {}
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for (start, end), fname in groups_with_names:
            writer = PdfWriter()
            for p in range(start, end + 1):
                writer.add_page(reader.pages[p])
            out = io.BytesIO()
            writer.write(out)
            out.seek(0)

            base = fname if fname.lower().endswith(".pdf") else f"{fname}.pdf"
            if base in used_names:
                used_names[base] += 1
                stem = base[:-4]
                base = f"{stem}_{used_names[base]}.pdf"
            else:
                used_names[base] = 0

            zf.writestr(base, out.read())
    zip_buf.seek(0)
    return zip_buf.getvalue()


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="sb-header">
        <h1>☕ PDF 掃描檔自動分割工具</h1>
        <p>一次掃描多份文件 → 自動判斷標題分頁 → 拖曳調整 → 自動命名 → 打包下載</p>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("上傳掃描 PDF（可能包含 30~50 頁、多份文件）", type=["pdf"])

if uploaded is None:
    st.info("請上傳掃描後的 PDF 檔案，系統會自動偵測每份文件的標題並建議分割點。")
    st.stop()

pdf_bytes = uploaded.read()
file_hash = hashlib.md5(pdf_bytes).hexdigest()

if st.session_state.get("file_hash") != file_hash:
    st.session_state["file_hash"] = file_hash
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    titles = detect_titles(doc)
    splits_sorted = sorted(titles.keys())
    groups = build_groups(splits_sorted, doc.page_count)

    filenames = {}
    for (start, end) in groups:
        title_text = titles.get(start, f"文件_{start+1}")
        filenames[start] = guess_filename(doc, start, end, title_text)

    thumbs_component = [
        to_data_url(resize_png_bytes(render_thumb_png(page, 0.9), COMPONENT_THUMB_WIDTH))
        for page in doc
    ]

    st.session_state["total_pages"] = doc.page_count
    st.session_state["splits"] = set(splits_sorted)
    st.session_state["filenames"] = filenames
    st.session_state["auto_titles_raw"] = dict(titles)
    st.session_state["thumbs_component"] = thumbs_component
    st.session_state["editor_revision"] = st.session_state.get("editor_revision", 0) + 1
    doc.close()

total_pages = st.session_state["total_pages"]
thumbs_component = st.session_state["thumbs_component"]

st.success(f"共 {total_pages} 頁，系統自動偵測到 {len(st.session_state['splits'])} 份文件（可拖曳虛線調整）。")

# ----------------------------------------------------------------------------
# 拖曳式分割線編輯器
# ----------------------------------------------------------------------------
st.markdown('<div class="sb-section">頁面預覽與分割點調整</div>', unsafe_allow_html=True)

pages_payload = []
for idx in range(total_pages):
    is_split = idx in st.session_state["splits"]
    pages_payload.append({
        "index": idx,
        "thumb_data_url": thumbs_component[idx],
        "is_split": is_split,
        "filename": st.session_state["filenames"].get(idx, f"文件_{idx+1}") if is_split else "",
    })

editor_value = split_editor(
    pages=pages_payload,
    revision=st.session_state["editor_revision"],
    key="pdf_split_editor_main",
)

if isinstance(editor_value, dict) and "splits" in editor_value:
    new_splits = {item["index"] for item in editor_value["splits"]}
    new_splits.add(0)  # 第一頁一定是分割起點
    new_filenames = {item["index"]: item["filename"] for item in editor_value["splits"]}
    if 0 not in new_filenames:
        new_filenames[0] = st.session_state["filenames"].get(0, "文件_1")
    st.session_state["splits"] = new_splits
    st.session_state["filenames"] = new_filenames

# ----------------------------------------------------------------------------
# 分件結果預覽
# ----------------------------------------------------------------------------
splits_sorted = sorted(st.session_state["splits"])
groups = build_groups(splits_sorted, total_pages)

st.markdown('<div class="sb-section">分割結果預覽</div>', unsafe_allow_html=True)

if st.button("🔄 依目前分割重新套用自動命名規則（型號／編號）"):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for (start, end) in groups:
        title_text = st.session_state["auto_titles_raw"].get(start) if "auto_titles_raw" in st.session_state else None
        if not title_text:
            # 沒有原始標題快取時，退回用目前檔名去掉編號部分當標題
            title_text = re.split(r"_[A-Za-z0-9\-\/\.]{4,}$", st.session_state["filenames"].get(start, f"文件_{start+1}"))[0]
        st.session_state["filenames"][start] = guess_filename(doc, start, end, title_text)
    doc.close()
    st.rerun()

groups_with_names = []
_debug_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
for (start, end) in groups:
    fname = sanitize_filename(st.session_state["filenames"].get(start, f"文件_{start+1}"))
    groups_with_names.append(((start, end), fname))
    st.markdown(
        f"""
        <div class="sb-doc-card">
            <div class="fname">📄 {fname}.pdf</div>
            <div class="pages">第 {start+1} 頁 – 第 {end+1} 頁（共 {end-start+1} 頁）</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander(f"🔍 顯示第 {start+1}–{end+1} 頁辨識到的文字（除錯用）"):
        raw_text = normalize_text(get_full_text(_debug_doc, start, end))
        st.text(raw_text[:800] + ("..." if len(raw_text) > 800 else ""))
_debug_doc.close()

st.divider()

if st.button("✅ 確認分割並產生壓縮檔", type="primary"):
    with st.spinner("分割中，請稍候..."):
        zip_bytes = split_pdf_to_zip(pdf_bytes, groups_with_names)
    st.session_state["zip_bytes"] = zip_bytes
    st.success("分割完成！請點選下方按鈕下載壓縮檔（瀏覽器安全限制，需手動點擊一次）。")

if st.session_state.get("zip_bytes"):
    st.download_button(
        "⬇️ 下載分割後的 PDF 壓縮檔 (zip)",
        data=st.session_state["zip_bytes"],
        file_name="split_documents.zip",
        mime="application/zip",
    )
