from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit.components.v1 as components

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
INDEX_FILE = FRONTEND_DIR / "index.html"

if not FRONTEND_DIR.is_dir():
    raise FileNotFoundError("找不到自訂元件資料夾：{}".format(FRONTEND_DIR))
if not INDEX_FILE.is_file():
    raise FileNotFoundError("找不到自訂元件入口檔：{}".format(INDEX_FILE))

_COMPONENT = components.declare_component(
    "split_editor",
    path=str(FRONTEND_DIR),
)


def split_editor(
    pages: List[Dict[str, Any]],
    revision: int = 0,
    download_data_url: Optional[str] = None,
    download_filename: Optional[str] = None,
    key: Optional[str] = None,
) -> Any:
    """
    互動式 PDF 分割線編輯器（拖曳虛線調整分割點，行內編輯檔名，內建結果預覽、
    除錯文字檢視、確認送出，以及下載連結——整個主畫面都在這個自訂元件裡）。

    pages: [
        {
            "index": int,
            "thumb_data_url": str,
            "is_split": bool,
            "filename": str,          # 若為分割起點，該份文件的檔名（不含 .pdf）
            "text": str,              # 該頁辨識到的原始文字（除錯用，可截斷）
        }, ...
    ]

    download_data_url：若已產生壓縮檔，傳入 "data:application/zip;base64,...."，
    元件會顯示下載連結。沒有的話傳 None。

    `revision` 用來讓呼叫端（app.py）分辨「使用者外部重置」跟「一般 rerun」，
    元件前端本身不會讀取這個值，純粹是 Python 端的記帳用途。

    回傳： {"splits": [...], "action": "confirm" 或 None, "nonce": int}
    """
    default_splits = [
        {"index": p["index"], "filename": p["filename"]}
        for p in pages
        if p.get("is_split")
    ]
    return _COMPONENT(
        pages=pages,
        revision=int(revision),
        download_data_url=download_data_url,
        download_filename=download_filename or "split_documents.zip",
        key=key,
        default={"splits": default_splits, "action": None, "nonce": 0},
    )

