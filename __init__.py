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
    key: Optional[str] = None,
) -> Any:
    """
    互動式 PDF 分割線編輯器（拖曳虛線調整分割點，行內編輯檔名）。

    pages: [
        {
            "index": int,                 # 0-based 頁碼
            "thumb_data_url": str,        # "data:image/png;base64,...."
            "is_split": bool,             # 這一頁是否為目前的分割起點
            "filename": str,              # 若為分割起點，該份文件的檔名（不含 .pdf）
        }, ...
    ]

    `revision` 用來讓呼叫端（app.py）分辨「使用者外部重置」跟「一般 rerun」，
    元件前端本身不會讀取這個值，純粹是 Python 端的記帳用途。

    回傳： {"splits": [{"index": int, "filename": str}, ...]}
    （即目前每一份文件的起始頁與檔名，依 index 由小到大排序）
    """
    default_splits = [
        {"index": p["index"], "filename": p["filename"]}
        for p in pages
        if p.get("is_split")
    ]
    return _COMPONENT(
        pages=pages,
        revision=int(revision),
        key=key,
        default={"splits": default_splits},
    )
