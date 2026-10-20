"""抓取公开真实 JD 文本 → data/jds/（只抓「文本」，标注另由人工/LLM 校对后写入）。

来源：腾讯招聘官网公开接口（careers.tencent.com），返回 JSON，含岗位正文与岗位链接。
为什么要脚本留痕：评测集里的「真实 JD」必须能复现——链接、抓取日期、原始正文都入库，
任何人重跑本脚本都能核对是不是同一份文本。

用法：
    python scripts/fetch_jds.py --out data/jds/raw --limit 8
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import date, datetime
from pathlib import Path

API_QUERY = "https://careers.tencent.com/tencentcareer/api/post/Query"
API_DETAIL = "https://careers.tencent.com/tencentcareer/api/post/ByPostId"

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _opener() -> urllib.request.OpenerDirector:
    """优先走本机代理（境外/部分站点需要），失败则直连。"""
    handler = urllib.request.ProxyHandler({"http": "http://127.0.0.1:7897", "https": "http://127.0.0.1:7897"})
    return urllib.request.build_opener(handler)


def _get_json(url: str, opener: urllib.request.OpenerDirector) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with opener.open(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def search(keyword: str, page_size: int = 20, opener=None) -> list[dict]:
    opener = opener or _opener()
    query = urllib.parse.urlencode(
        {
            "timestamp": int(time.time() * 1000),
            "keyword": keyword,
            "pageIndex": 1,
            "pageSize": page_size,
            "language": "zh-cn",
            "area": "cn",
        }
    )
    data = _get_json(f"{API_QUERY}?{query}", opener)
    return ((data.get("Data") or {}).get("Posts") or [])


def detail(post_id: str, opener=None) -> dict:
    opener = opener or _opener()
    query = urllib.parse.urlencode(
        {"timestamp": int(time.time() * 1000), "postId": post_id, "language": "zh-cn"}
    )
    data = _get_json(f"{API_DETAIL}?{query}", opener)
    return data.get("Data") or {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="抓取公开真实 JD 文本（只抓文本，不做标注）")
    parser.add_argument("--out", default="data/jds/raw", help="输出目录（相对仓库根）")
    parser.add_argument("--limit", type=int, default=8, help="抓取份数")
    parser.add_argument(
        "--keywords",
        default="AI应用开发,AI Infra,大模型算法,NLP算法,Java后端,模型评估,大模型评测,算法工程化",
        help="检索关键词，逗号分隔",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    opener = _opener()

    seen: set[str] = set()
    records: list[dict] = []
    for keyword in args.keywords.split(","):
        keyword = keyword.strip()
        if not keyword:
            continue
        try:
            posts = search(keyword, opener=opener)
        except Exception as exc:  # noqa: BLE001 —— 抓取失败要看得见，不能静默跳过
            print(f"[warn] 关键词 {keyword!r} 检索失败：{type(exc).__name__}: {exc}")
            continue
        for post in posts:
            pid = str(post.get("PostId") or "")
            if not pid or pid in seen:
                continue
            seen.add(pid)
            try:
                det = detail(pid, opener=opener)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] 岗位 {pid} 详情抓取失败：{type(exc).__name__}: {exc}")
                continue
            text = "\n".join(
                part for part in (det.get("Responsibility") or "", det.get("Requirement") or "") if part
            ).strip()
            if len(text) < 120:
                continue
            records.append(
                {
                    "post_id": pid,
                    "title": det.get("RecruitPostName") or "",
                    "company": "腾讯",
                    "business_group": det.get("BGName") or "",
                    "category": det.get("CategoryName") or "",
                    "location": det.get("LocationName") or "",
                    "source_url": det.get("PostURL") or f"http://careers.tencent.com/jobdesc.html?postId={pid}",
                    "source_site": "腾讯招聘官网",
                    "source_last_update": det.get("LastUpdateTime") or "",
                    "fetched_at": date.today().isoformat(),
                    "raw_text": text,
                }
            )
            print(f"[ok] {len(records):02d} {det.get('RecruitPostName')} ({det.get('CategoryName')})")

    payload = {
        "source_site": "腾讯招聘官网",
        "source_api": API_DETAIL,
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "count": len(records),
        "records": records,
    }
    target = out_dir / "tencent_raw.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {target.relative_to(Path.cwd()) if target.is_relative_to(Path.cwd()) else target}"
          f"（{len(records)} 份）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
