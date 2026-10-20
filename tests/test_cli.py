"""CLI 与 HTTP 服务：端到端跑通（不联网、不调 API）。"""

from __future__ import annotations

import json
import urllib.request

import pytest

from jobfit.cli import main
from jobfit.server import serve


def test_cli_analyze_md(tmp_path, repo_root, sample_jd):
    """CLI 能跑出 Markdown 报告。"""
    out = tmp_path / "report.md"
    code = main(
        [
            "analyze",
            "--jd", str(repo_root / "data" / "jds" / "authored_nlp_classification.json"),
            "--resume", str(repo_root / "data" / "resumes" / "简历202607-AI工程.md"),
            "--format", "md",
            "--out", str(out),
            "--llm", "none",
            "--no-mcp",
        ]
    )
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert "求职匹配分析报告" in text
    assert "JD 技能要求抽取" in text
    assert "预测面试题" in text


def test_cli_analyze_json(tmp_path, repo_root):
    """CLI 能跑出 JSON 报告（机器可读）。"""
    out = tmp_path / "report.json"
    code = main(
        [
            "analyze",
            "--jd", str(repo_root / "data" / "jds" / "authored_nlp_classification.json"),
            "--resume", str(repo_root / "data" / "resumes" / "简历202607-AI工程.md"),
            "--format", "json",
            "--out", str(out),
            "--llm", "none",
            "--no-mcp",
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    for key in ("requirements", "matches", "gaps", "rewrites", "questions", "trace"):
        assert key in payload


def test_cli_missing_file_returns_error_code(tmp_path):
    code = main(
        [
            "analyze",
            "--jd", str(tmp_path / "nope.txt"),
            "--resume", str(tmp_path / "nope.md"),
            "--llm", "none",
            "--no-mcp",
        ]
    )
    assert code == 2


def test_cli_doctor_runs():
    assert main(["doctor"]) in (0, 1)


# ------------------------------------------------------------------ HTTP

@pytest.fixture
def http_server():
    server = serve(host="127.0.0.1", port=0, llm=None, block=False)
    yield server
    server.shutdown()
    server.server_close()


def test_healthz(http_server):
    host, port = http_server.server_address[0], http_server.server_address[1]
    with urllib.request.urlopen(f"http://{host}:{port}/healthz", timeout=20) as resp:
        payload = json.loads(resp.read().decode())
    assert payload["status"] == "ok"
    assert "siblings" in payload
    assert payload["uptime_seconds"] >= 0


def test_analyze_endpoint(http_server, repo_root):
    host, port = http_server.server_address[0], http_server.server_address[1]
    body = json.dumps(
        {
            "jd": str(repo_root / "data" / "jds" / "authored_nlp_classification.json"),
            "resume": str(repo_root / "data" / "resumes" / "简历202607-AI工程.md"),
            "format": "json",
            "use_mcp": False,
        }
    ).encode()
    request = urllib.request.Request(
        f"http://{host}:{port}/analyze",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as resp:
        payload = json.loads(resp.read().decode())
    assert payload["requirements"]
    assert payload["matches"]


def test_analyze_endpoint_validates_body(http_server):
    host, port = http_server.server_address[0], http_server.server_address[1]
    request = urllib.request.Request(
        f"http://{host}:{port}/analyze",
        data=b"{}",
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(request, timeout=20)
    assert excinfo.value.code == 400
