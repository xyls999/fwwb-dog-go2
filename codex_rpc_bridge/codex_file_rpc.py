# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Lightweight file RPC bridge for Tencent Arena IDE / code-server containers.

功能：
- 普通 token 负责工作区内读写、列目录、查看树、删除空目录/文件。
- Admin token 负责全局读写、执行命令、停止服务。
- 覆盖/追加/删除前自动备份。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


VERSION = "0.7.1"
RUNTIME_DIR_NAME = "codex_rpc_bridge_runtime"
BACKUP_DIR_NAME = "backups"
MAX_BODY_BYTES = 64 * 1024 * 1024
MAX_TEXT_PREVIEW = 200000
DENY_DIR_NAMES = {RUNTIME_DIR_NAME}
DENY_FILE_NAMES = {"token", "admin_token", "codex_file_rpc.log"}


def _json(data: Any, status: int = 200) -> tuple[int, bytes]:
    return status, json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")


def _b64url_decode(data: str) -> bytes:
    pad = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + pad)


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _is_subpath(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


@dataclass
class AuthResult:
    role: str | None

    @property
    def ok(self) -> bool:
        return self.role is not None

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


class BridgeState:
    def __init__(self, root: Path, runtime_dir: Path, token: str, admin_token: str) -> None:
        self.root = root.resolve()
        self.runtime_dir = runtime_dir.resolve()
        self.token = token
        self.admin_token = admin_token
        self.backup_root = self.runtime_dir / BACKUP_DIR_NAME
        self.backup_root.mkdir(parents=True, exist_ok=True)
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def stopping(self) -> bool:
        return self._stop_event.is_set()


class BridgeRequestHandler(BaseHTTPRequestHandler):
    server_version = f"CodexFileRPC/{VERSION}"

    @property
    def state(self) -> BridgeState:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Codex-Token, X-Codex-Admin-Token, Authorization")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def do_GET(self) -> None:
        self.dispatch("GET")

    def do_POST(self) -> None:
        self.dispatch("POST")

    def dispatch(self, method: str) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            auth = self._auth(query)
            if method == "GET" and parsed.path == "/api/health":
                self._send_json(
                    {
                        "ok": True,
                        "version": VERSION,
                        "root": str(self.state.root),
                        "auth": auth.ok,
                        "admin_auth": auth.is_admin,
                        "stopping": self.state.stopping,
                    }
                )
                return
            if method == "GET" and parsed.path == "/api/token-check":
                self._send_json({"ok": True, "role": auth.role})
                return
            if method == "GET" and parsed.path == "/api/list":
                self._send_json(self._list_dir(self.state.root, auth))
                return
            if method == "GET" and parsed.path.startswith("/api/list/"):
                self._send_json(self._list_dir(self._resolve_target(parsed.path[len("/api/list/"):]), auth))
                return
            if method == "GET" and parsed.path == "/api/tree":
                depth = self._query_int(query, "depth", 3)
                self._send_json(self._tree(self.state.root, depth, auth))
                return
            if method == "GET" and parsed.path.startswith("/api/tree/"):
                depth = self._query_int(query, "depth", 3)
                self._send_json(self._tree(self._resolve_target(parsed.path[len("/api/tree/"):]), depth, auth))
                return
            if method == "GET" and parsed.path.startswith("/api/stat/"):
                self._send_json(self._stat(self._resolve_target(parsed.path[len("/api/stat/"):]), auth))
                return
            if method == "GET" and parsed.path.startswith("/api/read/"):
                enc = query.get("encoding", ["utf-8"])[0]
                self._send_json(self._read_file(self._resolve_target(parsed.path[len("/api/read/"):]), auth, enc))
                return
            if method == "GET" and parsed.path.startswith("/api/write_text/"):
                text = query.get("text", [""])[0]
                expected_sha256 = query.get("expected_sha256", [""])[0] or None
                self._send_json(self._write_text(self._resolve_target(parsed.path[len("/api/write_text/"):]), auth, text, expected_sha256))
                return
            if method == "GET" and parsed.path.startswith("/api/write_b64/"):
                data = query.get("data", [""])[0]
                expected_sha256 = query.get("expected_sha256", [""])[0] or None
                self._send_json(self._write_b64(self._resolve_target(parsed.path[len("/api/write_b64/"):]), auth, data, expected_sha256))
                return
            if method == "GET" and parsed.path.startswith("/api/append_b64/"):
                data = query.get("data", [""])[0]
                self._send_json(self._append_b64(self._resolve_target(parsed.path[len("/api/append_b64/"):]), auth, data))
                return
            if method == "GET" and parsed.path.startswith("/api/mkdir_get/"):
                self._send_json(self._mkdir(self._resolve_target(parsed.path[len("/api/mkdir_get/"):]), auth))
                return
            if method == "GET" and parsed.path.startswith("/api/touch/"):
                self._send_json(self._touch(self._resolve_target(parsed.path[len("/api/touch/"):]), auth))
                return
            if method == "GET" and parsed.path.startswith("/api/delete_get/"):
                self._send_json(self._delete(self._resolve_target(parsed.path[len("/api/delete_get/"):]), auth))
                return
            if method == "GET" and parsed.path == "/api/exec_b64":
                self._send_json(self._exec_b64(query, auth))
                return
            if method == "POST" and parsed.path == "/api/stop":
                self._send_json(self._stop(auth))
                return
            self._send_json({"ok": False, "error": "not found"}, 404)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"ok": False, "error": str(exc)}, 400)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        code, data = _json(payload, status)
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _auth(self, query: dict[str, list[str]]) -> AuthResult:
        header_admin = self.headers.get("X-Codex-Admin-Token", "").strip()
        header_token = self.headers.get("X-Codex-Token", "").strip()
        bearer = self.headers.get("Authorization", "").strip()
        query_token = query.get("token", [""])[0].strip()
        query_admin = query.get("admin_token", [""])[0].strip()
        query_any = query.get("auth", [""])[0].strip()
        if header_admin == self.state.admin_token or query_admin == self.state.admin_token:
            return AuthResult("admin")
        if header_token == self.state.token or query_token == self.state.token or query_any == self.state.token:
            return AuthResult("normal")
        if bearer.startswith("Bearer "):
            candidate = bearer.removeprefix("Bearer ").strip()
            if candidate == self.state.admin_token:
                return AuthResult("admin")
            if candidate == self.state.token:
                return AuthResult("normal")
        raise PermissionError("unauthorized")

    def _ensure_role(self, auth: AuthResult, admin: bool = False) -> None:
        if admin and not auth.is_admin:
            raise PermissionError("admin token required")
        if not auth.ok:
            raise PermissionError("token required")

    def _query_int(self, query: dict[str, list[str]], key: str, default: int) -> int:
        raw = query.get(key, [str(default)])[0]
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"invalid integer for {key}: {raw}") from exc

    def _deny_runtime_access(self, path: Path) -> None:
        if _is_subpath(path, self.state.runtime_dir):
            raise PermissionError("runtime directory access denied")

    def _resolve_target(self, raw_path: str, *, for_write: bool = False, admin_required: bool = False) -> Path:
        path = unquote(raw_path or "").strip()
        if path.startswith("__abs__/"):
            abs_tail = path.removeprefix("__abs__/").lstrip("/")
            abs_path = Path("/" + abs_tail).resolve()
            return abs_path
        if not path:
            target = self.state.root
        else:
            target = (self.state.root / path).resolve()
        if not _is_subpath(target, self.state.root):
            raise PermissionError("path escapes root")
        self._deny_runtime_access(target)
        if for_write:
            parent = target.parent.resolve()
            if not _is_subpath(parent, self.state.root):
                raise PermissionError("parent escapes root")
            self._deny_runtime_access(parent)
        return target

    def _require_admin_for_abs(self, path: Path, auth: AuthResult) -> None:
        if path.is_absolute() and not auth.is_admin:
            raise PermissionError("absolute path requires admin token")

    def _path_access_check(self, path: Path, auth: AuthResult, *, write: bool = False) -> None:
        if write and not _is_subpath(path, self.state.root) and not auth.is_admin:
            raise PermissionError("absolute path requires admin token")
        if not auth.ok:
            raise PermissionError("token required")
        if _is_subpath(path, self.state.runtime_dir):
            raise PermissionError("runtime directory access denied")
        if write and _is_subpath(path.resolve() if path.exists() else path.parent.resolve(), self.state.runtime_dir):
            raise PermissionError("runtime directory access denied")

    def _backup_path(self, path: Path) -> Path:
        rel = path.as_posix().lstrip("/")
        stamp = _now_stamp()
        dest = self.state.backup_root / stamp / f"{rel}.bak"
        dest.parent.mkdir(parents=True, exist_ok=True)
        return dest

    def _backup_existing(self, path: Path) -> None:
        if not path.exists():
            return
        if path.is_dir():
            return
        backup = self._backup_path(path)
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)

    def _list_dir(self, path: Path, auth: AuthResult) -> dict[str, Any]:
        self._path_access_check(path, auth)
        if not path.exists():
            raise FileNotFoundError(str(path))
        if not path.is_dir():
            raise NotADirectoryError(str(path))
        items = []
        for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if child.name in DENY_FILE_NAMES and _is_subpath(child, self.state.runtime_dir):
                continue
            if child.name in DENY_DIR_NAMES:
                continue
            stat = child.stat()
            items.append(
                {
                    "path": self._render_path(child),
                    "name": child.name,
                    "type": "dir" if child.is_dir() else "file",
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                }
            )
        return {"ok": True, "path": self._render_path(path), "items": items}

    def _tree(self, path: Path, depth: int, auth: AuthResult) -> dict[str, Any]:
        self._path_access_check(path, auth)
        depth = max(0, min(depth, 8))
        if not path.exists():
            raise FileNotFoundError(str(path))

        def build(node: Path, level: int) -> dict[str, Any]:
            stat = node.stat()
            item = {
                "path": self._render_path(node),
                "name": node.name or self._render_path(node),
                "type": "dir" if node.is_dir() else "file",
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            }
            if node.is_dir() and level < depth:
                children = []
                for child in sorted(node.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                    if child.name in DENY_DIR_NAMES:
                        continue
                    if _is_subpath(child, self.state.runtime_dir):
                        continue
                    children.append(build(child, level + 1))
                item["children"] = children
            return item

        return {"ok": True, "tree": build(path, 0)}

    def _render_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.state.root).as_posix()
        except ValueError:
            return f"__abs__/{path.as_posix().lstrip('/')}"

    def _stat(self, path: Path, auth: AuthResult) -> dict[str, Any]:
        self._path_access_check(path, auth)
        if not path.exists():
            raise FileNotFoundError(str(path))
        stat = path.stat()
        return {
            "ok": True,
            "path": self._render_path(path),
            "type": "dir" if path.is_dir() else "file",
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "sha256": None if path.is_dir() else _sha256_bytes(path.read_bytes()),
        }

    def _read_file(self, path: Path, auth: AuthResult, encoding: str) -> dict[str, Any]:
        self._path_access_check(path, auth)
        if not path.exists():
            raise FileNotFoundError(str(path))
        if not path.is_file():
            raise IsADirectoryError(str(path))
        data = path.read_bytes()
        result = {
            "ok": True,
            "path": self._render_path(path),
            "size": len(data),
            "sha256": _sha256_bytes(data),
        }
        if encoding == "base64":
            result["encoding"] = "base64"
            result["data"] = _b64url_encode(data)
        else:
            result["encoding"] = "utf-8"
            result["text"] = data.decode("utf-8", errors="replace")
        return result

    def _check_expected_sha(self, path: Path, expected_sha256: str | None) -> None:
        if not expected_sha256:
            return
        actual = _sha256_bytes(path.read_bytes()) if path.exists() else None
        if actual != expected_sha256:
            raise RuntimeError(f"file changed since read: expected_sha256={expected_sha256} actual_sha256={actual}")

    def _write_bytes(self, path: Path, auth: AuthResult, payload: bytes, expected_sha256: str | None = None) -> dict[str, Any]:
        self._path_access_check(path, auth, write=True)
        if path.exists():
            self._check_expected_sha(path, expected_sha256)
            self._backup_existing(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
        tmp.write_bytes(payload)
        os.replace(tmp, path)
        return {
            "ok": True,
            "path": self._render_path(path),
            "size": len(payload),
            "sha256": _sha256_bytes(payload),
        }

    def _write_text(self, path: Path, auth: AuthResult, text: str, expected_sha256: str | None = None) -> dict[str, Any]:
        return self._write_bytes(path, auth, text.encode("utf-8"), expected_sha256)

    def _write_b64(self, path: Path, auth: AuthResult, data: str, expected_sha256: str | None = None) -> dict[str, Any]:
        return self._write_bytes(path, auth, _b64url_decode(data), expected_sha256)

    def _append_b64(self, path: Path, auth: AuthResult, data: str) -> dict[str, Any]:
        self._path_access_check(path, auth, write=True)
        payload = _b64url_decode(data)
        if path.exists():
            self._backup_existing(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as fh:
            fh.write(payload)
        return {"ok": True, "path": self._render_path(path), "size": path.stat().st_size, "sha256": _sha256_bytes(path.read_bytes())}

    def _mkdir(self, path: Path, auth: AuthResult) -> dict[str, Any]:
        self._path_access_check(path, auth, write=True)
        path.mkdir(parents=True, exist_ok=True)
        return {"ok": True, "path": self._render_path(path)}

    def _touch(self, path: Path, auth: AuthResult) -> dict[str, Any]:
        self._path_access_check(path, auth, write=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        return {"ok": True, "path": self._render_path(path)}

    def _delete(self, path: Path, auth: AuthResult) -> dict[str, Any]:
        self._path_access_check(path, auth, write=True)
        if path.is_dir():
            if any(path.iterdir()):
                raise ValueError("directory not empty")
            path.rmdir()
            return {"ok": True, "path": self._render_path(path), "deleted": "dir"}
        if path.exists():
            self._backup_existing(path)
            path.unlink()
            return {"ok": True, "path": self._render_path(path), "deleted": "file"}
        raise FileNotFoundError(str(path))

    def _exec_b64(self, query: dict[str, list[str]], auth: AuthResult) -> dict[str, Any]:
        self._ensure_role(auth, admin=True)
        cwd = query.get("cwd", ["."])[0]
        timeout = self._query_int(query, "timeout", 30)
        timeout = max(1, min(timeout, 3600))
        cmd = _b64url_decode(query.get("cmd", [""])[0]).decode("utf-8", errors="replace")
        cwd_path = self._resolve_target(cwd)
        if not cwd_path.exists() or not cwd_path.is_dir():
            raise NotADirectoryError(str(cwd_path))
        result = subprocess.run(
            cmd,
            cwd=str(cwd_path),
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "ok": True,
            "cwd": self._render_path(cwd_path),
            "cmd": cmd,
            "timeout": timeout,
            "returncode": result.returncode,
            "stdout": result.stdout[-MAX_TEXT_PREVIEW:],
            "stderr": result.stderr[-MAX_TEXT_PREVIEW:],
            "role": auth.role,
        }

    def _stop(self, auth: AuthResult) -> dict[str, Any]:
        self._ensure_role(auth, admin=True)
        self.state.stop()
        threading.Thread(target=self.server.shutdown, daemon=True).start()  # type: ignore[attr-defined]
        return {"ok": True, "stopping": True}


def _load_token(path: Path) -> str:
    if path.exists():
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value
    value = secrets.token_urlsafe(32)
    path.write_text(value + "\n", encoding="utf-8")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Codex RPC bridge for file operations in Tencent Arena IDE.")
    parser.add_argument("--host", default=os.environ.get("CODEX_RPC_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("CODEX_RPC_PORT", "8765")))
    parser.add_argument("--root", default=os.environ.get("CODEX_RPC_ROOT", os.getcwd()))
    parser.add_argument("--runtime-dir", default=os.environ.get("CODEX_RPC_RUNTIME_DIR", str(Path(os.getcwd()) / RUNTIME_DIR_NAME)))
    parser.add_argument("--token", default=os.environ.get("CODEX_RPC_TOKEN", ""))
    parser.add_argument("--admin-token", default=os.environ.get("CODEX_RPC_ADMIN_TOKEN", ""))
    args = parser.parse_args()

    root = Path(args.root).resolve()
    runtime_dir = Path(args.runtime_dir).resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)

    token = args.token.strip() or _load_token(runtime_dir / "token")
    admin_token = args.admin_token.strip() or _load_token(runtime_dir / "admin_token")

    server = ThreadingHTTPServer((args.host, args.port), BridgeRequestHandler)
    server.state = BridgeState(root=root, runtime_dir=runtime_dir, token=token, admin_token=admin_token)  # type: ignore[attr-defined]

    print(f"Codex RPC bridge listening on http://{args.host}:{args.port}")
    print(f"Root: {root}")
    print(f"Runtime: {runtime_dir}")
    print(f"Health: http://127.0.0.1:{args.port}/api/health")
    print(f"Token file: {runtime_dir / 'token'}")
    print(f"Admin token file: {runtime_dir / 'admin_token'}")

    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
