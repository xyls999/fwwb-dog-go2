# Codex RPC Bridge 使用说明

这个目录提供一个轻量 HTTP RPC 服务，用于让外部 AI agent 操作腾讯开悟 / code-server 开发容器中的代码。

适用场景：

- 容器不能访问公网。
- 容器不能正常上传本地文件。
- 外部 AI 无法直接 SSH 进入容器。
- 但浏览器里的 Tencent Arena IDE 已经连接到容器，并且可以通过 code-server 的端口代理访问容器端口。

目录名使用 `codex_rpc_bridge`，不是点开头目录，方便手动复制。运行时 token、日志和备份不要提交到仓库。

## 文件说明

- `codex_file_rpc.py`：RPC 服务脚本。
- `AI_USAGE.md`：本说明文档。
- 运行时会在目标环境中生成：
  - `token`
  - `admin_token`
  - `codex_file_rpc.log`
  - `backups/`

建议把运行时文件放在：

```text
codex_rpc_bridge_runtime/
```

不要把 token、admin_token、日志、备份提交到仓库。

## 在腾讯容器中启动

在容器工作区根目录运行：

```bash
cd /data/projects/legged_robot_competition_26 || exit 1
mkdir -p codex_rpc_bridge_runtime
```

生成普通 token 和 admin token：

```bash
python3 - <<'PY'
from pathlib import Path
import secrets

for name in ("token", "admin_token"):
    p = Path("codex_rpc_bridge_runtime") / name
    if not p.exists() or not p.read_text().strip():
        p.write_text(secrets.token_urlsafe(32) + "\n")
    print(name + "=" + p.read_text().strip())
PY
```

启动服务：

```bash
export CODEX_RPC_TOKEN="$(cat codex_rpc_bridge_runtime/token)"
export CODEX_RPC_ADMIN_TOKEN="$(cat codex_rpc_bridge_runtime/admin_token)"

nohup python3 codex_rpc_bridge/codex_file_rpc.py \
  --host 0.0.0.0 \
  --port 8765 \
  --root /data/projects/legged_robot_competition_26 \
  --token "$CODEX_RPC_TOKEN" \
  --admin-token "$CODEX_RPC_ADMIN_TOKEN" \
  > codex_rpc_bridge_runtime/codex_file_rpc.log 2>&1 &

sleep 1
cat codex_rpc_bridge_runtime/codex_file_rpc.log
curl -s http://127.0.0.1:8765/api/health
echo
```

如果成功，会看到类似：

```json
{
  "ok": true,
  "version": "0.7.1",
  "root": "/data/projects/legged_robot_competition_26",
  "auth": true,
  "admin_auth": true
}
```

## 外部访问地址

在腾讯开悟 code-server 代理下，外部访问地址通常是：

```text
https://tencentarena.com/p5/ide/11428/proxy/8765
```

其中 `11428` 可能会随 IDE 会话变化。如果访问不通，应查看当前 IDE iframe 地址，使用其中的 `/p5/ide/<id>/proxy/8765`。

## 认证方式

普通 token：

```text
X-Codex-Token: <codex_rpc_bridge_runtime/token 的内容>
```

Admin token：

```text
X-Codex-Admin-Token: <codex_rpc_bridge_runtime/admin_token 的内容>
```

## 权限模型

普通 token：

- 可以全局读取、列目录、查看 stat、tree。
- 可以在工作区内写入、追加、创建目录、touch、删除文件或空目录。
- 不能读取运行时 token 文件。
- 不能执行 shell 命令。
- 不能写出工作区。
- 不能通过 symlink 写出工作区。

Admin token：

- 可以全局读取。
- 可以全局写入、追加、删除。
- 可以执行 shell 命令。
- 可以停止 RPC 服务。

全局绝对路径写法：

```text
__abs__/etc/hostname
__abs__/tmp/file.txt
```

工作区路径写法：

```text
train_test.py
agent_diy/agent.py
```

## 稳定接口

```text
GET /api/health
GET /api/token-check
GET /api/list
GET /api/list/<dir>
GET /api/tree?depth=N
GET /api/tree/<dir>?depth=N
GET /api/stat/<path>
GET /api/read/<file>
GET /api/read/<file>?encoding=base64
GET /api/write_text/<file>?text=<urlencoded-text>
GET /api/write_b64/<file>?data=<base64url>
GET /api/append_b64/<file>?data=<base64url>
GET /api/mkdir_get/<dir>
GET /api/touch/<file>
GET /api/delete_get/<file-or-empty-dir>
GET /api/exec_b64?cwd=<dir>&timeout=<seconds>&cmd=<base64url-command>
POST /api/stop
```

## 给 AI Agent 的要求

外部 AI agent 应遵守：

1. 先 `read`，再修改。
2. 写入时必须带 `expected_sha256`。
3. 写入后必须再次 `read` 验证。
4. 写源码优先使用 `write_b64`。
5. 不要并发写同一个文件。
6. 普通开发只用 normal token。
7. 只有需要执行命令或全局写入时才使用 admin token。
8. 不要把 token 写入日志、代码、提交记录或公开消息。
