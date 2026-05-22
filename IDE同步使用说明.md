# IDE 同步使用说明

用于把本地 `D:\fwwb-new` 中的以下目录同步覆盖到腾讯网页 IDE：

- `agent_diy/`
- `agent_ppo/`
- `conf/`

不会同步其他目录。

## 1. IDE 端启动服务

把本地 `ide_sync_server.py` 复制到 IDE 项目里，例如：

```bash
/data/projects/legged_robot_competition_26/conf/ide_sync_server.py
```

在 IDE 终端运行：

```bash
cd /data/projects/legged_robot_competition_26
/bin/python3 conf/ide_sync_server.py
```

看到下面类似输出表示服务已启动：

```text
IDE sync server: http://0.0.0.0:8765
Root: /data/projects/legged_robot_competition_26
Fixed outside URL: https://tencentarena.com/p5/ide/11428/proxy/8765/
```

这个终端不要关闭。

浏览器可以测试：

```text
https://tencentarena.com/p5/ide/11428/proxy/8765/health
```

正常会返回：

```json
{
  "ok": true,
  "root": "/data/projects/legged_robot_competition_26"
}
```

注意：地址后面不要带中文句号 `。`。

## 2. 本地填写 Cookie

打开本地：

```text
D:\fwwb-new\local_sync_client.py
```

找到顶部：

```python
USER_PROXY_COOKIE = ""
```

填入浏览器里的 `kaiwu-token` 值：

```python
USER_PROXY_COOKIE = "你的 kaiwu-token Cookie Value"
```

也可以填完整 Cookie：

```python
USER_PROXY_COOKIE = "DXUSS=...; kaiwu-token=...; select_lang=zh"
```

## 3. 本地一键同步

在本地 PowerShell 运行：

```powershell
cd D:\fwwb-new
& C:\Python314\python.exe d:/fwwb-new/local_sync_client.py
```

成功时会看到：

```text
remote root: /data/projects/legged_robot_competition_26
sync dirs: agent_diy, agent_ppo, conf
files to overwrite: ...
sync complete
```

新版同步使用 GET 分块上传，并校验远端 `sha256`，不会只看上传日志。

## 4. 常见问题

### 访问 `/health` 返回 500 / ECONNREFUSED

IDE 端服务没启动，或终端被关闭。

重新运行：

```bash
/bin/python3 conf/ide_sync_server.py
```

### 返回 `TOKEN_NOT_VALID`

腾讯代理 Cookie 过期或填错。

重新从浏览器复制 `kaiwu-token`，更新 `USER_PROXY_COOKIE`。

### 同步成功但 IDE 标签页没变化

网页 VS Code 可能缓存已打开文件。

关闭该文件标签后重新打开，或刷新文件树。

### 文件还是没变

确认 IDE 端运行的是最新版 `ide_sync_server.py`，必须支持：

- `/write_begin`
- `/write_chunk`
- `/write_finish`

旧版服务端会导致上传日志看似成功，但文件不落盘。

