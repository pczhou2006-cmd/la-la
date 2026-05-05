"""Streaming API Demo with FastAPI — SSE + WebSocket examples."""

import asyncio
import json
import random
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, StreamingResponse

app = FastAPI(title="Streaming API Demo", version="1.0.0")


# ---------- SSE: 有限计数器 finitecounter ----------

@app.get("/stream/count")
async def sse_counter(n: int = 10):
    """Stream `n` countdown events, one per second."""

    async def generate():
        for i in range(1, n + 1):
            yield f"event: tick\ndata: {json.dumps({'step': i, 'total': n, 'percent': round(i / n * 100, 1)})}\n\n"
            await asyncio.sleep(1)
        yield f"event: done\ndata: {json.dumps({'step': n, 'message': 'Completed!'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })


# ---------- SSE: LLM-like token stream ----------

@app.get("/stream/chat")
async def sse_chat(q: str = "hello"):
    """Simulate an LLM replying token-by-token."""

    replies = {
        "hello": "Hi there! I am a FastAPI streaming demo. Nice to meet you!",
        "what": "I can demonstrate SSE streaming, sending responses token by token like a real LLM would.",
        "help": "Try these endpoints:\n\n/stream/count  — finite countdown\n/stream/chat?q=... — chat simulation\n/stream/file — progress simulation\n/ws/demo     — WebSocket echo",
    }
    text = replies.get(q.lower(), f"Thanks for asking about '{q}'. Streaming is the future of web交互!")

    async def generate():
        for token in text:
            yield f"data: {json.dumps({'token': token})}\n\n"
            await asyncio.sleep(random.uniform(0.02, 0.06))
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })


# ---------- SSE: 文件上传进度模拟 file upload progress simulation ----------

@app.get("/stream/upload")
async def sse_upload(filename: str = "large_file.zip", total_size_mb: float = 50.0):
    """Simulate a file upload with progress events."""
    total_bytes = int(total_size_mb * 1024 * 1024)
    chunk_size = total_bytes // 50

    async def generate():
        for i in range(50):
            sent = chunk_size * (i + 1)
            percent = round(sent / total_bytes * 100, 1)
            eta = round((50 - i - 1) * 0.1, 1)
            data = {
                "event": "progress",
                "filename": filename,
                "chunk": i + 1,
                "total_chunks": 50,
                "sent_bytes": sent,
                "total_bytes": total_bytes,
                "percent": percent,
                "eta_seconds": eta,
            }
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(0.1)
        data = {"event": "complete", "filename": filename, "message": f"{filename} uploaded successfully"}
        yield f"data: {json.dumps(data)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })


# ---------- WebSocket: echo ----------

@app.websocket("/ws/echo")
async def ws_echo(websocket: WebSocket):
    """WebSocket echo — send anything, get it back immediately."""
    await websocket.accept()
    try:
        while True:
            text = await websocket.receive_text()
            await websocket.send_json({"echo": text, "length": len(text)})
    except Exception:
        await websocket.close()


# ---------- WebSocket: chat room simulation ----------

@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    """WebSocket chat with simulated streaming reply."""
    await websocket.accept()
    await websocket.send_json({"type": "system", "message": "Connected! Send a message and get a streamed reply."})

    replies = {
        "hello": "Hello! Welcome to the streaming chat demo.",
        "help": "I can stream replies token by token. Try sending any message!",
    }

    try:
        while True:
            user_msg = await websocket.receive_text()
            # Stream response token by token
            response_text = replies.get(user_msg.lower(), f"You said: '{user_msg}'. Interesting! Let me think about that...")
            for token in response_text:
                await websocket.send_json({"type": "token", "value": token, "source": "assistant"})
                await asyncio.sleep(random.uniform(0.02, 0.05))
            await websocket.send_json({"type": "done", "source": "assistant"})
    except Exception:
        await websocket.close()


# ---------- HTML demo page ----------

@app.get("/", response_class=HTMLResponse)
async def demo_page():
    return HTMLResponse(content=_demo_html())


def _demo_html():
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FastAPI Streaming Demo</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, sans-serif; background: #0f0f1a; color: #e0e0e0; min-height: 100vh; }
  .header { text-align: center; padding: 30px 20px 10px; }
  .header h1 { font-size: 28px; background: linear-gradient(90deg, #00d2ff, #7b2ff7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .header p { color: #888; margin-top: 6px; font-size: 14px; }
  .tabs { display: flex; justify-content: center; gap: 8px; padding: 16px; }
  .tab-btn { padding: 8px 20px; border: 1px solid #333; background: #1a1a2e; color: #aaa; border-radius: 20px; cursor: pointer; font-size: 14px; transition: all .2s; }
  .tab-btn.active { background: #7b2ff7; color: #fff; border-color: #7b2ff7; }
  .panel { display: none; max-width: 600px; margin: 0 auto; padding: 0 20px 20px; }
  .panel.active { display: block; }
  .card { background: #1a1a2e; border: 1px solid #2a2a4a; border-radius: 12px; padding: 20px; margin-bottom: 16px; }
  .card h3 { font-size: 16px; margin-bottom: 12px; color: #00d2ff; }
  label { display: block; font-size: 13px; color: #888; margin-bottom: 4px; }
  input { width: 100%; padding: 10px 14px; background: #0f0f1a; border: 1px solid #333; border-radius: 8px; color: #e0e0e0; font-size: 14px; margin-bottom: 10px; }
  input:focus { outline: none; border-color: #7b2ff7; }
  button { padding: 10px 24px; background: #7b2ff7; color: #fff; border: none; border-radius: 8px; cursor: pointer; font-size: 14px; }
  button:hover { background: #9b4fff; }
  button:disabled { opacity: .5; cursor: not-allowed; }
  pre { background: #0f0f1a; border: 1px solid #2a2a4a; border-radius: 8px; padding: 14px; font-size: 13px; line-height: 1.6; max-height: 400px; overflow-y: auto; white-space: pre-wrap; word-break: break-all; }
  .progress-bar { height: 6px; background: #2a2a4a; border-radius: 3px; overflow: hidden; margin: 10px 0; }
  .progress-fill { height: 100%; background: linear-gradient(90deg, #00d2ff, #7b2ff7); transition: width .3s; border-radius: 3px; }
  .stats { display: flex; gap: 16px; margin-bottom: 10px; font-size: 13px; }
  .stat { background: #0f0f1a; padding: 6px 12px; border-radius: 6px; }
  .stat span { color: #00d2ff; }
  .status { font-size: 12px; padding: 6px 0; }
  .status.connected { color: #4ade80; }
  .status.disconnected { color: #f87171; }
  .ws-messages { max-height: 300px; overflow-y: auto; }
  .ws-msg { padding: 4px 0; font-size: 13px; }
  .ws-msg .sender { font-weight: bold; }
  .ws-msg.user .sender { color: #fbbf24; }
  .ws-msg.assistant .sender { color: #00d2ff; }
  .cursor { display: inline-block; width: 2px; height: 14px; background: #7b2ff7; animation: blink .6s infinite; vertical-align: middle; margin-left: 2px; }
  @keyframes blink { 50% { opacity: 0; } }
  .api-table { width: 100%; font-size: 13px; border-collapse: collapse; }
  .api-table th, .api-table td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #2a2a4a; }
  .api-table th { color: #888; font-weight: normal; }
  .api-table code { background: #0f0f1a; padding: 2px 6px; border-radius: 4px; color: #00d2ff; }
</style>
</head>
<body>
<div class="header">
  <h1>FastAPI Streaming API Demo</h1>
  <p>SSE + WebSocket 流式 API 示例</p>
</div>

<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('count')">SSE 计数器</button>
  <button class="tab-btn" onclick="switchTab('chat')">SSE 聊天</button>
  <button class="tab-btn" onclick="switchTab('upload')">SSE 上传</button>
  <button class="tab-btn" onclick="switchTab('ws')">WebSocket</button>
  <button class="tab-btn" onclick="switchTab('api')">API 文档</button>
</div>

<!-- SSE Counter -->
<div id="panel-count" class="panel active">
  <div class="card">
    <h3>⏱ 计数流</h3>
    <div class="stats">
      <div class="stat">步骤: <span id="count-step">0</span></div>
      <div class="stat">进度: <span id="count-pct">0%</span></div>
    </div>
    <div class="progress-bar"><div class="progress-fill" id="count-bar" style="width:0%"></div></div>
    <pre id="count-log">等待开始...</pre>
    <button id="count-btn" onclick="startCount()">开始流式计数</button>
  </div>
</div>

<!-- SSE Chat -->
<div id="panel-chat" class="panel">
  <div class="card">
    <h3>💬 模拟 LLM 聊天</h3>
    <input id="chat-input" placeholder="输入消息..." onkeydown="if(event.key==='Enter')sendChat()">
    <button id="chat-send-btn" onclick="sendChat()">发送</button>
    <pre id="chat-output">等待输入...</pre>
  </div>
</div>

<!-- SSE Upload -->
<div id="panel-upload" class="panel">
  <div class="card">
    <h3>📁 文件上传模拟</h3>
    <label>文件名</label>
    <input id="upload-filename" value="large_file.zip">
    <label>大小 (MB)</label>
    <input id="upload-size" value="50">
    <div class="stats">
      <div class="stat">已发送: <span id="upload-sent">0 B</span></div>
      <div class="stat">进度: <span id="upload-pct">0%</span></div>
    </div>
    <div class="progress-bar"><div class="progress-fill" id="upload-bar" style="width:0%"></div></div>
    <pre id="upload-log">等待开始...</pre>
    <button onclick="startUpload()">开始上传模拟</button>
  </div>
</div>

<!-- WebSocket -->
<div id="panel-ws" class="panel">
  <div class="card">
    <h3>🔌 WebSocket 聊天</h3>
    <div class="status disconnected" id="ws-status">● 未连接</div>
    <input id="ws-input" placeholder="输入消息..." onkeydown="if(event.key==='Enter')sendWS()" disabled>
    <button id="ws-btn" onclick="toggleWS()">连接</button>
    <div class="ws-messages" id="ws-messages"></div>
  </div>
</div>

<!-- API Reference -->
<div id="panel-api" class="panel">
  <div class="card">
    <h3>📖 API 端点</h3>
    <table class="api-table">
      <tr><th>方法</th><th>端点</th><th>说明</th></tr>
      <tr><td>GET</td><td>/stream/count?n=10</td><td>SSE 计数流</td></tr>
      <tr><td>GET</td><td>/stream/chat?q=hello</td><td>SSE 聊天流</td></tr>
      <tr><td>GET</td><td>/stream/upload?filename=x&total_size_mb=50</td><td>SSE 上传流</td></tr>
      <tr><td>WS</td><td>/ws/echo</td><td>WebSocket 回显</td></tr>
      <tr><td>WS</td><td>/ws/chat</td><td>WebSocket 聊天流</td></tr>
      <tr><td>GET</td><td>/health</td><td>健康检查</td></tr>
    </table>
  </div>
</div>

<script>
const BASE = window.location.origin;

function switchTab(name) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
  event.target.classList.add('active');
}

/* ---- SSE Counter ---- */
let countRunning = false;
async function startCount() {
  if (countRunning) return;
  countRunning = true;
  const btn = document.getElementById('count-btn');
  btn.disabled = true; btn.textContent = '运行中...';
  const total = 10;
  let step = 0;
  try {
    const resp = await fetch(`${BASE}/stream/count?n=${total}`);
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    document.getElementById('count-log').textContent = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\\n\\n');
      buf = lines.pop();
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const d = JSON.parse(line.slice(6));
          step = d.step;
          document.getElementById('count-step').textContent = step;
          document.getElementById('count-pct').textContent = d.percent + '%';
          document.getElementById('count-bar').style.width = d.percent + '%';
          document.getElementById('count-log').textContent += `[${d.step}/${total}] ${JSON.stringify(d)}\\n`;
          document.getElementById('count-log').scrollTop = 99999;
        }
      }
    }
  } finally {
    countRunning = false;
    btn.disabled = false; btn.textContent = '重新开始';
  }
}

/* ---- SSE Chat ---- */
async function sendChat() {
  const input = document.getElementById('chat-input');
  const output = document.getElementById('chat-output');
  const q = input.value.trim() || 'hello';
  output.textContent = '';
  try {
    const resp = await fetch(`${BASE}/stream/chat?q=${encodeURIComponent(q)}`);
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\\n\\n');
      buf = lines.pop();
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const d = JSON.parse(line.slice(6));
          if (d.token) output.textContent += d.token;
          else if (d.done) output.textContent += '<span style="color:#4ade80"> [完成]</span>';
        }
      }
    }
  } catch (e) {
    output.textContent = 'Error: ' + e.message;
  }
}

/* ---- SSE Upload ---- */
async function startUpload() {
  const filename = document.getElementById('upload-filename').value || 'file';
  const sizeMB = document.getElementById('upload-size').value || 50;
  const log = document.getElementById('upload-log');
  log.textContent = '';
  try {
    const resp = await fetch(`${BASE}/stream/upload?filename=${encodeURIComponent(filename)}&total_size_mb=${sizeMB}`);
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\\n\\n');
      buf = lines.pop();
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const d = JSON.parse(line.slice(6));
          document.getElementById('upload-sent').textContent = formatBytes(d.sent_bytes);
          document.getElementById('upload-pct').textContent = d.percent + '%';
          document.getElementById('upload-bar').style.width = d.percent + '%';
          log.textContent += `${d.percent}% chunk ${d.chunk}/50\\n`;
          log.scrollTop = 99999;
        }
      }
    }
  } catch (e) { log.textContent = 'Error: ' + e.message; }
}

function formatBytes(b) {
  if (b >= 1e9) return (b/1e9).toFixed(1) + ' GB';
  if (b >= 1e6) return (b/1e6).toFixed(1) + ' MB';
  if (b >= 1e3) return (b/1e3).toFixed(1) + ' KB';
  return b + ' B';
}

/* ---- WebSocket ---- */
let ws = null;
function toggleWS() {
  if (ws) { ws.close(); return; }
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws/chat`);
  const status = document.getElementById('ws-status');
  const input = document.getElementById('ws-input');
  const btn = document.getElementById('ws-btn');
  ws.onopen = () => { status.className = 'status connected'; status.textContent = '● 已连接'; input.disabled = false; btn.textContent = '断开'; };
  ws.onclose = () => { status.className = 'status disconnected'; status.textContent = '● 未连接'; input.disabled = true; btn.textContent = '连接'; };
  ws.onmessage = e => { appendWS(JSON.parse(e.data)); };
}

function sendWS() {
  if (!ws) return;
  const input = document.getElementById('ws-input');
  const msg = { type: 'user', content: input.value };
  appendWS({ ...msg, sender: 'You' });
  ws.send(input.value);
  input.value = '';
}

function appendWS(d) {
  const box = document.getElementById('ws-messages');
  let html = '';
  if (d.type === 'system') html = `<div class="ws-msg"><span style="color:#888">[系统] ${d.message || d.event}</span></div>`;
  else if (d.type === 'token') html = `<div class="ws-msg assistant"><span class="sender">AI:</span> <span id="typing">${d.value}</span><span class="cursor"></span></div>`;
  else if (d.type === 'done') html = '';
  else html = `<div class="ws-msg ${d.sender ? 'user' : ''}"><span class="sender">${d.sender || 'AI'}:</span> ${d.content || JSON.stringify(d)}</div>`;
  box.innerHTML += html;
  box.scrollTop = 99999;
}
</script>
</body>
</html>"""


# ---------- Health check ----------

@app.get("/health")
async def health():
    return {"status": "ok", "services": ["sse-count", "sse-chat", "sse-upload", "ws-echo", "ws-chat"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
