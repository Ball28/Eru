#!/usr/bin/env python3
"""
Lightweight standalone local web chat interface for Eru LLM with real-time token streaming.
No external web dependencies required (uses built-in http.server and standard library).
Run:
    python scripts/web_chat.py --port 8000
"""

import argparse
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Ensure src is on path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import torch
from eru.generation.sampling import generate_stream
from eru.model.gpt import GPTModel, GPT_CONFIG_124M, GPT_CONFIG_SMALL
from eru.tokenizer.bpe import BPETokenizer
from eru.tokenizer.utils import text_to_token_ids
from eru.training.checkpoint import load_checkpoint
from eru.utils.config import load_config
from eru.utils.device import get_device

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Eru LLM &mdash; Interactive Web Chat</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background-color: #0f172a; color: #f8fafc; display: flex; height: 100vh; }
    #sidebar { width: 300px; background-color: #1e293b; padding: 24px; border-right: 1px solid #334155; display: flex; flex-direction: column; gap: 18px; }
    #sidebar h2 { font-size: 1.1rem; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.05em; }
    .param-group { display: flex; flex-direction: column; gap: 6px; }
    .param-group label { font-size: 0.85rem; color: #94a3b8; display: flex; justify-content: space-between; }
    .param-group input[type="range"] { accent-color: #38bdf8; }
    #chat-container { flex: 1; display: flex; flex-direction: column; }
    #header { padding: 18px 24px; border-bottom: 1px solid #334155; display: flex; align-items: center; justify-content: space-between; background: #1e293b; }
    #header h1 { font-size: 1.25rem; font-weight: 600; display: flex; align-items: center; gap: 8px; }
    #header .badge { font-size: 0.75rem; background: #0284c7; color: white; padding: 3px 8px; border-radius: 9999px; }
    #messages { flex: 1; padding: 24px; overflow-y: auto; display: flex; flex-direction: column; gap: 16px; }
    .msg { display: flex; gap: 12px; max-width: 80%; line-height: 1.5; }
    .msg.user { align-self: flex-end; flex-direction: row-reverse; }
    .msg.eru { align-self: flex-start; }
    .bubble { padding: 12px 18px; border-radius: 14px; font-size: 0.95rem; white-space: pre-wrap; }
    .msg.user .bubble { background-color: #2563eb; color: #ffffff; border-bottom-right-radius: 2px; }
    .msg.eru .bubble { background-color: #334155; color: #f1f5f9; border-bottom-left-radius: 2px; border: 1px solid #475569; }
    #input-area { padding: 16px 24px; background: #1e293b; border-top: 1px solid #334155; display: flex; gap: 12px; }
    #prompt-input { flex: 1; background: #0f172a; border: 1px solid #475569; color: white; padding: 12px 16px; border-radius: 10px; font-size: 0.95rem; outline: none; }
    #prompt-input:focus { border-color: #38bdf8; }
    #send-btn { background: #0284c7; color: white; border: none; padding: 12px 24px; border-radius: 10px; font-weight: 600; cursor: pointer; transition: background 0.2s; }
    #send-btn:hover { background: #0369a1; }
    #send-btn:disabled { background: #475569; cursor: not-allowed; }
  </style>
</head>
<body>
  <div id="sidebar">
    <h2>Eru Settings</h2>
    <div class="param-group">
      <label>Temperature <span id="temp-val">0.8</span></label>
      <input type="range" id="temp" min="0" max="2" step="0.05" value="0.8" oninput="document.getElementById('temp-val').innerText = this.value">
    </div>
    <div class="param-group">
      <label>Top-K <span id="topk-val">40</span></label>
      <input type="range" id="topk" min="0" max="100" step="1" value="40" oninput="document.getElementById('topk-val').innerText = this.value">
    </div>
    <div class="param-group">
      <label>Top-P (Nucleus) <span id="topp-val">0.9</span></label>
      <input type="range" id="topp" min="0.05" max="1.0" step="0.05" value="0.9" oninput="document.getElementById('topp-val').innerText = this.value">
    </div>
    <div class="param-group">
      <label>Repetition Penalty <span id="rep-val">1.1</span></label>
      <input type="range" id="rep" min="1.0" max="2.0" step="0.05" value="1.1" oninput="document.getElementById('rep-val').innerText = this.value">
    </div>
    <div class="param-group">
      <label>Max Tokens <span id="tokens-val">60</span></label>
      <input type="range" id="tokens" min="10" max="256" step="5" value="60" oninput="document.getElementById('tokens-val').innerText = this.value">
    </div>
  </div>
  <div id="chat-container">
    <div id="header">
      <h1>Eru LLM <span class="badge">KV-Cache Active</span></h1>
    </div>
    <div id="messages">
      <div class="msg eru"><div class="bubble">Hello! I am Eru, a causal language model built from scratch. Type a message below to start generating text!</div></div>
    </div>
    <div id="input-area">
      <input type="text" id="prompt-input" placeholder="Type a message or prompt..." onkeydown="if(event.key==='Enter') sendMessage()">
      <button id="send-btn" onclick="sendMessage()">Send</button>
    </div>
  </div>

  <script>
    async function sendMessage() {
      const input = document.getElementById("prompt-input");
      const btn = document.getElementById("send-btn");
      const text = input.value.trim();
      if (!text) return;

      appendMessage("user", text);
      input.value = "";
      input.disabled = true;
      btn.disabled = true;

      const eruMsgBubble = appendMessage("eru", "...");

      const payload = {
        prompt: text,
        max_tokens: parseInt(document.getElementById("tokens").value),
        temperature: parseFloat(document.getElementById("temp").value),
        top_k: parseInt(document.getElementById("topk").value),
        top_p: parseFloat(document.getElementById("topp").value),
        repetition_penalty: parseFloat(document.getElementById("rep").value)
      };

      try {
        const response = await fetch("/api/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        eruMsgBubble.innerText = "";

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value);
          const lines = chunk.split("\\n");
          for (const line of lines) {
            if (line.startsWith("data: ")) {
              const data = JSON.parse(line.substring(6));
              if (data.token) {
                eruMsgBubble.innerText += data.token;
                document.getElementById("messages").scrollTop = document.getElementById("messages").scrollHeight;
              }
            }
          }
        }
      } catch (err) {
        eruMsgBubble.innerText = "Error: " + err.message;
      } finally {
        input.disabled = false;
        btn.disabled = false;
        input.focus();
      }
    }

    function appendMessage(sender, text) {
      const msgs = document.getElementById("messages");
      const msgDiv = document.createElement("div");
      msgDiv.className = "msg " + sender;
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.innerText = text;
      msgDiv.appendChild(bubble);
      msgs.appendChild(msgDiv);
      msgs.scrollTop = msgs.scrollHeight;
      return bubble;
    }
  </script>
</body>
</html>
"""


class EruWebChatHandler(BaseHTTPRequestHandler):
    model = None
    tokenizer = None
    device = None
    cfg = None

    def log_message(self, format, *args):
        # Quiet standard HTTP access logs
        pass

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        else:
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/generate":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)

            prompt = data.get("prompt", "")
            max_tokens = int(data.get("max_tokens", 50))
            temperature = float(data.get("temperature", 0.8))
            top_k = int(data.get("top_k", 40))
            top_p = float(data.get("top_p", 0.9))
            repetition_penalty = float(data.get("repetition_penalty", 1.1))

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            input_ids = text_to_token_ids(prompt, self.tokenizer, device=self.device)

            for _, token_str in generate_stream(
                model=self.model,
                idx=input_ids,
                max_new_tokens=max_tokens,
                context_size=self.cfg["context_length"],
                tokenizer=self.tokenizer,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                use_cache=True,
            ):
                if token_str:
                    payload = json.dumps({"token": token_str})
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()

            self.wfile.write(b"data: {\"done\": true}\n\n")
            self.wfile.flush()
        else:
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()


def run_server(port: int = 8000, weights: Optional[str] = None, small: bool = True):
    device = get_device()
    cfg = GPT_CONFIG_SMALL if small else GPT_CONFIG_124M

    print(f"Loading Eru model ({'small' if small else '124M'}) on {device}...")
    model = GPTModel(cfg).to(device)
    if weights and Path(weights).exists():
        load_checkpoint(weights, model=model, device=device)
    model.eval()

    tokenizer = BPETokenizer()

    EruWebChatHandler.model = model
    EruWebChatHandler.tokenizer = tokenizer
    EruWebChatHandler.device = device
    EruWebChatHandler.cfg = cfg

    server = ThreadingHTTPServer(("0.0.0.0", port), EruWebChatHandler)
    print(f"\n=======================================================")
    print(f"  Eru Web Chat running at http://localhost:{port}")
    print(f"  Press Ctrl+C to stop the server.")
    print(f"=======================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eru Web Chat server.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    parser.add_argument("--weights", default=None, help="Weights checkpoint path.")
    parser.add_argument("--small", action="store_true", default=True, help="Use small fast CPU configuration.")
    args = parser.parse_args()

    run_server(port=args.port, weights=args.weights, small=args.small)
