# Building a Local Server Accessible from a Mobile Phone

What Claude Code does is not very different from a standard local server architecture:

1. A server process runs on your laptop/desktop.
2. It listens on a network port (e.g. 3000, 8080).
3. Your phone connects over the same LAN (Wi-Fi) using the computer's IP address.
4. Communication happens over HTTP, WebSockets, or Server-Sent Events.
5. Optionally add authentication and TLS.

A minimal implementation is surprisingly small.

---

## Example: Node.js HTTP Server

```js
// server.js
const express = require("express");

const app = express();
app.use(express.json());

app.get("/", (req, res) => {
  res.send("Hello from local machine");
});

app.post("/message", (req, res) => {
  console.log("Phone sent:", req.body);
  res.json({
    reply: "received"
  });
});

app.listen(3000, "0.0.0.0", () => {
  console.log("Listening on port 3000");
});
```

Install and run:

```bash
npm install express
node server.js
```

The important part is:

```js
app.listen(3000, "0.0.0.0");
```

`0.0.0.0` means "accept connections from other devices", not just localhost.

---

## Find Your Computer's IP Address

### Mac/Linux

```bash
ifconfig
```

or

```bash
ip addr
```

### Windows

```cmd
ipconfig
```

You might get:

```text
192.168.1.42
```

Then from your phone:

```text
http://192.168.1.42:3000
```

Both devices must be on the same Wi-Fi network.

---

## Real-Time Communication (Claude Code Style)

For interactive agent behavior, WebSockets are usually used.

### Server

```js
const WebSocket = require("ws");

const wss = new WebSocket.Server({
  port: 3001
});

wss.on("connection", ws => {
  ws.send("Connected");

  ws.on("message", msg => {
    console.log(msg.toString());

    ws.send(
      JSON.stringify({
        type: "response",
        text: "hello phone"
      })
    );
  });
});
```

### Phone Browser

```js
const ws = new WebSocket(
  "ws://192.168.1.42:3001"
);

ws.onmessage = event => {
  console.log(event.data);
};

ws.send("hello");
```

This gives bidirectional communication with very low latency.

---

## AI Agent Running Locally

A common architecture looks like:

```text
Phone
  ↓
Web UI
  ↓
Local Server
  ↓
Agent Runtime
  ↓
LLM
  ↓
Tools
   ├── filesystem
   ├── terminal
   ├── git
   └── browser
```

Example:

```text
Phone
  ↓
React frontend
  ↓
Express/FastAPI backend
  ↓
Agent loop
  ↓
OpenAI API
```

When the phone sends:

```json
{
  "message": "fix my bug"
}
```

the server:

1. Sends the request to the model.
2. Executes any tool calls.
3. Streams results back over WebSocket.

---

## Streaming Responses Like Claude Code

Instead of waiting for the full answer:

```text
Phone ---> Server
        <--- "Hel"
        <--- "lo "
        <--- "wor"
        <--- "ld"
```

Use:

- WebSockets
- Server-Sent Events (SSE)

### SSE Example

```js
res.setHeader(
  "Content-Type",
  "text/event-stream"
);

res.write(`data: hello\n\n`);
```

The browser receives tokens as they arrive.

---

## Making It Accessible Outside Your Home

Three common approaches:

### 1. Tailscale (Easiest)

Install on both devices.

Then connect via:

```text
http://100.x.x.x:3000
```

No router configuration needed.

### 2. Cloudflare Tunnel

Expose the local server securely without opening ports.

### 3. Port Forwarding

Open a router port and forward to your machine.

Generally least recommended unless you understand the security implications.

---

## Python Version (FastAPI)

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}
```

Run:

```bash
uvicorn main:app \
  --host 0.0.0.0 \
  --port 3000
```

Again, `0.0.0.0` is what allows your phone to connect.

---

## Claude Code Clone Architecture

If your goal is specifically to build a "Claude Code clone" (local coding agent controlled from a phone), a typical stack would be:

### Backend

- FastAPI or Express
- WebSocket streaming
- OpenAI Responses API
- Terminal execution
- File editing tools
- Git integration

### Frontend

- React
- Mobile-responsive UI
- WebSocket client
- Streaming chat interface

### Architecture

```text
┌─────────────┐
│ Mobile App  │
└──────┬──────┘
       │ WebSocket
       ▼
┌─────────────┐
│ Local API   │
│ FastAPI     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Agent Loop  │
└──────┬──────┘
       │
       ├── File Tools
       ├── Terminal
       ├── Git
       └── OpenAI API
```

This is essentially the same high-level pattern used by most local coding agents.
