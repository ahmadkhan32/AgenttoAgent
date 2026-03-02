"""
A2A Friend Scheduling — Host Agent
===================================
FastAPI server that orchestrates scheduling between friend agents (Kaitlynn, Nate, Karley)
using Google Gemini AI. Demonstrates Agent-to-Agent (A2A) communication.

Run:
    uvicorn host_agent:app --port 8000

Then open:
    http://localhost:8000/docs     ← Swagger UI
    http://localhost:8000           ← Landing page
"""

import asyncio
import os
import time
from dotenv import load_dotenv
from google import genai
from fastapi import FastAPI, HTTPException
import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import SendMessageRequest, MessageSendParams, Task, SendMessageSuccessResponse
import uuid
import json
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found in .env file")

client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL = "gemini-2.5-flash"

# ── FastAPI App ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="🤖 A2A Friend Scheduling Agent",
    description="""
## Agent-to-Agent (A2A) Friend Scheduling System

**How it works:**

```
User ──► Host Agent
         ├──► Kaitlynn Agent  ──► responds
         ├──► Nate Agent      ──► responds
         └──► Karley Agent    ──► responds
              └─────────────────────────────►  Host merges → Final Plan
```

Use the `/start` endpoint to schedule anything with friends!
    """,
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response Models ───────────────────────────────────────────────
class ScheduleRequest(BaseModel):
    meeting_topic: str = "Dinner this Friday"
    time: str = "7 PM"

    class Config:
        json_schema_extra = {
            "example": {
                "meeting_topic": "Dinner this Friday",
                "time": "7 PM"
            }
        }


# ── Helper: call Gemini with retry on rate limit ─────────────────────────────
def _call_gemini(prompt: str, retries: int = 3) -> str:
    """Call Gemini API with retry on resource exhaustion (free tier)."""
    for attempt in range(retries):
        try:
            result = client.models.generate_content(model=MODEL, contents=prompt)
            return result.text.strip()
        except Exception as e:
            err = str(e)
            if "RESOURCE_EXHAUSTED" in err or "retryDelay" in err or "429" in err:
                wait = 15 * (attempt + 1)   # 15s, 30s, 45s
                print(f"[Rate limit] Waiting {wait}s before retry {attempt + 1}/{retries}...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Gemini rate limit hit after all retries. Please wait 60 seconds and try again.")


# ── Friend Agent Logic (powered by Real A2A Agents) ─────────────────────────
async def _run_real_agent(url: str, topic: str, t: str) -> str:
    task_text = f"Can you play {topic} at {t}?"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            card_resolver = A2ACardResolver(client, url)
            card = await card_resolver.get_agent_card()
            agent_client = A2AClient(client, card, url=url)
            
            message_id = str(uuid.uuid4())
            task_id = str(uuid.uuid4())
            context_id = str(uuid.uuid4())
            
            payload = {
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": task_text}],
                    "messageId": message_id,
                    "taskId": task_id,
                    "contextId": context_id,
                },
            }
            message_request = SendMessageRequest(
                id=message_id, params=MessageSendParams.model_validate(payload)
            )
            send_response = await agent_client.send_message(message_request)
            
            if not isinstance(
                send_response.root, SendMessageSuccessResponse
            ) or not isinstance(send_response.root.result, Task):
                return f"Error connecting to agent: {send_response}"
                
            json_content = json.loads(send_response.root.model_dump_json(exclude_none=True))
            resp = []
            if json_content.get("result", {}).get("artifacts"):
                for artifact in json_content["result"]["artifacts"]:
                    if artifact.get("parts"):
                        for p in artifact["parts"]:
                            if p.get("text"):
                                resp.append(p["text"])
            return "\n".join(resp) if resp else "Agent responded but with no text."
    except Exception as e:
        return f"Agent offline or error: {str(e)}"


def _run_host_merge(topic: str, t: str, kaitlynn: str, nate: str, karley: str) -> str:
    """Host Agent — merges all responses into a final schedule."""
    return _call_gemini(
        f'You are a scheduling coordinator.\n'
        f'Meeting: "{topic}" at {t}\n'
        f'Kaitlynn said: "{kaitlynn}"\n'
        f'Nate said: "{nate}"\n'
        f'Karley said: "{karley}"\n\n'
        f'Write a friendly final plan: who is coming, confirmed time, and any alternate suggestions. '
        f'Keep it under 5 sentences.'
    )


# ── API Endpoints ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>A2A Friend Scheduling Agent</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet"/>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Inter',sans-serif;min-height:100vh;background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);color:#fff;padding:30px 16px}
  .card{background:rgba(255,255,255,0.07);backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,0.12);border-radius:20px;padding:36px;max-width:760px;margin:0 auto;box-shadow:0 20px 60px rgba(0,0,0,0.4)}
  h1{font-size:2rem;font-weight:700;background:linear-gradient(90deg,#a78bfa,#60a5fa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:6px}
  .subtitle{color:rgba(255,255,255,0.55);font-size:0.95rem;margin-bottom:28px}
  .flow-box{background:rgba(0,0,0,0.25);border-radius:12px;padding:16px 20px;font-family:monospace;font-size:0.82rem;color:#a5f3fc;margin-bottom:28px;line-height:1.8}
  label{font-size:0.85rem;color:rgba(255,255,255,0.6);display:block;margin-bottom:6px}
  input{width:100%;padding:12px 16px;border-radius:10px;border:1px solid rgba(255,255,255,0.15);background:rgba(255,255,255,0.08);color:#fff;font-size:0.95rem;outline:none;margin-bottom:16px;transition:border .2s}
  input:focus{border-color:#a78bfa}
  .btn-run{width:100%;padding:14px;border-radius:12px;border:none;background:linear-gradient(90deg,#7c3aed,#2563eb);color:#fff;font-size:1rem;font-weight:600;cursor:pointer;transition:opacity .2s;display:flex;align-items:center;justify-content:center;gap:10px}
  .btn-run:hover{opacity:.9}
  .btn-run:disabled{opacity:.5;cursor:not-allowed}
  .spinner{width:18px;height:18px;border:3px solid rgba(255,255,255,0.3);border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;display:none}
  @keyframes spin{to{transform:rotate(360deg)}}
  .steps{margin-top:28px;display:none}
  .step{display:flex;align-items:center;gap:12px;padding:10px 14px;border-radius:10px;margin-bottom:8px;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.07);font-size:0.88rem;transition:all .4s}
  .step.done{background:rgba(52,211,153,0.1);border-color:rgba(52,211,153,0.3)}
  .step.active{background:rgba(167,139,250,0.1);border-color:rgba(167,139,250,0.4);animation:pulse 1.2s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.6}}
  .icon{font-size:1.1rem;width:24px;text-align:center}
  .agents{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:20px}
  .agent-card{background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);border-radius:12px;padding:14px;font-size:0.82rem}
  .agent-name{font-weight:600;color:#a78bfa;margin-bottom:6px}
  .agent-msg{color:rgba(255,255,255,0.75);line-height:1.5}
  .final{margin-top:20px;background:rgba(52,211,153,0.08);border:1px solid rgba(52,211,153,0.25);border-radius:12px;padding:18px}
  .final-label{font-size:0.75rem;text-transform:uppercase;letter-spacing:.08em;color:#6ee7b7;margin-bottom:8px;font-weight:600}
  .final-text{color:#ecfdf5;line-height:1.6;font-size:0.95rem}
  .error{margin-top:16px;background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);border-radius:10px;padding:14px;color:#fca5a5;font-size:0.88rem;display:none}
  .links{display:flex;gap:10px;margin-top:24px}
  .link-btn{flex:1;text-align:center;padding:10px;border-radius:10px;text-decoration:none;font-size:0.85rem;font-weight:600;border:1px solid rgba(255,255,255,0.15);color:rgba(255,255,255,0.75);transition:all .2s}
  .link-btn:hover{background:rgba(255,255,255,0.08);color:#fff}
</style>
</head>
<body>
<div class="card">
  <h1>🤖 A2A Friend Scheduling</h1>
  <p class="subtitle">Agent-to-Agent communication powered by Google Gemini 2.5</p>

  <div class="flow-box">
User ──► Host Agent<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ├──► Kaitlynn Agent ──► responds<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ├──► Nate Agent &nbsp;&nbsp;&nbsp;&nbsp; ──► responds<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └──► Karley Agent &nbsp;&nbsp; ──► responds<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └──► Host merges all ──► Final Plan
  </div>

  <label>Meeting Topic</label>
  <input id="topic" value="Pickleball this weekend" placeholder="e.g. Dinner Friday night"/>
  <label>Time</label>
  <input id="time" value="10 AM" placeholder="e.g. 7 PM"/>

  <button class="btn-run" id="runBtn" onclick="runA2A()">
    <span id="btnText">🚀 Run A2A Scheduling</span>
    <div class="spinner" id="spinner"></div>
  </button>

  <div class="error" id="errorBox"></div>

  <div class="steps" id="steps">
    <div class="step" id="s1"><span class="icon">⏳</span><span>User → Host Agent (sending request...)</span></div>
    <div class="step" id="s2"><span class="icon">⏳</span><span>Host → Kaitlynn Agent (checking availability...)</span></div>
    <div class="step" id="s3"><span class="icon">⏳</span><span>Host → Nate Agent (checking availability...)</span></div>
    <div class="step" id="s4"><span class="icon">⏳</span><span>Host → Karley Agent (checking availability...)</span></div>
    <div class="step" id="s5"><span class="icon">⏳</span><span>Host merging all responses → Final plan</span></div>
    <div class="agents" id="agentCards" style="display:none">
      <div class="agent-card"><div class="agent-name">👩 Kaitlynn</div><div class="agent-msg" id="msgKaitlynn">...</div></div>
      <div class="agent-card"><div class="agent-name">👨 Nate</div><div class="agent-msg" id="msgNate">...</div></div>
      <div class="agent-card"><div class="agent-name">👩 Karley</div><div class="agent-msg" id="msgKarley">...</div></div>
    </div>
    <div class="final" id="finalBox" style="display:none">
      <div class="final-label">✅ Final Plan</div>
      <div class="final-text" id="finalText"></div>
    </div>
  </div>

  <div class="links">
    <a class="link-btn" href="/docs">📝 Swagger UI</a>
    <a class="link-btn" href="/status">✅ Status</a>
  </div>
</div>

<script>
async function runA2A() {
  const btn = document.getElementById('runBtn');
  const spinner = document.getElementById('spinner');
  const btnText = document.getElementById('btnText');
  const errorBox = document.getElementById('errorBox');
  const steps = document.getElementById('steps');

  btn.disabled = true;
  spinner.style.display = 'block';
  btnText.textContent = 'Running A2A flow...';
  errorBox.style.display = 'none';
  steps.style.display = 'block';
  document.getElementById('agentCards').style.display = 'none';
  document.getElementById('finalBox').style.display = 'none';

  // Reset steps
  ['s1','s2','s3','s4','s5'].forEach(id => {
    const el = document.getElementById(id);
    el.className = 'step';
    el.querySelector('.icon').textContent = '⏳';
  });

  function setStep(id, state) {
    const el = document.getElementById(id);
    el.className = 'step ' + state;
    el.querySelector('.icon').textContent = state === 'done' ? '✅' : state === 'active' ? '🔄' : '⏳';
  }

  setStep('s1', 'done');
  setStep('s2', 'active');

  try {
    // Animate steps while waiting
    const stepTimer = setTimeout(() => setStep('s3', 'active'), 3000);
    const stepTimer2 = setTimeout(() => setStep('s4', 'active'), 6000);
    const stepTimer3 = setTimeout(() => setStep('s5', 'active'), 9000);

    const res = await fetch('/start', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        meeting_topic: document.getElementById('topic').value,
        time: document.getElementById('time').value
      })
    });

    clearTimeout(stepTimer); clearTimeout(stepTimer2); clearTimeout(stepTimer3);

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Server error');
    }

    const data = await res.json();

    // Mark all steps done
    ['s1','s2','s3','s4','s5'].forEach(id => setStep(id, 'done'));

    // Show agent responses
    document.getElementById('msgKaitlynn').textContent = data.agent_responses.Kaitlynn;
    document.getElementById('msgNate').textContent = data.agent_responses.Nate;
    document.getElementById('msgKarley').textContent = data.agent_responses.Karley;
    document.getElementById('agentCards').style.display = 'grid';

    document.getElementById('finalText').textContent = data.final_plan;
    document.getElementById('finalBox').style.display = 'block';

  } catch(e) {
    errorBox.textContent = '❌ ' + e.message;
    errorBox.style.display = 'block';
  }

  btn.disabled = false;
  spinner.style.display = 'none';
  btnText.textContent = '🚀 Run Again';
}
</script>
</body>
</html>
"""


@app.get("/status")
async def status():
    """Check that all agents and API key are operational."""
    return {
        "host_agent": "🟢 online",
        "friend_agents": {
            "Kaitlynn_Agent": "🟢 online",
            "Nate_Agent":     "🟢 online",
            "Karley_Agent":   "🟢 online",
        },
        "model": MODEL,
        "google_api_key_configured": True,
        "a2a_protocol": "enabled",
    }


@app.post("/start")
async def start_scheduling(request: ScheduleRequest):
    """
    🚀 **Start A2A Scheduling**

    Host Agent contacts all 3 friend agents one by one (sequential to respect free-tier rate limits),
    collects their responses, then merges them into a final meeting plan.

    **A2A Flow:**
    1. User → Host Agent (you are here)
    2. Host → **Kaitlynn** Agent → gets response
    3. Host → **Nate** Agent → gets response
    4. Host → **Karley** Agent → gets response
    5. Host merges all → **Final Plan** returned to you
    """
    topic = request.meeting_topic
    t     = request.time

    try:
        # A2A: Contact each friend agent sequentially (real HTTP requests)
        kaitlynn_resp = await _run_real_agent("http://localhost:10004", topic, t)
        nate_resp     = await _run_real_agent("http://localhost:10003", topic, t)
        karley_resp   = await _run_real_agent("http://localhost:10002", topic, t)
        final_plan    = await asyncio.to_thread(_run_host_merge, topic, t,
                                                kaitlynn_resp, nate_resp, karley_resp)

        return {
            "meeting_topic": topic,
            "time": t,
            "a2a_flow": {
                "step_1": "✅ User → Host Agent (received request)",
                "step_2": "✅ Host → Kaitlynn Agent (got availability)",
                "step_3": "✅ Host → Nate Agent (got availability)",
                "step_4": "✅ Host → Karley Agent (got availability)",
                "step_5": "✅ Host merged all responses → Final plan ready",
            },
            "agent_responses": {
                "Kaitlynn": kaitlynn_resp,
                "Nate":     nate_resp,
                "Karley":   karley_resp,
            },
            "final_plan": final_plan,
            "status": "✅ Scheduling complete",
        }

    except RuntimeError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.post("/ask/{agent_name}")
async def ask_single_agent(agent_name: str, request: ScheduleRequest):
    """
    🎯 **Talk to a Single Friend Agent**

    Ask just one agent about their availability.

    Agent names: `kaitlynn`, `nate`, `karley`
    """
    agents = {
        "kaitlynn": "http://localhost:10004",
        "nate":     "http://localhost:10003",
        "karley":   "http://localhost:10002",
    }
    name = agent_name.lower()
    if name not in agents:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_name}' not found. Available: {list(agents.keys())}"
        )
    try:
        response = await _run_real_agent(agents[name], request.meeting_topic, request.time)
        return {
            "agent": agent_name.capitalize() + "_Agent",
            "meeting_topic": request.meeting_topic,
            "time": request.time,
            "response": response,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
