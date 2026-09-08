"""
S.A.R.A. Authentication Server
================================
Run this first:   python auth_server.py
Then open:        http://localhost:5001/
Then launch:      python main.py
"""

import os
import json
import time
import hashlib
import secrets
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

# ── Config ─────────────────────────────────────────────────────────────────
AUTH_PORT   = 5001
TOKEN_FILE  = Path(__file__).parent / ".sara_token"
SESSION_TTL = 8 * 60 * 60   # 8 hours

def _sha(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

# Demo accounts — add/change as needed
USERS = {
    "admin@sara.ai": _sha("sara2024"),
    "demo@sara.ai":  _sha("demo1234"),
    "user@sara.ai":  _sha("password"),
}

# ── Token helpers ───────────────────────────────────────────────────────────
def _write_token(email: str) -> str:
    token = secrets.token_hex(32)
    payload = {
        "token":   token,
        "email":   email,
        "issued":  time.time(),
        "expires": time.time() + SESSION_TTL,
    }
    TOKEN_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return token

def _read_token():
    if not TOKEN_FILE.exists():
        return None
    try:
        payload = json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
        if time.time() > payload.get("expires", 0):
            TOKEN_FILE.unlink(missing_ok=True)
            return None
        return payload
    except Exception:
        return None

def _clear_token():
    TOKEN_FILE.unlink(missing_ok=True)

# ── Login page HTML ─────────────────────────────────────────────────────────
LOGIN_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>S.A.R.A. — Sign in</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--bg:#0A0B1A;--violet:#6C5B9E;--indigo:#2E2668;--moon:#E3E1FA;--mist:#8B8FB5;--paper:#F6F5FF;--text:#F4F3FC;--font:'Manrope',sans-serif;--mono:'IBM Plex Mono',monospace;}
*{margin:0;padding:0;box-sizing:border-box;}
html{-webkit-font-smoothing:antialiased;}
body{font-family:var(--font);background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden;}
button{font-family:inherit;border:none;cursor:pointer;}
input{font-family:inherit;}

.scene{position:fixed;inset:0;z-index:0;background:var(--bg);}
.glow{position:absolute;top:-15%;right:-10%;width:55vw;height:55vw;max-width:700px;background:radial-gradient(circle,rgba(227,225,250,0.2),transparent 70%);filter:blur(20px);border-radius:50%;}
.aurora{position:absolute;border-radius:50%;filter:blur(100px);mix-blend-mode:screen;pointer-events:none;}
.a1{width:600px;height:250px;left:-10%;bottom:5%;background:linear-gradient(100deg,transparent,#2E2668 35%,#6C5B9E 60%,transparent);opacity:0.5;transform:rotate(-10deg);}
.a2{width:550px;height:200px;right:-12%;bottom:-5%;background:linear-gradient(100deg,transparent,#6C5B9E 40%,#2E2668 65%,transparent);opacity:0.4;transform:rotate(8deg);}
.stars{position:absolute;inset:0;}
.star{position:absolute;background:#E3E1FA;border-radius:50%;animation:twinkle ease-in-out infinite;}
@keyframes twinkle{0%,100%{opacity:.15}50%{opacity:.8}}

.stage{position:relative;z-index:1;min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:18px;padding:24px;}
.card{position:relative;width:100%;max-width:400px;background:linear-gradient(160deg,rgba(227,225,250,0.09),rgba(227,225,250,0.04));backdrop-filter:blur(28px) saturate(140%);-webkit-backdrop-filter:blur(28px) saturate(140%);border:1px solid rgba(227,225,250,0.14);border-radius:28px;padding:40px 36px;overflow:hidden;box-shadow:0 20px 60px rgba(0,0,0,0.45),inset 0 1px 0 rgba(255,255,255,0.12);animation:rise .9s cubic-bezier(.16,1,.3,1);}
.card::before{content:'';position:absolute;top:-30%;right:-20%;width:65%;height:65%;background:radial-gradient(circle,rgba(227,225,250,0.16),transparent 70%);pointer-events:none;}
@keyframes rise{from{opacity:0;transform:translateY(18px) scale(.98)}to{opacity:1;transform:none}}

.brand{display:flex;flex-direction:column;align-items:center;margin-bottom:26px;}
.logo-wrap{position:relative;width:46px;height:46px;display:flex;align-items:center;justify-content:center;margin-bottom:12px;}
.logo-wrap::before{content:'';position:absolute;inset:-10px;background:radial-gradient(circle,rgba(108,91,158,.5),transparent 70%);filter:blur(8px);}
.logo-wrap svg{position:relative;filter:drop-shadow(0 0 8px rgba(227,225,250,.45));}
.wordmark{font-family:var(--mono);font-size:11px;letter-spacing:.22em;color:var(--mist);}

h1{font-size:26px;font-weight:600;text-align:center;letter-spacing:-.02em;margin-bottom:6px;}
.sub{font-size:13.5px;color:var(--mist);text-align:center;margin-bottom:20px;}

.hint{background:rgba(108,91,158,.15);border:1px solid rgba(108,91,158,.3);border-radius:12px;padding:12px 14px;margin-bottom:20px;font-size:12px;color:var(--mist);line-height:1.7;}
.hint strong{color:var(--moon);}
.hint code{font-family:var(--mono);background:rgba(255,255,255,.07);padding:1px 5px;border-radius:4px;font-size:11px;}

.banner{display:none;border-radius:12px;padding:12px 16px;font-size:13.5px;margin-bottom:16px;text-align:center;animation:rise .35s ease;}
.ok{background:rgba(0,240,120,.12);border:1px solid rgba(0,240,120,.35);color:#5cffb0;}
.err{background:rgba(255,60,60,.12);border:1px solid rgba(255,60,60,.35);color:#ff8a8a;}

.field{margin-bottom:14px;}
.field label{display:block;font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--mist);margin-bottom:7px;}
.field input{width:100%;padding:12px 14px;background:rgba(227,225,250,.05);border:1px solid rgba(227,225,250,.14);border-radius:12px;color:var(--text);font-size:14px;transition:border-color .2s,box-shadow .2s,background .2s;}
.field input:focus{outline:none;border-color:var(--violet);background:rgba(227,225,250,.08);box-shadow:0 0 0 4px rgba(108,91,158,.25);}
.field input::placeholder{color:rgba(227,225,250,.25);}
.pw-wrap{position:relative;}
.pw-wrap input{padding-right:42px;}
.eye-btn{position:absolute;right:10px;top:50%;transform:translateY(-50%);color:var(--mist);padding:6px;display:flex;border-radius:6px;background:none;transition:color .2s;}
.eye-btn:hover{color:var(--text);}

.row{display:flex;align-items:center;justify-content:space-between;margin:4px 0 22px;font-size:13px;}
.check{display:flex;align-items:center;gap:8px;color:var(--mist);cursor:pointer;}
.check input{width:15px;height:15px;accent-color:var(--violet);}
.row a{color:var(--mist);text-decoration:none;transition:color .2s;}
.row a:hover{color:var(--text);}

.btn{width:100%;padding:14px;background:linear-gradient(135deg,var(--violet),var(--indigo));border-radius:14px;color:var(--text);font-size:15px;font-weight:600;transition:transform .25s cubic-bezier(.16,1,.3,1),box-shadow .25s;box-shadow:0 10px 30px rgba(108,91,158,.35);}
.btn:hover:not(:disabled){transform:translateY(-2px);box-shadow:0 14px 36px rgba(108,91,158,.5);}
.btn:disabled{opacity:.7;cursor:not-allowed;}
.spin{display:inline-block;width:15px;height:15px;border:2px solid rgba(255,255,255,.2);border-top-color:#fff;border-radius:50%;animation:sp .7s linear infinite;vertical-align:middle;margin-right:7px;}
@keyframes sp{to{transform:rotate(360deg)}}

.trust{z-index:1;display:flex;align-items:center;gap:6px;font-family:var(--mono);font-size:10px;letter-spacing:.05em;color:var(--mist);opacity:.6;}

@media(max-width:480px){.card{padding:32px 22px;}.hint{display:none;}}
</style>
</head>
<body>
<div class="scene" aria-hidden="true">
  <div class="stars" id="stars"></div>
  <div class="glow"></div>
  <div class="aurora a1"></div>
  <div class="aurora a2"></div>
</div>

<main class="stage">
  <div class="card">
    <div class="brand">
      <div class="logo-wrap">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
          <defs><linearGradient id="g" x1="0" y1="0" x2="24" y2="24"><stop offset="0%" stop-color="#E3E1FA"/><stop offset="100%" stop-color="#6C5B9E"/></linearGradient></defs>
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" fill="url(#g)"/>
        </svg>
      </div>
      <span class="wordmark">S.A.R.A. &bull; LUCID PORTAL</span>
    </div>

    <div class="banner ok" id="bannerOk"></div>
    <div class="banner err" id="bannerErr"></div>

    <h1 id="title">Welcome back</h1>
    <p class="sub" id="sub">Sign in to unlock the S.A.R.A. system.</p>

    <div class="hint">
      <strong>Demo accounts</strong><br>
      <code>admin@sara.ai</code> / <code>sara2024</code><br>
      <code>demo@sara.ai</code> &nbsp;/ <code>demo1234</code>
    </div>

    <form id="form">
      <div class="field">
        <label for="email">Email</label>
        <input id="email" type="email" placeholder="you@sara.ai" autocomplete="email" required>
      </div>
      <div class="field">
        <label for="pw">Password</label>
        <div class="pw-wrap">
          <input id="pw" type="password" placeholder="••••••••" autocomplete="current-password" required>
          <button type="button" class="eye-btn" id="eyeBtn" aria-label="Show password"></button>
        </div>
      </div>
      <div class="row">
        <label class="check"><input type="checkbox" checked> Remember me</label>
        <a href="#">Forgot password?</a>
      </div>
      <button type="submit" class="btn" id="submitBtn">Sign in to S.A.R.A.</button>
    </form>
  </div>

  <div class="trust">
    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
    Token stored locally &bull; 8-hour session
  </div>
</main>

<script>
(function(){
  var rm = window.matchMedia('(prefers-reduced-motion:reduce)').matches;

  // Stars
  var sf=document.getElementById('stars'), frag=document.createDocumentFragment();
  for(var i=0;i<55;i++){
    var s=document.createElement('div'); s.className='star';
    var sz=(Math.random()*1.8+0.7).toFixed(2);
    s.style.cssText='width:'+sz+'px;height:'+sz+'px;left:'+(Math.random()*100).toFixed(2)+'%;top:'+(Math.random()*100).toFixed(2)+'%;opacity:'+(Math.random()*0.5+0.15).toFixed(2)+';'+(rm?'animation:none':'animation-duration:'+(Math.random()*3+3).toFixed(2)+'s;animation-delay:'+(Math.random()*5).toFixed(2)+'s');
    frag.appendChild(s);
  }
  sf.appendChild(frag);

  // Aurora drift
  if(!rm){
    var au=document.querySelectorAll('.aurora'),br=[-10,8],t=0;
    (function d(){ t+=.0025; au.forEach(function(e,i){ e.style.transform='translate('+Math.sin(t+i*2)*22+'px,'+Math.cos(t*.8+i)*10+'px) rotate('+br[i%2]+'deg)'; }); requestAnimationFrame(d); })();
  }

  // Eye toggle
  var eb=document.getElementById('eyeBtn'), pi=document.getElementById('pw');
  var ei='<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>';
  var eo='<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.94 17.94A10.94 10.94 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/></svg>';
  eb.innerHTML=ei;
  eb.onclick=function(){ var sh=pi.type==='password'; pi.type=sh?'text':'password'; eb.innerHTML=sh?eo:ei; eb.setAttribute('aria-label',sh?'Hide':'Show'); };

  var ok=document.getElementById('bannerOk'), err=document.getElementById('bannerErr');
  function hide(){ ok.style.display='none'; err.style.display='none'; }

  // Check existing session on load
  fetch('/api/status').then(function(r){return r.json();}).then(function(d){
    if(d.authenticated){
      ok.innerHTML='&#10003; Already signed in as <strong>'+d.email+'</strong> &mdash; session active.';
      ok.style.display='block';
      document.getElementById('title').textContent='S.A.R.A. Ready';
      document.getElementById('sub').textContent='Session active. You may now run main.py.';
      document.getElementById('form').style.display='none';
      document.querySelector('.hint').style.display='none';
    }
  }).catch(function(){});

  // Submit
  var form=document.getElementById('form'), btn=document.getElementById('submitBtn');
  form.addEventListener('submit', function(e){
    e.preventDefault();
    if(btn.disabled) return;
    hide();
    btn.disabled=true;
    btn.innerHTML='<span class="spin"></span>Verifying…';

    var email=document.getElementById('email').value.trim();
    var pw=document.getElementById('pw').value;

    fetch('/api/login',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({email:email,password:pw})
    }).then(function(r){return r.json();}).then(function(d){
      btn.disabled=false;
      btn.innerHTML='Sign in to S.A.R.A.';
      if(d.ok){
        ok.innerHTML='&#10003; <strong>'+email+'</strong> authenticated. Run <code>python main.py</code> to launch S.A.R.A.';
        ok.style.display='block';
        document.getElementById('title').textContent='S.A.R.A. Unlocked';
        document.getElementById('sub').textContent='Authentication complete. Start main.py in your terminal.';
        form.style.display='none';
        document.querySelector('.hint').style.display='none';
      } else {
        err.textContent='✗ '+d.message;
        err.style.display='block';
      }
    }).catch(function(){
      btn.disabled=false;
      btn.innerHTML='Sign in to S.A.R.A.';
      err.textContent='✗ Cannot reach auth server. Is auth_server.py running on port 5001?';
      err.style.display='block';
    });
  });
})();
</script>
</body>
</html>
"""

# ── HTTP Handler ─────────────────────────────────────────────────────────────
class SARAAuthHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        print(f"[AUTH] {self.address_string()} — {fmt % args}")

    def _json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html, status=200):
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n)) if n > 0 else {}

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/login", "/index.html"):
            self._html(LOGIN_HTML)
        elif path == "/api/status":
            p = _read_token()
            if p:
                self._json({"authenticated": True, "email": p["email"], "expires": p["expires"]})
            else:
                self._json({"authenticated": False})
        else:
            self._html("<h1>404</h1>", 404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/login":
            b = self._body()
            email = (b.get("email") or "").strip().lower()
            pw    = b.get("password") or ""
            expected = USERS.get(email)
            if expected and secrets.compare_digest(expected, _sha(pw)):
                _write_token(email)
                self._json({"ok": True,  "message": "Authenticated", "email": email})
                print(f"[AUTH] ✓ Login OK — {email}")
            else:
                self._json({"ok": False, "message": "Invalid email or password."}, 401)
                print(f"[AUTH] ✗ Failed — '{email}'")
        elif path == "/api/logout":
            _clear_token()
            self._json({"ok": True, "message": "Logged out."})
        else:
            self._html("<h1>404</h1>", 404)

# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from urllib.parse import urlparse
    server = HTTPServer(("localhost", AUTH_PORT), SARAAuthHandler)
    print(f"""
╔══════════════════════════════════════════════╗
║   S.A.R.A. Auth Server — ready              ║
║   http://localhost:{AUTH_PORT}/                   ║
╠══════════════════════════════════════════════╣
║  Accounts:                                   ║
║    admin@sara.ai  /  sara2024                ║
║    demo@sara.ai   /  demo1234                ║
╠══════════════════════════════════════════════╣
║  Steps:                                      ║
║  1) Open http://localhost:{AUTH_PORT}/ & log in  ║
║  2) Run: python main.py                      ║
╚══════════════════════════════════════════════╝
""")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[AUTH] Stopped.")
        server.server_close()
