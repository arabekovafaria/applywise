"""Local ApplyWise browser demo. Start with: python -m applywise.app"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from html import escape
from urllib.parse import parse_qs, urlparse
from datetime import datetime, timedelta, timezone
from threading import BoundedSemaphore, RLock
import os
from .agent import API_PROVIDER, AgentError, DEFAULT_MODEL, approve_draft, export_draft, run_agent

# Intentionally in-memory only. Restarting the server clears all staged drafts.
SESSIONS: dict[str, dict] = {}
SESSION_TIMES: dict[str, datetime] = {}
SESSION_LOCK = RLock()
SESSION_TTL = timedelta(minutes=45)
MAX_REQUEST_BYTES = 300_000
MAX_CONCURRENT_REQUESTS = 6

PAGE = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>ApplyWise</title>
<style>
:root{--ink:#19283e;--blue:#315dc8;--muted:#66758b;--line:#e0e6ef;--bg:#f5f7fa}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,sans-serif}header{background:#172b4d;color:white;padding:20px max(calc((100% - 1060px)/2),24px)}header b{font-size:22px;letter-spacing:.2px}header span{margin-left:12px;color:#c7d3e8;font-size:13px}main{max-width:1060px;margin:28px auto;padding:0 18px}.intro{max-width:780px;color:var(--muted);margin:0 0 20px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.card{background:white;border:1px solid var(--line);border-radius:14px;padding:20px;margin:14px 0;box-shadow:0 6px 22px #172b4d08}h1{font-size:25px;margin:0 0 6px}h2{font-size:19px;margin:4px 0 10px}h3{font-size:15px;margin:12px 0 5px}textarea,input{font:inherit}textarea{width:100%;height:230px;border:1px solid #cbd4e1;border-radius:8px;padding:12px;font:13px/1.5 ui-monospace,monospace;resize:vertical}input[type=password],input[type=text]{width:100%;border:1px solid #cbd4e1;border-radius:8px;padding:10px}label{font-weight:650;font-size:14px}.muted{color:var(--muted);font-size:13px}.consent{background:#fff6df;border:1px solid #f1dfb3;border-radius:10px;padding:12px;color:#55461f;font-size:13px}.check{display:flex;gap:10px;align-items:flex-start;margin:12px 0}.check input{margin-top:4px}.btn{display:inline-block;background:var(--blue);color:white;border:0;border-radius:8px;padding:11px 17px;font-weight:700;text-decoration:none;cursor:pointer}.btn.secondary{background:#eef2fa;color:#244890}.error{background:#fff0f0;border:1px solid #edc8c8;border-radius:9px;padding:12px;color:#8a2525}.tag{display:inline-block;background:#eaf0ff;color:#254b9a;border-radius:20px;padding:3px 9px;margin:2px 4px 2px 0;font-size:13px}.gap{background:#fff2d9;color:#855900}.ev{background:#f4f6f9;border-left:3px solid #7e9be0;border-radius:4px;padding:8px 10px;margin:5px 0;font-size:13px}.trace{font:12px/1.7 ui-monospace,monospace;background:#f7f8fa;padding:12px;border-radius:8px}.success{background:#e8f5ec;color:#225b32;padding:12px;border-radius:9px}.claim-warning{color:#8b5611}.actions{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.pre{white-space:pre-wrap;background:#fbfcfe;border:1px solid var(--line);border-radius:9px;padding:14px;font:14px/1.65 system-ui}.req{border-bottom:1px solid var(--line);padding:10px 0}.small{font-size:12px;color:var(--muted)}@media(max-width:720px){.grid{grid-template-columns:1fr}header span{display:block;margin-left:0}}
</style></head><body><header><b>ApplyWise</b><span>AI agent · real tool use · human approval</span></header><main>
<h1>Prepare a thoughtful application</h1><p class="intro">The agent reads the role, searches your resume for evidence using local tools, compares requirements, and stages a tailored draft. It will never submit an application.</p>
{content}</main></body></html>'''


def _layout(content: str) -> bytes:
    return PAGE.replace("{content}", content).encode("utf-8")


def _error(message: str) -> str:
    return f'<div class="error">{escape(message)}</div>'


def _render_report(report: dict) -> str:
    requirements = []
    for row in report["requirements"]:
        kind = "tag" if row["status"] == "supported" else "tag gap"
        status = "Evidence found" if row["status"] == "supported" else "Gap to address"
        citations = "".join(f'<div class="ev"><b>{escape(x["id"])}</b> · {escape(x["excerpt"])}</div>' for x in row["evidence"])
        requirements.append(f'<div class="req"><span class="{kind}">{escape(row["requirement"])} · {status}</span>{citations}</div>')
    actions = "".join(f'<li><b>{escape(x["requirement"])}</b>: {escape(x["action"])}' + "".join(f'<div class="ev">{escape(e["id"])} · {escape(e["excerpt"])}</div>' for e in x["evidence"]) + "</li>" for x in report["action_plan"])
    claims = "".join(f'<li>{escape(x["claim"])} — ' + (", ".join(escape(e["id"]) for e in x["evidence"]) if x["evidence"] else '<span class="claim-warning">no resume evidence attached</span>') + '</li>' for x in report["cover_letter_claims"])
    trace = "<br>".join(f'{escape(x["name"])} → {escape(x["output_summary"])}' for x in report["tool_trace"])
    if report["unsupported_claims"]:
        warning = '<div class="error"><b>Review these factual claims carefully:</b><ul>' + "".join(f"<li>{escape(x)}</li>" for x in report["unsupported_claims"]) + "</ul></div>"
    else:
        warning = '<div class="consent">The draft claims include evidence references. Still verify every sentence and replace any wording that does not reflect your experience.</div>'
    approval = (f'<div class="success"><b>Approved for local download.</b> This only creates a text file for you; it does not send anything.</div><p><a class="btn" href="/download?id={escape(report["draft_id"])}">Download cover letter</a></p>' if report["approved"] else f'''<form method="post" action="/approve"><input type="hidden" name="draft_id" value="{escape(report["draft_id"])}"><label class="check"><input type="checkbox" name="reviewed" value="yes" required><span>I reviewed the draft and its evidence, and approve making a local copy.</span></label><button class="btn" type="submit">Approve and unlock download</button></form>''')
    summary = f'<p class="muted">Model: {escape(report["model"])} · Draft: {escape(report["draft_id"][:10])}</p>'
    return f'''<section class="card"><h2>Agent review: {escape(report["role_title"])}</h2>{summary}<p class="success">Draft prepared. Review the evidence, gaps, and cover letter below.</p><h3>Requirements and resume evidence</h3>{''.join(requirements) or '<p>No requirements were returned. Check the role description.</p>'}<h3>Action plan</h3><ul>{actions}</ul><h3>Cover letter draft</h3><div class="pre">{escape(report["cover_letter"])}</div><h3>Factual-claim evidence audit</h3><ul>{claims}</ul>{warning}<h3>Tool activity</h3><div class="trace">{trace}</div><h3>Review and approve</h3><p class="small">Approval is required to unlock the local download. No job application will be sent.</p>{approval}</section>'''


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, status: int = 200, content_type: str = "text/html; charset=utf-8", extra: dict | None = None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        # Same-origin form submissions can use Referer as a CSRF fallback
        # when the browser omits Origin. Never send the referrer cross-origin.
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'")
        for k, v in (extra or {}).items(): self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/download":
            draft_id = parse_qs(parsed.query).get("id", [""])[0]
            report = _get_session(draft_id)
            if not report or not report.get("approved"):
                return self._send(_layout(_error("Review and approve the draft before downloading.")), 403)
            data = export_draft(report).encode("utf-8")
            filename = "applywise-cover-letter.txt"
            return self._send(data, 200, "text/plain; charset=utf-8", {"Content-Disposition": f'attachment; filename="{filename}"'})
        self._send(_layout(_form()))

    def do_POST(self):
        parsed = urlparse(self.path)
        if not self._same_origin():
            return self._send(_layout(_error("For security, submit the form from this same ApplyWise page.")), 403)
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            return self._send(_layout(_error("Invalid request size.")), 400)
        if length <= 0 or length > MAX_REQUEST_BYTES:
            return self._send(_layout(_error("The request is empty or too large. Keep each text field under 30,000 characters.")), 413)
        try:
            form = parse_qs(self.rfile.read(length).decode("utf-8"), keep_blank_values=True)
        except UnicodeDecodeError:
            return self._send(_layout(_error("The request must use UTF-8 text.")), 400)
        if parsed.path == "/approve":
            draft_id = form.get("draft_id", [""])[0]
            report = _get_session(draft_id)
            if not report or form.get("reviewed", [""])[0] != "yes":
                return self._send(_layout(_error("A draft and explicit review confirmation are required.")), 400)
            approve_draft(report)
            return self._send(_layout(_render_report(report)))
        if parsed.path != "/":
            return self._send(_layout(_error("Page not found.")), 404)
        if form.get("consent", [""])[0] != "yes":
            return self._send(_layout(_error(f"Confirm that you understand the job description and resume will be sent to {API_PROVIDER} for analysis.") + _form(form)), 400)
        job = form.get("job", [""])[0]
        resume = form.get("resume", [""])[0]
        # A key configured in the hosting environment takes precedence and
        # never needs to pass through the browser. Local users can still enter
        # their own key when no server key is configured.
        key = os.environ.get("GROQ_API_KEY", "").strip() or form.get("api_key", [""])[0]
        model = form.get("model", [DEFAULT_MODEL])[0].strip() or DEFAULT_MODEL
        try:
            report = run_agent(job, resume, key, model)
            _put_session(report["draft_id"], report)
            content = _render_report(report)
        except AgentError as exc:
            content = _error(str(exc)) + _form(form)
        except Exception:
            content = _error("The agent could not complete this run. Please retry; the API key is not stored.") + _form(form)
        self._send(_layout(content))

    def log_message(self, *_args):
        # Suppress default request logs so form content and query strings are not logged.
        return

    def _same_origin(self) -> bool:
        host = self.headers.get("Host", "")
        # Railway's edge can forward requests with an internal Host header.
        # Match the deployment-provided public domain and Railway's forwarded
        # original host. Only accept the latter for Railway-provided domains.
        railway_domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "").strip()
        configured_domain = os.environ.get("APPLYWISE_PUBLIC_DOMAIN", "").strip()
        forwarded_host = self.headers.get("X-Forwarded-Host", "").split(",", 1)[0].strip()
        origin = self.headers.get("Origin", "")
        referer = self.headers.get("Referer", "")
        source = urlparse(origin or referer)
        allowed_hosts = {value for value in (host, railway_domain, configured_domain) if value}
        if forwarded_host.casefold().endswith(".up.railway.app"):
            allowed_hosts.add(forwarded_host)
        return bool(allowed_hosts and source.netloc in allowed_hosts and source.scheme in {"http", "https"})


class BoundedHTTPServer(ThreadingHTTPServer):
    """Keep slow model calls from creating an unbounded number of threads."""
    daemon_threads = True
    request_queue_size = 16
    _slots = BoundedSemaphore(MAX_CONCURRENT_REQUESTS)

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            try:
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\nContent-Length: 0\r\nRetry-After: 30\r\n\r\n")
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


def _expire_sessions() -> None:
    now = datetime.now(timezone.utc)
    expired = [key for key, created in SESSION_TIMES.items() if now - created > SESSION_TTL]
    for key in expired:
        SESSION_TIMES.pop(key, None)
        SESSIONS.pop(key, None)


def _get_session(draft_id: str) -> dict | None:
    with SESSION_LOCK:
        _expire_sessions()
        return SESSIONS.get(draft_id)


def _put_session(draft_id: str, report: dict) -> None:
    with SESSION_LOCK:
        _expire_sessions()
        SESSIONS[draft_id] = report
        SESSION_TIMES[draft_id] = datetime.now(timezone.utc)


def _form(values: dict | None = None) -> str:
    v = values or {}
    job = escape(v.get("job", [""])[0], quote=True)
    resume = escape(v.get("resume", [""])[0], quote=True)
    model = escape(v.get("model", [DEFAULT_MODEL])[0], quote=True)
    checked = ' checked' if v.get("consent", [""])[0] == "yes" else ""
    provider = escape(API_PROVIDER)
    server_key_configured = bool(os.environ.get("GROQ_API_KEY", "").strip())
    if server_key_configured:
        key_control = f'<p class="small">The {provider} API key is configured securely by the site owner and is used for this request.</p>'
        consent_copy = f'The job description and resume are sent through the ApplyWise server to {provider} using the site owner’s configured API key. The server operator can access information processed by the server. The app does not save these values to disk; {provider} processes the request under the key owner’s account settings.'
        consent_label = f'I understand and consent to sending these texts through this ApplyWise server to {provider} for this analysis.'
    else:
        key_control = f'<p><label for="api_key">{provider} API key</label><input id="api_key" type="password" name="api_key" autocomplete="off" required><span class="small">Used for this request only; ApplyWise does not save it.</span></p>'
        consent_copy = f'The job description, resume, and your API key are sent to the ApplyWise server to make a request to {provider}. The server operator can technically access information processed by the server. This app does not save these values to disk; the API provider processes the request under your account settings.'
        consent_label = f'I understand and consent to sending these texts and my API key through the ApplyWise server to {provider} for this analysis.'
    return f'''<form method="post" class="card"><div class="grid"><div><label for="job">Job description</label><p class="muted">Paste the complete listing or the parts you care about most.</p><textarea id="job" name="job" maxlength="30000" required>{job}</textarea></div><div><label for="resume">Resume</label><p class="muted">Paste resume text. It is used only for this run.</p><textarea id="resume" name="resume" maxlength="30000" required>{resume}</textarea></div></div>{key_control}<p><label for="model">Model ID</label><input id="model" type="text" name="model" value="{model}" maxlength="100"></p><div class="consent">{consent_copy}</div><label class="check"><input type="checkbox" name="consent" value="yes"{checked} required><span>{consent_label}</span></label><button class="btn" type="submit">Run the AI agent</button></form>'''


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    print(f"ApplyWise is running on {host}:{port}")
    BoundedHTTPServer((host, port), Handler).serve_forever()
