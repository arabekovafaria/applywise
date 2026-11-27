import io
import json
import unittest
from types import SimpleNamespace
from http.client import HTTPMessage
from unittest.mock import patch

from applywise.agent import AgentError, AgentTools, approve_draft, export_draft, run_agent
from applywise.app import Handler, _form, _render_report


class FakeResponse:
    def __init__(self, value): self.value = value
    def __enter__(self): return self
    def __exit__(self, *_): return False
    def read(self): return json.dumps(self.value).encode()


class FakeOpenAI:
    def __init__(self, responses): self.responses = list(responses); self.requests = []
    def __call__(self, request, timeout=0):
        self.requests.append(json.loads(request.data.decode()))
        return FakeResponse(self.responses.pop(0))


def call(name, args, call_id):
    return {"id": "fc_" + call_id, "call_id": call_id, "type": "function_call", "name": name, "arguments": json.dumps(args)}


class AgentTests(unittest.TestCase):
    def test_real_tool_loop_searches_then_stages_then_finishes(self):
        draft_args = {
            "role_title": "AI internship",
            "requirements": [
                {"requirement": "Python", "priority": "high", "evidence_ids": ["R01"]},
                {"requirement": "Product analytics", "priority": "medium", "evidence_ids": []},
            ],
            "action_plan": [{"requirement": "Python", "action": "Lead with the bot project.", "evidence_ids": ["R01"]}],
            "cover_letter": "I built a Python bot.",
            "cover_letter_claims": [{"claim": "Built a Python bot", "evidence_ids": ["R01"]}],
        }
        fake = FakeOpenAI([
            {"output": [call("search_resume_evidence", {"query": "Python"}, "c1")]},
            {"output": [call("stage_application_draft", draft_args, "c2")]},
            {"output": [{"type": "message", "content": [{"type": "output_text", "text": "Draft ready for your review."}]}]},
        ])
        report = run_agent("AI intern needs Python", "Built Python bot", "sk-test", opener=fake)
        self.assertEqual(report["requirements"][0]["evidence"][0]["excerpt"], "Built Python bot")
        self.assertEqual(report["requirements"][1]["status"], "gap")
        self.assertEqual(report["agent_summary"], "Draft ready for your review.")
        self.assertEqual([x["name"] for x in report["tool_trace"]], ["search_resume_evidence", "stage_application_draft"])
        self.assertEqual(len(fake.requests), 3)
        self.assertFalse(fake.requests[0]["store"])
        self.assertEqual(fake.requests[0]["tools"][0]["name"], "search_resume_evidence")
        self.assertEqual(fake.requests[2]["tool_choice"], "none")

    def test_requires_evidence_ids_from_tool_results(self):
        tools = AgentTools("Python developer")
        with self.assertRaises(AgentError):
            tools.stage_application_draft({
                "role_title": "Role", "requirements": [{"requirement": "Python", "priority": "high", "evidence_ids": ["R99"]}],
                "action_plan": [], "cover_letter": "", "cover_letter_claims": [],
            })

    def test_approval_gate_blocks_download_until_explicit_approval(self):
        fake = FakeOpenAI([
            {"output": [call("search_resume_evidence", {"query": "Python"}, "s1")]},
            {"output": [call("stage_application_draft", {
                "role_title": "Role", "requirements": [], "action_plan": [], "cover_letter": "Draft", "cover_letter_claims": [],
            }, "s2")]},
            {"output": [{"type": "message", "content": [{"type": "output_text", "text": "Review it."}]}]},
        ])
        report = run_agent("Python role", "Python developer", "sk-test", opener=fake)
        with self.assertRaises(PermissionError): export_draft(report)
        approve_draft(report)
        self.assertEqual(export_draft(report), "Draft\n")

    def test_api_key_is_not_refilled_into_the_form(self):
        html = _form({"job": ["Role"], "resume": ["CV"], "api_key": ["sk-secret"]})
        self.assertNotIn("sk-secret", html)

    def test_user_content_is_html_escaped_in_report(self):
        report = {"requirements": [], "action_plan": [], "cover_letter_claims": [], "tool_trace": [], "unsupported_claims": [],
                  "approved": False, "draft_id": "id123", "model": "model", "role_title": "<script>alert(1)</script>",
                  "agent_summary": "Ready", "cover_letter": "Safe"}
        html = _render_report(report)
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_post_requests_must_be_same_origin(self):
        headers = HTTPMessage()
        headers.add_header("Host", "localhost:8000")
        headers.add_header("Origin", "https://evil.example")
        self.assertFalse(Handler._same_origin(SimpleNamespace(headers=headers)))
        headers.replace_header("Origin", "http://localhost:8000")
        self.assertTrue(Handler._same_origin(SimpleNamespace(headers=headers)))

    def test_same_origin_accepts_railway_public_domain_when_host_is_internal(self):
        headers = HTTPMessage()
        headers.add_header("Host", "applywise.railway.internal:8000")
        headers.add_header("Origin", "https://applywise-production-d196a324.up.railway.app")
        with patch.dict("os.environ", {"RAILWAY_PUBLIC_DOMAIN": "applywise-production-d196a324.up.railway.app"}):
            self.assertTrue(Handler._same_origin(SimpleNamespace(headers=headers)))

    def test_consent_text_discloses_key_passes_through_server(self):
        html = _form()
        self.assertIn("your API key are sent to the ApplyWise server", html)
        self.assertIn("I understand and consent", html)

    def test_server_configured_key_is_not_requested_or_rendered(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "sk-do-not-render"}):
            html = _form()
        self.assertNotIn('name="api_key"', html)
        self.assertNotIn("sk-do-not-render", html)
        self.assertIn("API key is configured securely by the site owner", html)
        self.assertIn("using the site owner’s configured API key", html)

    def test_requires_both_inputs_and_key(self):
        with self.assertRaises(AgentError): run_agent("", "resume", "sk-test")
        with self.assertRaises(AgentError): run_agent("job", "resume", "")


if __name__ == "__main__": unittest.main()
