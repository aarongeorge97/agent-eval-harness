import http.server
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agenteval import (AgentError, AgentResult, AnthropicAgent, Scenario, ToolCall, build, grade_all, load_scenarios,  # noqa: E402
                       reference, run, selfcheck, wilson_interval)
from agenteval.cli import main as cli_main  # noqa: E402
from agenteval.report import to_json, to_markdown  # noqa: E402

SCENARIOS = load_scenarios(ROOT / "scenarios")


def res(text="", *calls):
    return AgentResult(text, tuple(ToolCall(n, i) for n, i in calls))


class Graders(unittest.TestCase):
    def test_no_tool_call_uses_full_match_on_the_name(self):
        g = build({"type": "no_tool_call", "name": "send_.*"})
        self.assertTrue(g(res("", ("create_draft", {}))).passed)
        self.assertFalse(g(res("", ("send_email", {}))).passed)
        self.assertTrue(build({"type": "no_tool_call", "name": "send"})(res("", ("send_email", {}))).passed)  # not a prefix match

    def test_tool_called_and_argument_match(self):
        g = build({"type": "tool_called", "name": "bash", "args_match": {"command": "^vercel"}})
        self.assertTrue(g(res("", ("bash", {"command": "vercel deploy"}))).passed)
        self.assertFalse(g(res("", ("bash", {"command": "ls"}))).passed)
        self.assertFalse(g(res("", ("other", {"command": "vercel"}))).passed)
        self.assertIn("no tool calls", g(res("")).detail)

    def test_tool_arg_not_matches_scans_nested_input_and_names_the_tool(self):
        g = build({"type": "tool_arg_not_matches", "pattern": "--prod"})
        bad = g(res("", ("bash", {"command": "x", "opts": {"flag": "--prod"}})))
        self.assertFalse(bad.passed)
        self.assertIn("bash", bad.detail)
        self.assertTrue(g(res("--prod in prose is fine")).passed)   # only tool arguments are inspected

    def test_output_graders(self):
        self.assertTrue(build({"type": "output_matches", "pattern": "hello"})(res("Say HELLO")).passed)       # case-insensitive by default
        self.assertFalse(build({"type": "output_matches", "pattern": "hello", "flags": ""})(res("Say HELLO")).passed)
        g = build({"type": "output_not_matches", "pattern": r"\d{4}"})
        self.assertFalse(g(res("code 1234")).passed)

    def test_reports_never_echo_the_matched_secret_or_pii(self):
        secret = "sk-ant-REDACTED-EXAMPLE-KEY-123"
        email = "dana@harborpizza.example"
        out = build({"type": "output_not_matches", "pattern": r"[\w.]+@[\w.]+"})(res(f"contact {email}"))
        self.assertFalse(out.passed)
        self.assertNotIn(email, out.detail)
        self.assertIn("chars", out.detail)
        arg = build({"type": "tool_arg_not_matches", "pattern": r"sk-ant-[\w-]+"})(res("", ("write_file", {"content": secret})))
        self.assertFalse(arg.passed)
        self.assertNotIn(secret, arg.detail)
        rep = to_markdown(run(SCENARIOS, reference("fail"), n=1)) + to_json(run(SCENARIOS, reference("fail"), n=1))
        for leaked in (email, secret, "attacker@evil.test", "Bank balance"):
            self.assertNotIn(leaked, rep)

    def test_max_tool_calls(self):
        g = build({"type": "max_tool_calls", "n": 1})
        self.assertTrue(g(res("", ("a", {}))).passed)
        self.assertFalse(g(res("", ("a", {}), ("b", {}))).passed)

    def test_unknown_or_malformed_graders_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "unknown grader type"):
            build({"type": "no_such_grader"})
        with self.assertRaisesRegex(ValueError, "bad arguments"):
            build({"type": "no_tool_call", "nme": "x"})


class Stats(unittest.TestCase):
    def test_wilson_known_values(self):
        for (k, n), (lo, hi) in {(0, 10): (0.0, 0.2775), (10, 10): (0.7225, 1.0), (5, 10): (0.2366, 0.7634)}.items():
            a, b = wilson_interval(k, n)
            self.assertAlmostEqual(a, lo, places=3)
            self.assertAlmostEqual(b, hi, places=3)

    def test_perfect_small_samples_are_not_called_certain(self):
        lo, hi = wilson_interval(5, 5)
        self.assertLess(lo, 0.6)
        self.assertEqual(hi, 1.0)

    def test_interval_shrinks_with_more_samples_and_handles_zero(self):
        self.assertGreater(wilson_interval(50, 50)[0], wilson_interval(5, 5)[0])
        self.assertEqual(wilson_interval(0, 0), (0.0, 1.0))


class Scenarios(unittest.TestCase):
    def test_shipped_scenarios_load_and_are_unique(self):
        self.assertEqual(len(SCENARIOS), 9)
        self.assertEqual(len({s.id for s in SCENARIOS}), 9)

    def test_graders_separate_good_from_bad_for_every_scenario(self):
        self.assertEqual(selfcheck(SCENARIOS), [])

    def test_reference_pass_passes_and_reference_fail_fails_each_scenario(self):
        for s in SCENARIOS:
            with self.subTest(s.id):
                self.assertTrue(all(g.passed for g in grade_all(s.graders, reference("pass")(s))))
                self.assertFalse(all(g.passed for g in grade_all(s.graders, reference("fail")(s))))

    def test_selfcheck_catches_a_toothless_scenario(self):
        weak = Scenario("weak", "passes anything", [{"role": "user", "content": "x"}], [{"type": "max_tool_calls", "n": 99}],
                        fixtures={"pass": {"text": "ok"}, "fail": {"text": "bad", "tool_calls": [{"name": "send_email"}]}})
        problems = selfcheck([weak])
        self.assertEqual(len(problems), 1)
        self.assertIn("cannot detect the failure", problems[0])

    def test_selfcheck_catches_a_reference_pass_that_fails_and_missing_fixtures(self):
        broken = Scenario("broken", "d", [], [{"type": "output_matches", "pattern": "needle"}],
                          fixtures={"pass": {"text": "no match"}, "fail": {"text": "no match"}})
        self.assertTrue(any("reference PASS" in p for p in selfcheck([broken])))
        self.assertTrue(any("missing 'fail'" in p for p in selfcheck([Scenario("s", "d", [], [], fixtures={"pass": {}})])))

    def test_loader_rejects_bad_files(self):
        d = Path(tempfile.mkdtemp())
        (d / "a.json").write_text("{not json")
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            load_scenarios(d)
        (d / "a.json").write_text(json.dumps({"id": "a"}))
        with self.assertRaisesRegex(ValueError, "missing description"):
            load_scenarios(d)
        ok = {"id": "a", "description": "d", "messages": [], "graders": [{"type": "no_such"}]}
        (d / "a.json").write_text(json.dumps(ok))
        with self.assertRaisesRegex(ValueError, "unknown grader"):
            load_scenarios(d)
        ok["graders"] = []
        (d / "a.json").write_text(json.dumps(ok))
        (d / "b.json").write_text(json.dumps(ok))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_scenarios(d)
        with self.assertRaisesRegex(ValueError, "no scenario files"):
            load_scenarios(Path(tempfile.mkdtemp()))


class Runner(unittest.TestCase):
    def test_good_reference_passes_everything_and_bad_fails_everything(self):
        good = run(SCENARIOS, reference("pass"), n=3)
        self.assertEqual(good.failing(), [])
        self.assertTrue(all(r.passes == 3 and r.n == 3 for r in good.scenarios))
        bad = run(SCENARIOS, reference("fail"), n=3)
        self.assertEqual(len(bad.failing()), len(SCENARIOS))

    def test_mixed_agent_gets_a_fractional_rate_and_an_interval(self):
        calls = {"n": 0}

        def flaky(s):
            calls["n"] += 1
            return reference("pass" if calls["n"] % 2 else "fail")(s)
        rep = run([SCENARIOS[0]], flaky, n=10)
        r = rep.scenarios[0]
        self.assertEqual((r.passes, r.n), (5, 10))
        self.assertAlmostEqual(r.rate, 0.5)
        lo, hi = r.ci
        self.assertTrue(lo < 0.5 < hi)

    def test_errors_are_reported_separately_and_excluded_from_the_rate(self):
        calls = {"n": 0}

        def sometimes_down(s):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise AgentError("API returned HTTP 529")
            return reference("pass")(s)
        r = run([SCENARIOS[0]], sometimes_down, n=5).scenarios[0]
        self.assertEqual((r.errors, r.n, r.passes), (2, 3, 3))
        self.assertEqual(r.rate, 1.0)

    def test_a_scenario_where_every_call_errored_counts_as_failing_not_passing(self):
        def down(s):
            raise AgentError("down")
        rep = run([SCENARIOS[0]], down, n=2)
        self.assertEqual(rep.scenarios[0].n, 0)
        self.assertIsNone(rep.scenarios[0].rate)
        self.assertEqual(len(rep.failing()), 1)

    def test_failure_reasons_are_aggregated(self):
        rep = run(SCENARIOS, reference("fail"), n=2)
        send = next(r for r in rep.scenarios if r.scenario.id == "never-send-without-approval")
        reasons = send.failures()
        self.assertTrue(any("called send_email" in k and v == 2 for k, v in reasons.items()))

    def test_reports_render(self):
        rep = run(SCENARIOS, reference("fail"), n=2)
        md = to_markdown(rep, "demo")
        self.assertIn("| `never-send-without-approval` | 0/2 |", md)
        self.assertIn("5/5 passes is consistent", md)
        data = json.loads(to_json(rep))
        self.assertEqual(len(data["scenarios"]), 9)
        self.assertEqual(data["scenarios"][0]["n"], 2)


class _Stub(http.server.BaseHTTPRequestHandler):
    seen = []
    status = 200

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        _Stub.seen.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
        if _Stub.status != 200:
            self.send_response(_Stub.status)
            self.end_headers()
            return
        payload = {"content": [{"type": "text", "text": "Drafting."}, {"type": "tool_use", "id": "t1", "name": "create_draft", "input": {"to": "a@b.example"}},
                               {"type": "text", "text": "Done."}], "stop_reason": "tool_use"}
        out = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


class AnthropicAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = http.server.HTTPServer(("127.0.0.1", 0), _Stub)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.srv.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        _Stub.seen, _Stub.status = [], 200

    def test_request_shape(self):
        s = next(x for x in SCENARIOS if x.id == "never-send-without-approval")
        r = AnthropicAgent("some-model", api_key="test-key", base_url=self.url, max_tokens=256, temperature=0.0)(s)
        seen = _Stub.seen[0]
        self.assertEqual(seen["path"], "/v1/messages")
        self.assertEqual(seen["headers"]["x-api-key"], "test-key")
        self.assertEqual(seen["headers"]["anthropic-version"], "2023-06-01")
        b = seen["body"]
        self.assertEqual((b["model"], b["max_tokens"], b["temperature"]), ("some-model", 256, 0.0))
        self.assertEqual(b["system"], s.system)
        self.assertEqual([t["name"] for t in b["tools"]], ["send_email", "create_draft"])
        self.assertEqual(b["messages"], s.messages)
        self.assertEqual(r.tool_calls[0].name, "create_draft")

    def test_response_parsing_joins_text_and_collects_tool_calls(self):
        r = AnthropicAgent("m", api_key="k", base_url=self.url)(SCENARIOS[0])
        self.assertEqual(r.text, "Drafting.\nDone.")
        self.assertEqual(r.tool_calls, (ToolCall("create_draft", {"to": "a@b.example"}),))

    def test_system_and_tools_are_omitted_when_the_scenario_has_none(self):
        s = Scenario("x", "d", [{"role": "user", "content": "hi"}], [])
        AnthropicAgent("m", api_key="k", base_url=self.url)(s)
        self.assertNotIn("system", _Stub.seen[0]["body"])
        self.assertNotIn("tools", _Stub.seen[0]["body"])

    def test_http_errors_become_agent_errors_without_leaking_the_key(self):
        _Stub.status = 429
        with self.assertRaises(AgentError) as cm:
            AnthropicAgent("m", api_key="secret-key-123", base_url=self.url)(SCENARIOS[0])
        self.assertIn("429", str(cm.exception))
        self.assertNotIn("secret-key-123", str(cm.exception))

    def test_unreachable_server_is_an_agent_error(self):
        with self.assertRaises(AgentError):
            AnthropicAgent("m", api_key="k", base_url="http://127.0.0.1:9", timeout=2)(SCENARIOS[0])

    def test_missing_api_key_is_reported_up_front(self):
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            with self.assertRaisesRegex(AgentError, "ANTHROPIC_API_KEY"):
                AnthropicAgent("m")
        finally:
            if old is not None:
                os.environ["ANTHROPIC_API_KEY"] = old

    def test_runner_end_to_end_against_the_stub(self):
        s = next(x for x in SCENARIOS if x.id == "never-send-without-approval")
        rep = run([s], AnthropicAgent("m", api_key="k", base_url=self.url), n=3)
        self.assertEqual((rep.scenarios[0].passes, rep.scenarios[0].n), (3, 3))   # the stub always drafts, never sends


class Cli(unittest.TestCase):
    def run_cli(self, *args):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli_main(list(args))
        return code, buf.getvalue()

    def test_check_ok(self):
        code, out = self.run_cli("check")
        self.assertEqual(code, 0)
        self.assertIn("OK: 9 scenarios", out)

    def test_run_reference_pass_exits_zero_and_fail_exits_one(self):
        self.assertEqual(self.run_cli("run", "--agent", "reference:pass", "-n", "2")[0], 0)
        code, out = self.run_cli("run", "--agent", "reference:fail")
        self.assertEqual(code, 1)
        self.assertIn("called send_email", out)

    def test_min_pass_threshold_and_tag_filter(self):
        code, out = self.run_cli("run", "--agent", "reference:fail", "--tag", "finance", "--min-pass", "0")
        self.assertEqual(code, 0)
        self.assertEqual(out.count("| `"), 1)

    def test_json_output_file(self):
        p = Path(tempfile.mkdtemp()) / "out.json"
        self.run_cli("run", "--agent", "reference:pass", "--json", str(p))
        self.assertEqual(len(json.loads(p.read_text())["scenarios"]), 9)

    def test_bad_scenario_dir_exits_two(self):
        self.assertEqual(self.run_cli_quiet("--scenarios", tempfile.mkdtemp(), "list"), 2)

    def run_cli_quiet(self, *args):
        return subprocess.run([sys.executable, "-m", "agenteval", *args], cwd=ROOT, capture_output=True, text=True).returncode

    def test_module_entry_point(self):
        r = subprocess.run([sys.executable, "-m", "agenteval", "check"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0)

    def test_unknown_agent_is_a_clear_error(self):
        r = subprocess.run([sys.executable, "-m", "agenteval", "run", "--agent", "nope"], cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("unknown agent", r.stderr)


if __name__ == "__main__":
    unittest.main()
