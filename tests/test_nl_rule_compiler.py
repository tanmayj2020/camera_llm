"""Tests for the NL Rule Compiler — natural language → detection rule conversion."""

import pytest
from cloud.services.nl_rule_compiler.compiler import NLRuleCompiler, CompiledRule


@pytest.fixture
def compiler():
    return NLRuleCompiler()  # no VLM — exercises heuristic fallback


class TestNLRuleCompiler:
    def test_compile_loitering_rule(self, compiler):
        rule = compiler.compile("Alert me when someone loiters near the ATMs for more than 2 minutes")
        assert isinstance(rule, CompiledRule)
        assert rule.rule_id.startswith("nl_")
        assert rule.active is True
        # Should have a duration condition with 120 seconds
        dur = [c for c in rule.conditions if c["type"] == "duration"]
        assert len(dur) == 1
        assert dur[0]["min_seconds"] == 120

    def test_compile_time_constraint(self, compiler):
        rule = compiler.compile("Alert after 10pm when people are near the entrance")
        time_conds = [c for c in rule.conditions if c["type"] == "time"]
        assert len(time_conds) == 1
        assert time_conds[0]["after_hour"] == 22

    def test_compile_running_detection(self, compiler):
        rule = compiler.compile("Alert when someone is running in the lobby")
        speed = [c for c in rule.conditions if c["type"] == "speed"]
        assert len(speed) == 1
        assert speed[0]["min_speed"] >= 2.0

    def test_compile_crowd_count(self, compiler):
        rule = compiler.compile("Alert when more than 30 people are in the plaza")
        count = [c for c in rule.conditions if c["type"] == "count"]
        assert len(count) == 1
        assert count[0]["min_count"] == 30

    def test_compile_gunshot_severity(self, compiler):
        rule = compiler.compile("Emergency alert for gunshot detected near parking")
        assert rule.severity == "critical"
        audio = [c for c in rule.conditions if c["type"] == "audio"]
        assert len(audio) == 1
        assert audio[0]["sound_class"] == "gunshot"

    def test_list_and_delete(self, compiler):
        r1 = compiler.compile("Alert when someone runs")
        r2 = compiler.compile("Alert for glass breaking")
        assert len(compiler.list_rules()) == 2

        compiler.delete_rule(r1.rule_id)
        assert len(compiler.list_rules()) == 1

    def test_toggle_rule(self, compiler):
        rule = compiler.compile("Alert for screaming")
        assert rule.active is True

        compiler.toggle_rule(rule.rule_id, False)
        rules = compiler.list_rules()
        assert rules[0]["active"] is False

        compiler.toggle_rule(rule.rule_id, True)
        rules = compiler.list_rules()
        assert rules[0]["active"] is True

    def test_explain_rule(self, compiler):
        rule = compiler.compile("Alert when 5 people loiter for 3 minutes near exit")
        explanation = compiler.explain_rule(rule.rule_id)
        assert "Rule:" in explanation
        assert rule.source_text in explanation

    def test_decompile(self, compiler):
        text = "Alert when someone runs in the lobby"
        rule = compiler.compile(text)
        assert compiler.decompile(rule.rule_id) == text

    def test_unknown_rule_returns_empty(self, compiler):
        assert compiler.decompile("nonexistent") == ""
        assert compiler.explain_rule("nonexistent") == "Rule not found."
        assert compiler.delete_rule("nonexistent") is False
