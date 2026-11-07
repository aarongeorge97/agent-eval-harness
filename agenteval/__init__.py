"""agenteval: deterministic, scenario-based checks of whether an agent follows rules."""
from .agents import AnthropicAgent, reference
from .graders import build, grade_all
from .runner import Report, load_scenarios, run, selfcheck
from .stats import wilson_interval
from .types import AgentError, AgentResult, Grade, Scenario, ToolCall

__all__ = ["AnthropicAgent", "reference", "build", "grade_all", "Report", "load_scenarios", "run", "selfcheck",
           "wilson_interval", "AgentError", "AgentResult", "Grade", "Scenario", "ToolCall"]
__version__ = "0.1.0"
