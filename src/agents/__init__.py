# agents package
"""
Agent 层 - AI 量化交易智能体
=====================================================
5 个 Agent 构成完整交易链路:
  Research -> Strategy -> Backtest -> Execution -> Monitor
"""

from src.agents.base_agent import BaseAgent, AgentContext
from src.agents.research_agent import ResearchAgent
from src.agents.strategy_agent import StrategyAgent
from src.agents.backtest_agent import BacktestAgent
from src.agents.execution_agent import ExecutionAgent
from src.agents.monitor_agent import MonitorAgent
from src.agents.registry import agent_registry

__all__ = [
    "BaseAgent", "AgentContext",
    "ResearchAgent", "StrategyAgent", "BacktestAgent",
    "ExecutionAgent", "MonitorAgent",
    "agent_registry",
]
