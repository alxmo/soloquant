"""
Agent 注册与工厂 - 统一管理所有量化交易 Agent
=====================================================
提供:
- Agent 实例化与注册
- 统一的 Agent 获取接口
- 共享上下文管理
"""

from typing import Dict, List, Optional

from loguru import logger

from src.agents.base_agent import AgentContext, BaseAgent
from src.agents.research_agent import ResearchAgent
from src.agents.strategy_agent import StrategyAgent
from src.agents.backtest_agent import BacktestAgent
from src.agents.execution_agent import ExecutionAgent
from src.agents.monitor_agent import MonitorAgent


class AgentRegistry:
    """Agent 注册表 - 管理所有 Agent 实例"""

    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self._context: AgentContext = AgentContext()
        self._initialized = False

    def initialize(self) -> None:
        """初始化所有 Agent"""
        if self._initialized:
            return

        # 创建并注册所有 Agent
        agents = [
            ResearchAgent(),
            StrategyAgent(),
            BacktestAgent(),
            ExecutionAgent(),
            MonitorAgent(),
        ]

        for agent in agents:
            agent.bind_context(self._context)
            self._agents[agent.name] = agent
            logger.info(f"  Agent 已注册: {agent.name} ({agent.role})")

        self._initialized = True
        logger.info(f"✅ Agent 注册表初始化完成: {len(self._agents)} 个 Agent")

    def get(self, name: str) -> Optional[BaseAgent]:
        """获取指定 Agent"""
        if not self._initialized:
            self.initialize()
        return self._agents.get(name)

    def get_research_agent(self) -> ResearchAgent:
        return self.get("research")

    def get_strategy_agent(self) -> StrategyAgent:
        return self.get("strategy")

    def get_backtest_agent(self) -> BacktestAgent:
        return self.get("backtest")

    def get_execution_agent(self) -> ExecutionAgent:
        return self.get("execution")

    def get_monitor_agent(self) -> MonitorAgent:
        return self.get("monitor")

    def list_agents(self) -> List[str]:
        """列出所有 Agent 名称"""
        if not self._initialized:
            self.initialize()
        return list(self._agents.keys())

    def all_status(self) -> List[dict]:
        """获取所有 Agent 状态"""
        if not self._initialized:
            self.initialize()
        return [agent.status() for agent in self._agents.values()]

    @property
    def context(self) -> AgentContext:
        return self._context

    def reset_context(self) -> None:
        """重置共享上下文"""
        self._context = AgentContext()
        for agent in self._agents.values():
            agent.bind_context(self._context)


# 全局实例
agent_registry = AgentRegistry()
