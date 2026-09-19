# workflow package
"""
工作流编排层 - 串联 Agent 完成完整交易链路
=====================================================
工作流:
  A. daily_trading_loop  - 每日自动运转主循环
  B. strategy_iteration  - 策略迭代优化
  C. emergency_stop      - 紧急止损
"""

from src.workflow.engine import WorkflowEngine, Workflow, WorkflowStep, workflow_engine

__all__ = [
    "WorkflowEngine", "Workflow", "WorkflowStep",
    "workflow_engine",
]
