"""
主入口 - AI 全自动美股量化交易系统
=====================================================
v0.4.0: LLM 智能对话升级 (意图识别 + 回复生成 + 多轮对话)
v0.3.0: 新增 dashboard 和 scheduler 命令

用法:
  python main.py              # 默认进入对话模式
  python main.py --debug      # 开启 debug 模式 (显示完整日志路径)
  python main.py chat         # 自然语言对话
  python main.py chat --debug  # 对话模式 + debug 日志
  python main.py menu         # 菜单模式
  python main.py auto         # 全自动循环
  python main.py goal "我有10万美元，想赚20%"  # 目标驱动 AI 自主交易
  python main.py status       # 系统状态
  python main.py health       # 健康检查
  python main.py demo         # 数据演示
  python main.py init         # 初始化 Qlib 数据
  python main.py dashboard    # 启动 Streamlit 仪表盘
  python main.py scheduler    # 启动定时任务调度器
  python main.py emergency    # 紧急止损
  python main.py realtime     # 启动实时行情 WebSocket
  python main.py realtime AAPL,MSFT,NVDA  # 指定股票
  python main.py cloud        # 启动云端服务 (API+调度器+Dashboard)
  python main.py api          # 仅启动模型同步 API

  --debug 标志可附加到任意命令后，开启完整日志输出 (含模块:函数:行号)
"""

import sys
import os

# Ensure project root is in Python path (for embedded Python environments)
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 统一委托到 src.ui.cli 的 main 函数
from src.ui.cli import main

if __name__ == "__main__":
    main()
