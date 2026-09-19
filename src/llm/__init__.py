"""
LLM 智能对话模块
=====================================================
v0.4.0 新增

模块:
- llm_config: 多模型配置管理
- llm_client: OpenAI 兼容 API 客户端
- dialogue_manager: 对话上下文管理器
- intent_recognizer: LLM 意图识别器
- response_generator: LLM 回复生成器
"""

from src.llm.llm_config import LLMConfig, ModelEntry, llm_config
from src.llm.llm_client import LLMClient, llm_client
from src.llm.dialogue_manager import DialogueManager, dialogue_manager
from src.llm.intent_recognizer import IntentRecognizer, intent_recognizer
from src.llm.response_generator import ResponseGenerator, response_generator

__all__ = [
    "LLMConfig",
    "ModelEntry",
    "llm_config",
    "LLMClient",
    "llm_client",
    "DialogueManager",
    "dialogue_manager",
    "IntentRecognizer",
    "intent_recognizer",
    "ResponseGenerator",
    "response_generator",
]
