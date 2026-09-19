"""
LLM 多模型配置管理
=====================================================
v0.4.0 新增

功能:
- 从 .env 读取多个 OpenAI 兼容 API 配置 (完全由用户配置，无预设)
- 每个模型配置: name, endpoint, api_key, model_id, temperature, max_tokens
- 默认模型选择 + 运行时模型切换
- 可用性检测: 哪些模型已配置且 Key 有效

支持的环境变量格式:
  LLM_DEFAULT=mymodel                 # 默认使用的模型名称

  LLM_MYMODEL_ENDPOINT=https://api.example.com/v1
  LLM_MYMODEL_API_KEY=sk-xxx
  LLM_MYMODEL_MODEL=example-model
  LLM_MYMODEL_TEMPERATURE=0.7
  LLM_MYMODEL_MAX_TOKENS=2000

命名规则: LLM_{NAME}_ENDPOINT / LLM_{NAME}_API_KEY / LLM_{NAME}_MODEL / ...
NAME 不区分大小写，会统一转为小写作为模型标识。
用户必须完整配置 ENDPOINT、API_KEY、MODEL 三项才能使模型可用。
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from dotenv import load_dotenv
from loguru import logger

from src.utils.config import PROJECT_ROOT


@dataclass
class ModelEntry:
    """单个 LLM 模型配置"""
    name: str                           # 模型标识 (用户自定义名称)
    endpoint: str = ""                  # API endpoint
    api_key: str = ""                   # API Key
    model_id: str = ""                  # 模型ID (如 "deepseek-chat", "gpt-4o-mini")
    temperature: float = 0.7            # 温度参数
    max_tokens: int = 2000              # 最大 token 数

    def is_available(self) -> bool:
        """是否已配置且 Key 有效 (非空且非占位符)"""
        return (
            bool(self.endpoint)
            and bool(self.api_key)
            and not self.api_key.startswith("your_")
            and bool(self.model_id)
        )

    def to_dict(self, mask_key: bool = True) -> dict:
        """转字典 (脱敏)"""
        key_display = f"***{self.api_key[-4:]}" if self.is_available() and mask_key else self.api_key
        return {
            "name": self.name,
            "endpoint": self.endpoint,
            "api_key": key_display,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "available": self.is_available(),
        }


class LLMConfig:
    """LLM 多模型配置管理器 (完全由用户 .env 配置驱动)"""

    def __init__(self):
        self._models: Dict[str, ModelEntry] = {}
        self._default_model: str = ""
        self._loaded: bool = False

    def load(self) -> None:
        """从环境变量加载所有 LLM 模型配置"""
        self._models.clear()
        self._loaded = False

        # 读取 .env (确保最新)
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=True)

        # 扫描所有 LLM_{NAME}_* 环境变量
        model_names: set[str] = set()
        for key in os.environ:
            if key.startswith("LLM_") and not key.startswith("LLM_DEFAULT"):
                # LLM_{NAME}_ENDPOINT -> NAME
                parts = key.split("_", 2)  # ["LLM", "NAME", "ENDPOINT"]
                if len(parts) >= 3:
                    model_names.add(parts[1].lower())

        # 为每个发现的模型名构建配置 (全部从环境变量读取，无预设)
        for name in sorted(model_names):
            prefix = f"LLM_{name.upper()}"
            endpoint = os.getenv(f"{prefix}_ENDPOINT", "")
            api_key = os.getenv(f"{prefix}_API_KEY", "")
            model_id = os.getenv(f"{prefix}_MODEL", "")
            temp_str = os.getenv(f"{prefix}_TEMPERATURE", "")
            max_tok_str = os.getenv(f"{prefix}_MAX_TOKENS", "")

            entry = ModelEntry(
                name=name,
                endpoint=endpoint,
                api_key=api_key,
                model_id=model_id,
                temperature=float(temp_str) if temp_str else 0.7,
                max_tokens=int(max_tok_str) if max_tok_str else 2000,
            )
            self._models[name] = entry

            if entry.is_available():
                logger.info(f"  LLM 模型已配置: {name} ({model_id}) @ {endpoint}")
            else:
                logger.debug(f"  LLM 模型未完整配置: {name}")

        # 默认模型
        self._default_model = os.getenv("LLM_DEFAULT", "").lower()
        if not self._default_model:
            # 未指定默认模型时，选择第一个可用的
            for name, entry in self._models.items():
                if entry.is_available():
                    self._default_model = name
                    break

        if self._default_model and self._default_model in self._models:
            entry = self._models[self._default_model]
            if entry.is_available():
                logger.info(f"  LLM 默认模型: {self._default_model}")
            else:
                logger.warning(f"  LLM 默认模型 {self._default_model} 的 API Key 未配置")
        else:
            if model_names:
                logger.warning(f"  LLM_DEFAULT={self._default_model} 未找到匹配的模型配置")

        # 如果没有任何 LLM_{NAME}_* 配置，扫描 OPENAI_API_KEY (旧格式兼容)
        if not self._models:
            old_key = os.getenv("OPENAI_API_KEY", "")
            if old_key and not old_key.startswith("your_"):
                entry = ModelEntry(
                    name="openai",
                    endpoint=os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1"),
                    api_key=old_key,
                    model_id=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                )
                self._models["openai"] = entry
                self._default_model = "openai"
                logger.info("  检测到旧格式 OPENAI_API_KEY，已自动迁移为 openai 模型配置")

        self._loaded = True

        if not self._models:
            logger.info("  未配置任何 LLM 模型，对话将使用规则引擎降级模式")
        elif not self.get_available_models():
            logger.warning("  已配置 LLM 模型但无可用项 (API Key 为空)，对话将使用规则引擎降级模式")

    def _ensure_loaded(self) -> None:
        """确保配置已加载 (懒加载)"""
        if not self._loaded:
            self.load()

    def get_model(self, name: str = "") -> Optional[ModelEntry]:
        """获取指定模型配置，不传 name 则返回默认模型"""
        self._ensure_loaded()
        if name:
            return self._models.get(name.lower())
        if self._default_model:
            return self._models.get(self._default_model)
        return None

    def get_default_name(self) -> str:
        """获取默认模型名称"""
        self._ensure_loaded()
        return self._default_model

    def set_default(self, name: str) -> bool:
        """设置默认模型，返回是否成功"""
        self._ensure_loaded()
        name = name.lower()
        if name in self._models:
            self._default_model = name
            logger.info(f"LLM 默认模型已切换为: {name}")
            return True
        logger.warning(f"模型 {name} 不存在，切换失败")
        return False

    def get_available_models(self) -> List[ModelEntry]:
        """获取所有已配置可用的模型"""
        self._ensure_loaded()
        return [m for m in self._models.values() if m.is_available()]

    def has_available(self) -> bool:
        """是否有至少一个可用模型"""
        self._ensure_loaded()
        return any(m.is_available() for m in self._models.values())

    def is_enabled(self) -> bool:
        """LLM 功能是否启用 (至少有一个可用模型且默认模型可用)"""
        self._ensure_loaded()
        if not self._default_model:
            return False
        entry = self._models.get(self._default_model)
        return entry is not None and entry.is_available()

    def list_models(self) -> List[dict]:
        """列出所有模型配置 (脱敏)"""
        self._ensure_loaded()
        return [m.to_dict() for m in self._models.values()]

    def reload(self) -> None:
        """重新加载配置"""
        self._loaded = False
        self.load()

    def status(self) -> dict:
        """返回配置状态摘要"""
        self._ensure_loaded()
        available = self.get_available_models()
        return {
            "enabled": self.is_enabled(),
            "default_model": self._default_model,
            "total_models": len(self._models),
            "available_models": [m.name for m in available],
            "all_models": [m.name for m in self._models.values()],
        }


# 全局单例
llm_config = LLMConfig()
