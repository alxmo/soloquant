"""
LLM API 客户端
=====================================================
v0.4.0 新增

功能:
- 基于 openai 库，兼容所有 OpenAI 格式 API (用户自行配置任意兼容提供商)
- chat(messages, model=None) -> str:  核心对话方法
- chat_json(messages, model=None) -> dict: 结构化 JSON 输出
- 自动重试 (3次) + 超时处理 + 异常封装
- is_available() -> bool: 可用性检测

使用示例:
    from src.llm.llm_client import llm_client

    if llm_client.is_available():
        reply = llm_client.chat([
            {"role": "system", "content": "你是量化交易助手"},
            {"role": "user", "content": "帮我分析 AAPL"},
        ])
"""

import json
import time
from typing import Dict, List, Optional

from loguru import logger

from src.llm.llm_config import ModelEntry, llm_config


class LLMCallError(Exception):
    """LLM 调用异常"""
    pass


class LLMClient:
    """LLM API 客户端"""

    # 最大重试次数
    MAX_RETRIES = 3
    # 重试间隔 (秒)
    RETRY_DELAY = 1.0
    # 请求超时 (秒)
    TIMEOUT = 30

    def __init__(self):
        self._client_cache: Dict[str, object] = {}  # model_name -> openai.Client

    def _get_client(self, model: ModelEntry):
        """
        获取或创建 OpenAI 客户端实例

        使用缓存避免重复创建连接
        """
        if model.name in self._client_cache:
            return self._client_cache[model.name]

        try:
            from openai import OpenAI
        except ImportError:
            raise LLMCallError("openai 库未安装，请运行: pip install openai")

        client = OpenAI(
            api_key=model.api_key,
            base_url=model.endpoint,
            timeout=self.TIMEOUT,
        )
        self._client_cache[model.name] = client
        return client

    def is_available(self, model_name: str = "") -> bool:
        """
        检测 LLM 是否可用

        Args:
            model_name: 指定模型名，不传则检查默认模型

        Returns:
            是否可用
        """
        model = llm_config.get_model(model_name)
        if model is None or not model.is_available():
            return False

        # 检查 openai 库是否安装
        try:
            import openai  # noqa: F401
        except ImportError:
            logger.warning("openai 库未安装，LLM 功能不可用")
            return False

        return True

    def chat(
        self,
        messages: List[dict],
        model_name: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        调用 LLM 对话

        Args:
            messages: 消息列表 [{"role": "system/user/assistant", "content": "..."}]
            model_name: 指定模型名，不传则用默认模型
            temperature: 温度参数，不传则用模型配置默认值
            max_tokens: 最大 token 数，不传则用模型配置默认值

        Returns:
            LLM 回复文本

        Raises:
            LLMCallError: 调用失败
        """
        model = llm_config.get_model(model_name)
        if model is None or not model.is_available():
            raise LLMCallError(
                f"LLM 模型不可用: {model_name or llm_config.get_default_name() or '未配置'}"
            )

        client = self._get_client(model)

        # 构建请求参数
        kwargs = {
            "model": model.model_id,
            "messages": messages,
            "temperature": temperature if temperature is not None else model.temperature,
            "max_tokens": max_tokens if max_tokens is not None else model.max_tokens,
        }

        last_error = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                logger.debug(
                    f"LLM 请求 [尝试 {attempt}/{self.MAX_RETRIES}] "
                    f"模型={model.name}({model.model_id}) 消息数={len(messages)}"
                )

                response = client.chat.completions.create(**kwargs)

                content = response.choices[0].message.content
                if not content:
                    raise LLMCallError("LLM 返回空内容")

                # 记录 token 使用量
                if hasattr(response, "usage") and response.usage:
                    usage = response.usage
                    logger.debug(
                        f"LLM 响应成功 模型={model.name} "
                        f"tokens: prompt={usage.prompt_tokens} "
                        f"completion={usage.completion_tokens} "
                        f"total={usage.total_tokens}"
                    )

                return content.strip()

            except Exception as e:
                last_error = e
                logger.warning(f"LLM 请求失败 [尝试 {attempt}]: {e}")
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_DELAY * attempt)  # 递增退避

        raise LLMCallError(
            f"LLM 调用失败 (重试 {self.MAX_RETRIES} 次): {last_error}"
        )

    def chat_json(
        self,
        messages: List[dict],
        model_name: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """
        调用 LLM 并解析 JSON 输出

        在 messages 中自动追加指令要求返回 JSON 格式。
        尝试从回复中提取 JSON 并解析。

        Args:
            messages: 消息列表
            model_name: 指定模型名
            temperature: 温度参数 (结构化输出建议用低温度)
            max_tokens: 最大 token 数

        Returns:
            解析后的 dict

        Raises:
            LLMCallError: 调用失败或 JSON 解析失败
        """
        # 追加 JSON 格式要求 (不修改原始 messages)
        augmented = messages.copy()
        augmented.append({
            "role": "system",
            "content": (
                "请以合法的 JSON 格式回复，不要包含 markdown 代码块标记。"
                "只输出 JSON 内容本身。"
            ),
        })

        # 结构化输出使用较低温度
        temp = temperature if temperature is not None else 0.1

        raw = self.chat(augmented, model_name=model_name, temperature=temp, max_tokens=max_tokens)

        return self._extract_json(raw)

    @staticmethod
    def _extract_json(text: str) -> dict:
        """
        从文本中提取 JSON

        处理以下情况:
        - 纯 JSON 文本
        - 包含在 ```json ... ``` 代码块中的 JSON
        - 前后有额外文本的 JSON

        Returns:
            解析后的 dict

        Raises:
            LLMCallError: JSON 解析失败
        """
        text = text.strip()

        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试去除 markdown 代码块标记
        if "```" in text:
            # 提取代码块内容
            lines = text.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    if in_block:
                        break  # 代码块结束
                    in_block = True
                    continue
                if in_block:
                    json_lines.append(line)
            if json_lines:
                try:
                    return json.loads("\n".join(json_lines))
                except json.JSONDecodeError:
                    pass

        # 尝试找到第一个 { 和最后一个 } 之间的内容
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str = text[first_brace:last_brace + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        raise LLMCallError(f"无法从 LLM 回复中解析 JSON: {text[:200]}...")

    def clear_cache(self) -> None:
        """清空客户端缓存"""
        self._client_cache.clear()

    # ═══════════════════════════════════════════════════════
    # 流式调用 (v0.5.0)
    # ═══════════════════════════════════════════════════════

    def chat_stream(
        self,
        messages: List[dict],
        model_name: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        流式调用 LLM 对话，逐块 yield 文本

        Args:
            messages: 消息列表
            model_name: 指定模型名，不传则用默认模型
            temperature: 温度参数
            max_tokens: 最大 token 数

        Yields:
            str: 每次返回一个文本片段

        Raises:
            LLMCallError: 调用失败
        """
        model = llm_config.get_model(model_name)
        if model is None or not model.is_available():
            raise LLMCallError(
                f"LLM 模型不可用: {model_name or llm_config.get_default_name() or '未配置'}"
            )

        client = self._get_client(model)

        kwargs = {
            "model": model.model_id,
            "messages": messages,
            "temperature": temperature if temperature is not None else model.temperature,
            "max_tokens": max_tokens if max_tokens is not None else model.max_tokens,
            "stream": True,
        }

        logger.debug(
            f"LLM 流式请求 模型={model.name}({model.model_id}) 消息数={len(messages)}"
        )

        try:
            response = client.chat.completions.create(**kwargs)
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise LLMCallError(f"LLM 流式调用失败: {e}")


# 全局单例
llm_client = LLMClient()
