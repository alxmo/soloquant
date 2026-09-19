"""
模型同步 API 模块
=====================================================
云端 HTTP API 服务，接收本地训练好的模型并管理模型热加载
"""

from src.api.model_sync import create_app

__all__ = ["create_app"]
