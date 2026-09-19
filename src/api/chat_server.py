"""
智能助手聊天 API 服务 - FastAPI + SSE
=====================================================
v1.3.0 多会话支持:
- 会话历史持久化到 data/chat_sessions/ 目录 (每个会话一个 JSON 文件)
- 支持新建会话 / 切换会话 / 删除会话 / 重命名会话
- 自动从旧版 chat_history.json 迁移历史对话
- 切换会话时自动恢复 LLM 对话上下文

接口:
  GET  /                       → 返回聊天 HTML 页面
  GET  /api/sessions           → 会话列表 (按更新时间倒序)
  POST /api/sessions           → 新建会话
  GET  /api/sessions/{sid}     → 获取会话消息 (并恢复对话上下文)
  DELETE /api/sessions/{sid}   → 删除会话
  POST /api/sessions/{sid}/rename → 重命名会话
  GET  /api/history            → 加载对话历史 (兼容旧接口, ?session_id= 可选)
  POST /api/chat               → SSE 流式响应 (thinking + content)
  POST /api/clear              → 清空当前会话消息

启动:
  python -m src.api.chat_server
  或
  uvicorn src.api.chat_server:app --host 0.0.0.0 --port 8502
"""

import json
import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from loguru import logger

from src.utils.config import config

# ─── 路径常量 ───
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SESSIONS_DIR = PROJECT_ROOT / "data" / "chat_sessions"
LEGACY_HISTORY_FILE = PROJECT_ROOT / "data" / "chat_history.json"  # 旧版单文件历史
CHAT_HTML_FILE = PROJECT_ROOT / "src" / "ui" / "chat_page.html"

# ─── FastAPI 应用 ───

# i18n: 翻译函数 (CLI 模式, 读 SOLOQUANT_UI_LANG 环境变量)
from src.ui.i18n import t

app = FastAPI(title="AI 智能助手聊天服务", version="1.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── 请求模型 ───

class ChatRequest(BaseModel):
    message: str
    page_context: str = ""
    session_id: str = ""  # v1.3.0: 会话 ID (为空时自动新建会话)


class RenameRequest(BaseModel):
    title: str


class ClearRequest(BaseModel):
    session_id: str = ""


# ═══════════════════════════════════════════════════════
# v1.3.0: 多会话持久化
# ═══════════════════════════════════════════════════════

def _now_iso() -> str:
    """当前时间 ISO 字符串"""
    return datetime.now().isoformat(timespec="seconds")


def _session_path(session_id: str) -> Path:
    """会话文件路径 (session_id 仅允许 uuid 安全字符, 防路径穿越)"""
    safe = "".join(c for c in session_id if c.isalnum() or c in "-_")
    if not safe or safe != session_id:
        raise ValueError("非法会话 ID")
    return SESSIONS_DIR / f"{safe}.json"


def _migrate_legacy_history() -> None:
    """一次性迁移: 旧版 chat_history.json → 会话文件"""
    try:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        if not LEGACY_HISTORY_FILE.exists():
            return
        # 已有会话则不再迁移
        if any(SESSIONS_DIR.glob("*.json")):
            return
        with open(LEGACY_HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            # 用第一条用户消息作为标题
            title = ""
            for m in data:
                if m.get("role") == "user":
                    title = (m.get("content") or "")[:20]
                    break
            session = _new_session_obj(title or t("历史对话"))
            session["messages"] = data
            _save_session(session)
            logger.info(f"已迁移旧版对话历史到会话: {session['id']}")
    except Exception as e:
        logger.warning(f"迁移旧版对话历史失败 (忽略): {e}")


def _new_session_obj(title: str = "") -> dict:
    """构造新会话对象"""
    return {
        "id": uuid.uuid4().hex[:12],
        "title": title,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "messages": [],
    }


def _list_sessions() -> list:
    """列出所有会话 (按更新时间倒序), 返回摘要信息"""
    _migrate_legacy_history()
    sessions = []
    try:
        for f in SESSIONS_DIR.glob("*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    s = json.load(fp)
                if isinstance(s, dict) and "id" in s:
                    sessions.append({
                        "id": s["id"],
                        "title": s.get("title") or t("新会话"),
                        "created_at": s.get("created_at", ""),
                        "updated_at": s.get("updated_at", ""),
                        "message_count": len(s.get("messages", [])),
                    })
            except Exception as e:
                logger.warning(f"读取会话文件失败 {f.name}: {e}")
    except Exception as e:
        logger.warning(f"列出会话失败: {e}")
    # 按更新时间倒序
    sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
    return sessions


def _load_session(session_id: str) -> Optional[dict]:
    """加载单个会话 (含消息), 不存在返回 None"""
    try:
        p = _session_path(session_id)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                s = json.load(f)
            if isinstance(s, dict) and s.get("id") == session_id:
                return s
    except Exception as e:
        logger.warning(f"加载会话失败 {session_id}: {e}")
    return None


def _save_session(session: dict) -> None:
    """保存会话到文件"""
    try:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        session["updated_at"] = _now_iso()
        p = _session_path(session["id"])
        with open(p, "w", encoding="utf-8") as f:
            json.dump(session, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"保存会话失败: {e}")


def _delete_session(session_id: str) -> bool:
    """删除会话文件"""
    try:
        p = _session_path(session_id)
        if p.exists():
            p.unlink()
            return True
    except Exception as e:
        logger.warning(f"删除会话失败 {session_id}: {e}")
    return False


def _restore_dialogue_context(messages: list) -> None:
    """切换会话时, 将历史消息回放到 LLM 对话上下文 (dialogue_manager)"""
    try:
        from src.ui.nl_router import nl_router
        nl_router.load_session(messages)
    except Exception as e:
        logger.warning(f"恢复对话上下文失败: {e}")


# ─── 兼容旧接口: 单文件历史读写 (仅迁移用) ───

def _load_history() -> list:
    """从旧文件加载对话历史 (兼容)"""
    try:
        if LEGACY_HISTORY_FILE.exists():
            with open(LEGACY_HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
    except Exception as e:
        logger.warning(f"加载对话历史失败: {e}")
    return []


# ─── 接口: 返回 HTML 页面 ───

@app.get("/", response_class=HTMLResponse)
async def index():
    """返回聊天 HTML 页面"""
    try:
        if CHAT_HTML_FILE.exists():
            return HTMLResponse(content=CHAT_HTML_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"读取 HTML 页面失败: {e}")
    return HTMLResponse(content="<h1>" + t("聊天页面加载失败") + "</h1>", status_code=500)


# ─── 接口: 会话管理 (v1.3.0) ───

@app.get("/api/sessions")
async def get_sessions():
    """获取会话列表 (按更新时间倒序)"""
    return JSONResponse(content={"sessions": _list_sessions()})


@app.post("/api/sessions")
async def create_session():
    """新建会话, 并重置 LLM 对话上下文"""
    session = _new_session_obj()
    _save_session(session)
    # 新会话: 清空对话上下文
    try:
        from src.ui.nl_router import nl_router
        nl_router.reset_dialogue()
    except Exception as e:
        logger.warning(f"重置对话上下文失败: {e}")
    return JSONResponse(content={
        "id": session["id"],
        "title": session["title"],
        "created_at": session["created_at"],
        "updated_at": session["updated_at"],
        "messages": [],
    })


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """获取会话消息, 并将该会话历史回放到 LLM 对话上下文"""
    session = _load_session(session_id)
    if session is None:
        return JSONResponse(content={"error": t("会话不存在")}, status_code=404)
    # 恢复对话上下文 (前端仅在切换会话时调用此接口)
    _restore_dialogue_context(session.get("messages", []))
    return JSONResponse(content={
        "id": session["id"],
        "title": session.get("title", ""),
        "created_at": session.get("created_at", ""),
        "updated_at": session.get("updated_at", ""),
        "messages": session.get("messages", []),
    })


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    ok = _delete_session(session_id)
    if not ok:
        return JSONResponse(content={"error": t("会话不存在")}, status_code=404)
    return JSONResponse(content={"status": "ok", "message": t("会话已删除")})


@app.post("/api/sessions/{session_id}/rename")
async def rename_session(session_id: str, req: RenameRequest):
    """重命名会话"""
    session = _load_session(session_id)
    if session is None:
        return JSONResponse(content={"error": t("会话不存在")}, status_code=404)
    session["title"] = req.title.strip()[:50] or t("新会话")
    _save_session(session)
    return JSONResponse(content={"status": "ok", "title": session["title"]})


# ─── 接口: 获取对话历史 (兼容旧接口) ───

@app.get("/api/history")
async def get_history(session_id: str = ""):
    """加载对话历史 (兼容旧接口)
    - 指定 session_id: 返回该会话消息
    - 未指定: 返回最近一个会话的消息 (无会话时回退到旧版文件)
    """
    if session_id:
        session = _load_session(session_id)
        if session is None:
            return JSONResponse(content={"error": t("会话不存在")}, status_code=404)
        return JSONResponse(content={
            "session_id": session["id"],
            "messages": session.get("messages", []),
        })
    # 未指定: 取最近的会话
    sessions = _list_sessions()
    if sessions:
        session = _load_session(sessions[0]["id"])
        if session:
            return JSONResponse(content={
                "session_id": session["id"],
                "messages": session.get("messages", []),
            })
    # 回退: 旧版单文件
    return JSONResponse(content={"session_id": "", "messages": _load_history()})


# ─── 接口: SSE 流式聊天 ───

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """SSE 流式响应: 逐块推送 thinking + content"""
    message = req.message.strip()
    page_context = req.page_context  # v1.1.0: 提取页面上下文供线程使用
    if not message:
        return JSONResponse(content={"error": t("消息不能为空")}, status_code=400)

    # v1.3.0: 定位/创建会话
    session = None
    if req.session_id:
        session = _load_session(req.session_id)
    if session is None:
        session = _new_session_obj()
    session_id = session["id"]

    async def event_stream() -> AsyncGenerator[str, None]:
        """生成 SSE 事件流 (真正的逐块推送)"""
        import threading as _threading

        loop = asyncio.get_event_loop()

        # 使用 asyncio.Queue 替代 queue.Queue
        # asyncio.Queue 的 get() 是真正的异步等待，不会阻塞事件循环
        _q: asyncio.Queue = asyncio.Queue()
        _SENTINEL = object()
        _cancelled = _threading.Event()  # v1.1.0: 中断信号

        def _run_in_thread():
            """在后台线程中运行同步生成器，逐块放入队列"""
            try:
                from src.ui.nl_router import nl_router
                for chunk in nl_router.respond_stream(message, page_context):
                    if _cancelled.is_set():  # 用户中断
                        break
                    # 从线程安全地向 asyncio.Queue 放入数据
                    loop.call_soon_threadsafe(_q.put_nowait, chunk)
            except Exception as e:
                loop.call_soon_threadsafe(
                    _q.put_nowait, {"type": "error", "text": str(e)}
                )
            finally:
                loop.call_soon_threadsafe(_q.put_nowait, _SENTINEL)

        # 启动后台线程
        thread = _threading.Thread(target=_run_in_thread, daemon=True)
        thread.start()

        thinking_text = ""
        content_text = ""

        def _persist(partial: bool = False):
            """v1.3.0: 将本轮对话写入会话文件"""
            try:
                s = _load_session(session_id) or session
                s["messages"].append({"role": "user", "content": message})
                if content_text:
                    s["messages"].append({"role": "assistant", "content": content_text})
                # 标题为空时用首条用户消息生成
                if not s.get("title"):
                    s["title"] = message[:20]
                _save_session(s)
            except Exception as e:
                logger.warning(f"保存会话消息失败: {e}")

        try:
            while True:
                # await q.get() 是真正的异步等待
                # 队列为空时让出控制权，事件循环可以 flush 已写入的 SSE 数据
                item = await _q.get()

                if item is _SENTINEL:
                    break

                chunk_type = item.get("type", "content")
                chunk_text = item.get("text", "")

                if chunk_type == "thinking":
                    thinking_text += chunk_text
                    # 推送思考过程 (累积全文，前端替换显示)
                    yield f"data: {json.dumps({'type': 'thinking', 'text': thinking_text}, ensure_ascii=False)}\n\n"
                elif chunk_type == "content":
                    content_text += chunk_text
                    # 推送回复内容 (增量片段，前端追加)
                    yield f"data: {json.dumps({'type': 'content', 'text': chunk_text}, ensure_ascii=False)}\n\n"
                elif chunk_type == "suggestions":
                    # v2.0: 推送跟进建议按钮
                    suggestions = item.get("suggestions", [])
                    yield f"data: {json.dumps({'type': 'suggestions', 'suggestions': suggestions}, ensure_ascii=False)}\n\n"
                elif chunk_type == "card":
                    # v2.0: 推送富卡片数据
                    card_type = item.get("card_type", "")
                    card_data = item.get("data", {})
                    yield f"data: {json.dumps({'type': 'card', 'card_type': card_type, 'data': card_data}, ensure_ascii=False)}\n\n"
                elif chunk_type == "error":
                    # 推送错误事件
                    yield f"data: {json.dumps({'type': 'error', 'text': chunk_text}, ensure_ascii=False)}\n\n"

            # 推送完成信号 (v1.3.0: 附带 session_id, 前端首次发消息时用于绑定会话)
            yield f"data: {json.dumps({'type': 'done', 'thinking': thinking_text, 'content': content_text, 'session_id': session_id}, ensure_ascii=False)}\n\n"

            # 保存对话历史到会话
            _persist()

        except asyncio.CancelledError:
            # v1.1.0: 客户端中断连接，通知后台线程停止
            _cancelled.set()
            logger.info("客户端中断了流式响应")
            # 保存已生成的部分内容
            _persist(partial=True)
            raise  # 重新抛出以正确关闭 StreamingResponse
        except Exception as e:
            logger.error(f"聊天流式响应异常: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'text': t('抱歉，处理请求时出错: {v1}', v1=e)}, ensure_ascii=False)}\n\n"
        finally:
            # 确保线程结束
            if thread.is_alive():
                _cancelled.set()
                thread.join(timeout=1)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ─── 接口: 清空对话 ───

@app.post("/api/clear")
async def clear_history(req: ClearRequest = None):
    """清空当前会话的消息 (保留会话本身)"""
    session_id = (req.session_id if req else "") or ""

    # 重置 LLM 对话上下文
    try:
        from src.ui.nl_router import nl_router
        nl_router.reset_dialogue()
    except Exception as e:
        logger.warning(f"重置对话上下文失败: {e}")

    if session_id:
        # v1.3.0: 清空指定会话的消息
        session = _load_session(session_id)
        if session:
            session["messages"] = []
            _save_session(session)
            return JSONResponse(content={"status": "ok", "message": t("对话已清空")})
        return JSONResponse(content={"error": t("会话不存在")}, status_code=404)

    # 兼容旧逻辑: 无 session_id 时清空旧版文件
    try:
        LEGACY_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LEGACY_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
    except Exception as e:
        logger.warning(f"清空旧版历史失败: {e}")
    return JSONResponse(content={"status": "ok", "message": t("对话已清空")})


# ─── 健康检查 ───

@app.get("/api/health")
async def health():
    """健康检查"""
    return {"status": "ok", "service": "chat_server", "version": "1.3.0"}


# ─── 便捷启动 ───

if __name__ == "__main__":
    import uvicorn

    # 启动时执行一次旧数据迁移
    _migrate_legacy_history()

    port = int(config.CHAT_PORT) if hasattr(config, "CHAT_PORT") else 8502
    logger.info(f"启动智能助手聊天服务: http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
