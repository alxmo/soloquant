"""
模型训练仪表盘 - AI 量化交易系统
=====================================================
本地模型训练、多模型竞赛、因子搜索、训练历史管理

功能页面:
1. 📦 已训练模型列表 (扫描 data/models/ 目录)
2. 🚀 训练新模型 (LightGBM / LSTM / Linear)
3. ⚔️ 多模型竞赛 (同时训练 + IC 对比)
4. 🔬 因子搜索 (Alpha158 因子 IC/IR 排名)
5. 📜 训练历史记录 (MLflow + 模型文件元数据)

启动: 通过主仪表盘侧边栏「🧠 模型训练」进入
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime


# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# i18n: 翻译函数 (必须在 sys.path 设置之后导入)
from src.ui.i18n import t

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ═══════════════════════════════════════════════════════
# 延迟导入
# ═══════════════════════════════════════════════════════

_config = None
_model_manager = None
_factor_searcher = None
_strategy_manager = None


def _get_config():
    """延迟获取 config"""
    global _config
    if _config is None:
        from src.utils.config import config
        _config = config
    return _config


def _get_model_manager():
    """延迟获取 model_manager"""
    global _model_manager
    if _model_manager is None:
        from src.engine.model_manager import model_manager
        _model_manager = model_manager
    return _model_manager


def _get_factor_searcher():
    """延迟获取 factor_searcher"""
    global _factor_searcher
    if _factor_searcher is None:
        from src.engine.factor_search import FactorSearcher
        _factor_searcher = FactorSearcher()
    return _factor_searcher


def _get_strategy_manager():
    """延迟获取 strategy_manager"""
    global _strategy_manager
    if _strategy_manager is None:
        from src.engine.strategy_manager import strategy_manager
        _strategy_manager = strategy_manager
    return _strategy_manager


# ═══════════════════════════════════════════════════════
# 数据扫描函数
# ═══════════════════════════════════════════════════════

@st.cache_data(ttl=10)
def scan_model_files():
    """扫描 data/models/ 目录下所有模型文件

    返回模型文件列表，包含文件名、类型、大小、创建时间等元数据
    """
    config = _get_config()
    models_dir = config.DATA_DIR / "models"
    if not models_dir.exists():
        return []

    files = []
    for f in sorted(models_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if f.is_file():
            stat = f.stat()
            # 判断模型类型
            if f.suffix == ".txt":
                model_type = "LightGBM"
            elif f.suffix == ".pth":
                model_type = "LSTM"
            elif f.suffix == ".json":
                model_type = "元数据"
            else:
                continue

            files.append({
                "文件名": f.name,
                "模型类型": model_type,
                "大小 (KB)": round(stat.st_size / 1024, 1),
                "修改时间": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                t("路径"): str(f),
            })
    return files


@st.cache_data(ttl=10)
def get_best_model_info():
    """读取 best_model.json 获取当前激活模型信息"""
    config = _get_config()
    marker = config.DATA_DIR / "models" / "best_model.json"
    if not marker.exists():
        return None
    try:
        with open(marker, "r") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(ttl=30)
def get_mlflow_runs():
    """扫描 mlruns/ 目录获取训练历史记录

    解析 MLflow 的目录结构提取 run 信息
    """
    mlruns_dir = PROJECT_ROOT / "mlruns"
    if not mlruns_dir.exists():
        return []

    runs = []
    try:
        # MLflow 目录结构: mlruns/<experiment_id>/<run_id>/
        for exp_dir in mlruns_dir.iterdir():
            if not exp_dir.is_dir() or exp_dir.name == ".trash":
                continue
            for run_dir in exp_dir.iterdir():
                if not run_dir.is_dir():
                    continue

                run_info = {
                    "实验ID": exp_dir.name,
                    "Run ID": run_dir.name,
                    "创建时间": "",
                    "状态": "",
                    "指标": {},
                }

                # 读取 meta.yaml
                meta_file = run_dir / "meta.yaml"
                if meta_file.exists():
                    try:
                        import yaml
                        with open(meta_file, "r") as f:
                            meta = yaml.safe_load(f)
                        run_info["创建时间"] = meta.get("creation_time", "")
                        run_info["状态"] = meta.get("status", "")
                    except Exception:
                        pass

                # 读取 metrics
                metrics_dir = run_dir / "metrics"
                if metrics_dir.exists():
                    for metric_file in metrics_dir.iterdir():
                        try:
                            with open(metric_file, "r") as f:
                                lines = f.read().strip().split("\n")
                                if lines:
                                    # 格式: timestamp value step
                                    parts = lines[-1].split()
                                    if len(parts) >= 2:
                                        run_info["指标"][metric_file.name] = float(parts[1])
                        except Exception:
                            pass

                # 读取 params
                params_dir = run_dir / "params"
                if params_dir.exists():
                    for param_file in params_dir.iterdir():
                        try:
                            with open(param_file, "r") as f:
                                run_info["指标"][f"param_{param_file.name}"] = f.read().strip()
                        except Exception:
                            pass

                if run_info["创建时间"] or run_info["指标"]:
                    runs.append(run_info)
    except Exception:
        pass

    return runs


# ═══════════════════════════════════════════════════════
# 页面主函数
# ═══════════════════════════════════════════════════════

def page_model_training():
    """模型训练仪表盘主页面"""
    st.title("🧠 模型训练")
    st.caption(t("本地模型训练 · 多模型竞赛 · 因子搜索 · 训练历史管理"))

    # 子页面选择
    tab_names = [
        t("📦 已训练模型"),
        t("🚀 训练新模型"),
        t("⚔️ 多模型竞赛"),
        t("🔬 因子搜索"),
        t("📜 训练历史"),
    ]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        _render_model_list()
    with tabs[1]:
        _render_train_single()
    with tabs[2]:
        _render_train_competition()
    with tabs[3]:
        _render_factor_search()
    with tabs[4]:
        _render_training_history()


# ═══════════════════════════════════════════════════════
# Tab 1: 已训练模型列表
# ═══════════════════════════════════════════════════════

def _render_model_list():
    """展示已训练模型文件列表"""
    st.subheader(t("📦 已训练模型"))
    st.caption(t("扫描 data/models/ 目录，展示所有已保存的模型文件"))

    # 当前最佳模型
    best_model = get_best_model_info()
    if best_model:
        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        col_b1.metric(t("🏆 当前最佳模型类型"), best_model.get("model_type", "N/A"))
        col_b2.metric(t("策略ID"), best_model.get("strategy_id", "N/A"))
        col_b3.metric(t("年化收益"), f"{best_model.get('annual_return', 0):.1%}" if best_model.get("annual_return") else "N/A")
        col_b4.metric(t("夏普比率"), f"{best_model.get('sharpe_ratio', 0):.2f}" if best_model.get("sharpe_ratio") else "N/A")
    else:
        st.info(t("💡 尚未设置最佳模型。训练完成后可标记最佳模型。"))

    st.divider()

    # 模型文件列表
    model_files = scan_model_files()
    if not model_files:
        st.info(t("📭 暂无模型文件。前往「训练新模型」标签页开始训练。"))
        return

    # 统计卡片
    lgb_count = sum(1 for f in model_files if f["模型类型"] == "LightGBM")
    lstm_count = sum(1 for f in model_files if f["模型类型"] == "LSTM")
    total_size = sum(f["大小 (KB)"] for f in model_files)

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric(t("模型总数"), str(len(model_files)))
    col_s2.metric("LightGBM", str(lgb_count))
    col_s3.metric("LSTM", str(lstm_count))
    col_s4.metric(t("总大小"), f"{total_size / 1024:.1f} MB")

    st.divider()

    # 模型文件表格
    df_models = pd.DataFrame(model_files)

    # 标记最佳模型
    if best_model:
        best_strategy_id = best_model.get("strategy_id", "")
        df_models[t("最佳模型")] = df_models["文件名"].apply(
            lambda x: "🏆" if best_strategy_id and best_strategy_id in x else ""
        )

    st.dataframe(df_models, width='stretch', hide_index=True)

    # 模型类型分布饼图
    st.subheader(t("📊 模型类型分布"))
    type_counts = df_models["模型类型"].value_counts()
    fig_pie = px.pie(
        values=type_counts.values,
        names=type_counts.index,
        title=t("模型类型占比"),
        hole=0.4,
    )
    fig_pie.update_layout(height=300)
    st.plotly_chart(fig_pie, width='stretch')

    # 操作区: 设置最佳模型 / 删除模型
    st.divider()
    st.subheader(t("⚙️ 模型操作"))

    col_op1, col_op2 = st.columns(2)
    with col_op1:
        st.markdown(t("**🏆 设置最佳模型**"))
        model_names = [f["文件名"] for f in model_files if f["模型类型"] in ("LightGBM", "LSTM")]
        if model_names:
            selected_model = st.selectbox(t("选择模型"), model_names, key="select_best_model")
            if st.button(t("设为最佳模型"), type="primary", key="btn_set_best"):
                _set_best_model(selected_model, best_model)
        else:
            st.info(t("无可设置的模型文件"))

    with col_op2:
        st.markdown(t("**🗑️ 删除模型文件**"))
        if model_names:
            del_model = st.selectbox(t("选择要删除的模型"), model_names, key="select_del_model")
            if st.button(t("删除"), key="btn_del_model"):
                _delete_model_file(del_model)
        else:
            st.info(t("无可删除的模型文件"))


def _set_best_model(model_name: str, current_best: dict):
    """设置最佳模型"""
    config = _get_config()
    marker = config.DATA_DIR / "models" / "best_model.json"

    # 判断模型类型
    if model_name.endswith(".txt"):
        model_type = "lightgbm"
    elif model_name.endswith(".pth"):
        model_type = "lstm"
    else:
        model_type = "unknown"

    # 保留原有字段，更新模型信息
    info = current_best or {}
    info["model_type"] = model_type
    info["model_file"] = model_name
    info["updated_at"] = datetime.now().isoformat()

    try:
        with open(marker, "w") as f:
            json.dump(info, f, indent=2)
        st.success(t("✅ 已将 '{v1}' 设为最佳模型!", v1=model_name)        st.cache_data.clear()
        st.rerun()
    except Exception as e:
        st.error(t('设置失败: {v1}', v1=e)

def _delete_model_file(model_name: str):
    """删除模型文件"""
    config = _get_config()
    model_path = config.DATA_DIR / "models" / model_name
    try:
        model_path.unlink()
        st.success(t('✅ 已删除模型文件: {v1}', v1=model_name)        st.cache_data.clear()
        st.rerun()
    except Exception as e:
        st.error(t('删除失败: {v1}', v1=e)

# ═══════════════════════════════════════════════════════
# Tab 2: 训练新模型
# ═══════════════════════════════════════════════════════

def _render_train_single():
    """训练单个模型"""
    st.subheader(t("🚀 训练新模型"))
    st.caption(t("选择模型类型、配置参数，一键启动本地训练"))

    # 训练配置
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown(t("#### 基本配置"))
        model_type = st.selectbox(
            "模型类型",
            ["lightgbm", "lstm", "linear"],
            format_func=lambda x: {"lightgbm": t("LightGBM (梯度提升树)"), "lstm": t("LSTM (深度学习)"), "linear": t("Linear (线性回归)")}.get(x, x),
            key="train_model_type",
        )

        # 股票池输入
        default_pool = "AAPL, MSFT, GOOGL, AMZN, NVDA"
        stock_pool_str = st.text_area(
            t("股票池 (逗号分隔)"),
            value=default_pool,
            height=80,
            key="train_stock_pool",
        )
        stock_pool = [s.strip().upper() for s in stock_pool_str.split(",") if s.strip()]

        # 训练日期范围
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            train_start = st.date_input(t("训练开始日期"), value=datetime(2023, 1, 1), key="train_start")
        with col_d2:
            train_end = st.date_input(t("训练结束日期"), value=datetime(2024, 12, 31), key="train_end")

    with col_c2:
        st.markdown(t("#### 超参数配置"))

        if model_type == "lightgbm":
            lgb_params = {
                "learning_rate": st.slider(t("学习率"), 0.001, 0.5, 0.05, 0.001, key="p_lr"),
                "max_depth": st.slider(t("最大深度"), 3, 15, 8, key="p_depth"),
                "num_leaves": st.slider(t("叶子数"), 15, 511, 31, key="p_leaves"),
                "subsample": st.slider(t("子采样"), 0.5, 1.0, 0.8, 0.05, key="p_subsample"),
                "colsample_bytree": st.slider(t("列采样"), 0.5, 1.0, 0.8, 0.05, key="p_colsample"),
            }
            custom_params = lgb_params
        elif model_type == "lstm":
            lstm_params = {
                "hidden_size": st.slider(t("隐藏层维度"), 16, 256, 64, 16, key="p_hidden"),
                "num_layers": st.slider(t("LSTM层数"), 1, 5, 2, key="p_layers"),
                "dropout": st.slider("Dropout", 0.0, 0.5, 0.1, 0.05, key="p_dropout"),
                "n_epochs": st.slider(t("训练轮数"), 10, 200, 50, 10, key="p_epochs"),
                "lr": st.select_slider(t("学习率"), [1e-4, 5e-4, 1e-3, 5e-3, 1e-2], value=1e-3, key="p_lstm_lr"),
                "lookback": st.slider(t("回看窗口"), 5, 60, 20, 5, key="p_lookback"),
                "batch_size": st.select_slider(t("批次大小"), [16, 32, 64, 128], value=32, key="p_batch"),
            }
            custom_params = lstm_params
        else:
            st.info(t("Linear 模型无需超参数配置"))
            custom_params = {}

    # 环境检查
    st.divider()
    st.markdown(t("#### 🔍 环境检查"))

    col_e1, col_e2, col_e3 = st.columns(3)

    # 检查 Qlib (尝试初始化一次)
    try:
        from src.engine.qlib_wrapper import qlib_wrapper
        if not qlib_wrapper.available:
            qlib_wrapper.init()  # 尝试初始化
        qlib_ok = qlib_wrapper.available
    except Exception:
        qlib_ok = False
    col_e1.metric("Qlib", t("✅ 可用") if qlib_ok else t("⚠️ 不可用 (降级模式)"))

    # 检查 PyTorch (LSTM 需要)
    if model_type == "lstm":
        try:
            from src.engine.lstm_model import is_lstm_available
            torch_ok = is_lstm_available()
        except Exception:
            torch_ok = False
        col_e2.metric("PyTorch", t("✅ 可用") if torch_ok else t("❌ 未安装"))
    else:
        col_e2.metric("PyTorch", t("— (非必需)"))

    # 检查 LightGBM
    try:
        import lightgbm
        lgb_ok = True
    except ImportError:
        lgb_ok = False
    col_e3.metric("LightGBM", t("✅ 可用") if lgb_ok else t("❌ 未安装"))

    # 训练执行
    st.divider()

    if st.button(t("🚀 开始训练"), type="primary", key="btn_start_train"):
        if not stock_pool:
            st.error(t("请输入至少一只股票"))
            return

        if model_type == "lstm":
            try:
                from src.engine.lstm_model import is_lstm_available
                if not is_lstm_available():
                    st.error(t("❌ PyTorch 未安装，无法训练 LSTM 模型。请运行: pip install torch"))
                    return
            except Exception:
                st.error(t("❌ 无法检查 PyTorch 状态"))
                return

        # 执行训练
        with st.spinner(t('正在训练 {v1} 模型... (可能需要 1-5 分钟)', v1=model_type.upper())            try:
                mm = _get_model_manager()
                result = mm.train_model(
                    model_type=model_type,
                    factors=[],  # 使用默认因子集
                    stock_pool=stock_pool,
                    train_start=train_start.strftime("%Y-%m-%d"),
                    train_end=train_end.strftime("%Y-%m-%d"),
                    params=custom_params if custom_params else None,
                )

                # 保存结果到 session_state
                st.session_state["last_train_result"] = result
                st.session_state["last_train_model_type"] = model_type

                if result.model_path or result.valid_ic != 0:
                    st.success(t('✅ {v1} 训练完成!', v1=model_type.upper())                else:
                    st.warning(t('⚠️ {v1} 训练完成，但可能未产生有效结果 (数据不足?)', v1=model_type.upper())
            except Exception as e:
                st.error(t('训练失败: {v1}', v1=e)                import traceback
                st.text(traceback.format_exc())

    # 展示训练结果
    result = st.session_state.get("last_train_result")
    if result:
        st.divider()
        st.subheader(t("📊 训练结果"))

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        col_r1.metric("模型类型", result.model_type.upper())
        col_r2.metric(t("训练集 IC"), f"{result.train_ic:.4f}")
        col_r3.metric(t("验证集 IC"), f"{result.valid_ic:.4f}")
        col_r4.metric(t("验证集 Rank IC"), f"{result.valid_rank_ic:.4f}")

        # IC 对比柱状图
        fig_ic = go.Figure()
        fig_ic.add_trace(go.Bar(
            x=[t("训练集 IC"), t("验证集 IC"), t("训练集 Rank IC"), t("验证集 Rank IC")],
            y=[result.train_ic, result.valid_ic, result.train_rank_ic, result.valid_rank_ic],
            marker_color=["blue", "green", "lightblue", "lightgreen"],
            text=[f"{v:.4f}" for v in [result.train_ic, result.valid_ic, result.train_rank_ic, result.valid_rank_ic]],
            textposition="auto",
        ))
        fig_ic.update_layout(
            title=t("IC / Rank IC 对比"),
            yaxis_title=t("IC 值"),
            height=350,
        )
        st.plotly_chart(fig_ic, width='stretch')

        # 模型路径和参数
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown(t("**模型文件路径:**"))
            if result.model_path:
                st.code(result.model_path)
            else:
                st.info(t("未保存模型文件"))
        with col_p2:
            st.markdown(t("**训练参数:**"))
            if result.params:
                params_str = "\n".join(f"{k}: {v}" for k, v in result.params.items())
                st.code(params_str)
            else:
                st.info(t("使用默认参数"))

        # 设为最佳模型按钮
        if result.model_path:
            if st.button(t("🏆 将此模型设为最佳模型"), key="btn_set_best_result"):
                model_filename = Path(result.model_path).name
                _set_best_model(model_filename, get_best_model_info())


# ═══════════════════════════════════════════════════════
# Tab 3: 多模型竞赛
# ═══════════════════════════════════════════════════════

def _render_train_competition():
    """多模型竞赛"""
    st.subheader(t("⚔️ 多模型竞赛"))
    st.caption(t("同时训练多个模型，对比 IC/Rank IC，自动选出最佳模型"))

    # 配置区
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown(t("#### 竞赛配置"))

        # 模型选择
        selected_models = st.multiselect(
            t("参与竞赛的模型"),
            ["lightgbm", "lstm", "linear"],
            default=["lightgbm", "linear"],
            format_func=lambda x: {"lightgbm": "LightGBM", "lstm": "LSTM", "linear": "Linear"}.get(x, x),
            key="competition_models",
        )

        # 股票池
        default_pool = "AAPL, MSFT, GOOGL, AMZN, NVDA"
        comp_pool_str = st.text_area(
            t("股票池 (逗号分隔)"),
            value=default_pool,
            height=80,
            key="comp_stock_pool",
        )
        comp_pool = [s.strip().upper() for s in comp_pool_str.split(",") if s.strip()]

        # 日期范围
        col_cd1, col_cd2 = st.columns(2)
        with col_cd1:
            comp_start = st.date_input(t("训练开始日期"), value=datetime(2023, 1, 1), key="comp_start")
        with col_cd2:
            comp_end = st.date_input(t("训练结束日期"), value=datetime(2024, 12, 31), key="comp_end")

    with col_c2:
        st.markdown(t("#### 竞赛说明"))
        st.info(t("\n        **多模型竞赛流程:**\n        1. 同时训练选中的模型\n        2. 每个模型使用相同的数据和因子\n        3. 比较验证集 IC / Rank IC\n        4. 自动选出 |IC| 最大的模型\n\n        **模型特点:**\n        - **LightGBM**: 速度快，适合表格数据\n        - **LSTM**: 捕捉时序依赖，需 PyTorch\n        - **Linear**: 最简基线，快速对比\n\n        ⚠️ LSTM 训练时间较长 (5-15分钟)\n        "))

    # 执行竞赛
    st.divider()

    if st.button(t("⚔️ 开始竞赛"), type="primary", key="btn_start_competition"):
        if not selected_models:
            st.error(t("请至少选择一个模型"))
            return
        if not comp_pool:
            st.error(t("请输入股票池"))
            return

        # 检查 LSTM 依赖
        if "lstm" in selected_models:
            try:
                from src.engine.lstm_model import is_lstm_available
                if not is_lstm_available():
                    st.warning(t("⚠️ PyTorch 未安装，LSTM 将跳过。其他模型继续训练。"))
                    selected_models = [m for m in selected_models if m != "lstm"]
                    if not selected_models:
                        st.error(t("没有可训练的模型"))
                        return
            except Exception:
                st.warning(t("⚠️ 无法检查 PyTorch，LSTM 可能失败"))

        # 执行竞赛
        progress = st.progress(0, text=t("准备开始竞赛..."))
        status_text = st.empty()

        try:
            mm = _get_model_manager()
            total = len(selected_models)

            # 逐个训练 (展示进度)
            results = {}
            for i, mt in enumerate(selected_models):
                status_text.info(t('正在训练 {v1} ({v2}/{v3})...', v1=mt.upper(), v2=i+1, v3=total)                progress.progress((i / total), text=t('训练 {v1} ({v2}/{v3})', v1=mt.upper(), v2=i+1, v3=total)
                try:
                    result = mm.train_model(
                        model_type=mt,
                        factors=[],
                        stock_pool=comp_pool,
                        train_start=comp_start.strftime("%Y-%m-%d"),
                        train_end=comp_end.strftime("%Y-%m-%d"),
                    )
                    results[mt] = result
                except Exception as e:
                    st.error(t('{v1} 训练失败: {v2}', v1=mt, v2=e)                    from src.models import ModelResult
                    results[mt] = ModelResult(model_type=mt)

            progress.progress(1.0, text=t("竞赛完成!"))
            status_text.success(t("✅ 多模型竞赛完成!"))

            # 保存结果
            st.session_state["competition_results"] = results

        except Exception as e:
            st.error(t('竞赛失败: {v1}', v1=e)            import traceback
            st.text(traceback.format_exc())

    # 展示竞赛结果
    comp_results = st.session_state.get("competition_results")
    if comp_results:
        st.divider()
        st.subheader(t("📊 竞赛结果对比"))

        # 找出最佳模型
        best_model_key = max(comp_results.keys(), key=lambda k: abs(comp_results[k].valid_ic))
        best_result = comp_results[best_model_key]

        # 汇总指标
        col_b1, col_b2, col_b3 = st.columns(3)
        col_b1.metric(t("🏆 最佳模型"), best_model_key.upper())
        col_b2.metric(t("最佳验证 IC"), f"{best_result.valid_ic:.4f}")
        col_b3.metric(t("最佳 Rank IC"), f"{best_result.valid_rank_ic:.4f}")

        st.divider()

        # 结果表格
        comp_data = []
        for mt, result in comp_results.items():
            comp_data.append({
                t("模型"): mt.upper(),
                t("训练 IC"): f"{result.train_ic:.4f}",
                t("验证 IC"): f"{result.valid_ic:.4f}",
                t("训练 Rank IC"): f"{result.train_rank_ic:.4f}",
                t("验证 Rank IC"): f"{result.valid_rank_ic:.4f}",
                t("模型路径"): result.model_path or t("未保存"),
                t("是否最佳"): "🏆" if mt == best_model_key else "",
            })
        df_comp = pd.DataFrame(comp_data)
        st.dataframe(df_comp, width='stretch', hide_index=True)

        # IC 对比柱状图
        st.subheader(t("📈 IC 对比图"))
        fig_comp = go.Figure()

        models_list = list(comp_results.keys())
        train_ics = [comp_results[mt].train_ic for mt in models_list]
        valid_ics = [comp_results[mt].valid_ic for mt in models_list]
        train_rank_ics = [comp_results[mt].train_rank_ic for mt in models_list]
        valid_rank_ics = [comp_results[mt].valid_rank_ic for mt in models_list]

        fig_comp.add_trace(go.Bar(name=t("训练 IC"), x=models_list, y=train_ics, marker_color="blue"))
        fig_comp.add_trace(go.Bar(name=t("验证 IC"), x=models_list, y=valid_ics, marker_color="green"))
        fig_comp.add_trace(go.Bar(name=t("训练 Rank IC"), x=models_list, y=train_rank_ics, marker_color="lightblue"))
        fig_comp.add_trace(go.Bar(name=t("验证 Rank IC"), x=models_list, y=valid_rank_ics, marker_color="lightgreen"))

        fig_comp.update_layout(
            title=t("多模型 IC / Rank IC 对比"),
            xaxis_title=t("模型"),
            yaxis_title=t("IC 值"),
            barmode="group",
            height=400,
        )
        st.plotly_chart(fig_comp, width='stretch')

        # 雷达图对比
        if len(models_list) >= 2:
            st.subheader(t("🎯 模型能力雷达图"))
            categories = [t("训练 IC"), t("验证 IC"), t("训练 Rank IC"), t("验证 Rank IC")]
            fig_radar = go.Figure()
            for mt in models_list:
                r = comp_results[mt]
                values = [abs(r.train_ic), abs(r.valid_ic), abs(r.train_rank_ic), abs(r.valid_rank_ic)]
                fig_radar.add_trace(go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=mt.upper(),
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                title=t("模型能力对比 (绝对值)"),
                height=400,
            )
            st.plotly_chart(fig_radar, width='stretch')

        # 将最佳模型设为全局最佳
        if best_result.model_path:
            st.divider()
            if st.button(t("🏆 将最佳模型设为全局最佳"), type="primary", key="btn_set_best_comp"):
                model_filename = Path(best_result.model_path).name
                _set_best_model(model_filename, get_best_model_info())


# ═══════════════════════════════════════════════════════
# Tab 4: 因子搜索
# ═══════════════════════════════════════════════════════

def _render_factor_search():
    """因子搜索"""
    st.subheader(t("🔬 因子搜索"))
    st.caption(t("使用 Alpha158 因子库，自动搜索有效因子并计算 IC/IR"))

    # 配置区
    col_f1, col_f2 = st.columns(2)

    with col_f1:
        st.markdown(t("#### 搜索配置"))
        factor_pool_str = st.text_area(
            t("股票池 (逗号分隔)"),
            value="AAPL, MSFT, GOOGL, AMZN, NVDA",
            height=80,
            key="factor_pool",
        )
        factor_pool = [s.strip().upper() for s in factor_pool_str.split(",") if s.strip()]

        col_fd1, col_fd2 = st.columns(2)
        with col_fd1:
            factor_start = st.date_input(t("开始日期"), value=datetime(2023, 1, 1), key="factor_start")
        with col_fd2:
            factor_end = st.date_input(t("结束日期"), value=datetime(2024, 12, 31), key="factor_end")

        top_n = st.slider(t("返回 Top-N 因子"), 5, 50, 20, key="factor_top_n")
        ic_threshold = st.slider(t("IC 阈值"), 0.0, 0.1, 0.03, 0.005, key="factor_ic_threshold")
        corr_threshold = st.slider(t("因子相关性阈值"), 0.3, 0.95, 0.7, 0.05, key="factor_corr_threshold")

    with col_f2:
        st.markdown(t("#### 因子库说明"))
        st.info(t("\n        **Alpha158 因子库** 包含以下类别:\n\n        - **价格类**: KMID, KLEN, KUPP, KLOW 等\n        - **动量类**: ROC, MA, STD, BETA 等\n        - **成交量类**: VSTD, WVMA, VSUMP 等\n        - **技术指标类**: RSI, KDJ, MACD, CCI 等\n\n        **筛选流程:**\n        1. 计算每个因子的 IC / Rank IC / IR\n        2. 按 |IR| 排序\n        3. 去相关 (因子间相关性 < 阈值)\n        4. 返回 Top-N 有效因子\n\n        ⚠️ 因子搜索需要 Qlib 数据支持\n        "))

    # 环境检查
    st.divider()
    try:
        from src.engine.qlib_wrapper import qlib_wrapper
        qlib_ok = qlib_wrapper.available
    except Exception:
        qlib_ok = False

    if not qlib_ok:
        st.warning(t("⚠️ Qlib 未初始化，因子搜索可能无法正常工作。请确保已运行数据初始化。"))

    # 执行搜索
    if st.button(t("🔍 开始因子搜索"), type="primary", key="btn_factor_search"):
        if not factor_pool:
            st.error(t("请输入股票池"))
            return

        with st.spinner(t("正在搜索有效因子... (可能需要 1-3 分钟)")):
            try:
                fs = _get_factor_searcher()
                results = fs.search(
                    stock_pool=factor_pool,
                    start=factor_start.strftime("%Y-%m-%d"),
                    end=factor_end.strftime("%Y-%m-%d"),
                    top_n=top_n,
                    ic_threshold=ic_threshold,
                    correlation_threshold=corr_threshold,
                )
                st.session_state["factor_results"] = results

                if results:
                    st.success(t('✅ 找到 {v1} 个有效因子!', v1=len(results))                else:
                    st.warning(t("未找到符合条件的因子，尝试调整阈值"))

            except Exception as e:
                st.error(t('因子搜索失败: {v1}', v1=e)                import traceback
                st.text(traceback.format_exc())

    # 展示因子搜索结果
    factor_results = st.session_state.get("factor_results")
    if factor_results:
        st.divider()
        st.subheader(t("📊 因子搜索结果"))

        # 因子表格
        factor_data = []
        for fr in factor_results:
            factor_data.append({
                t("因子名称"): fr.name,
                "IC": f"{fr.ic:.4f}",
                "Rank IC": f"{fr.rank_ic:.4f}",
                t("IR (信息比率)"): f"{fr.ir:.4f}",
                t("是否有效"): "✅" if fr.is_effective else "❌",
            })
        df_factors = pd.DataFrame(factor_data)
        st.dataframe(df_factors, width='stretch', hide_index=True)

        # IC 排名柱状图
        st.subheader(t("📈 因子 IC 排名"))
        fig_factor = go.Figure()
        colors = ["green" if fr.ic >= 0 else "red" for fr in factor_results]
        fig_factor.add_trace(go.Bar(
            x=[fr.name for fr in factor_results],
            y=[fr.ic for fr in factor_results],
            marker_color=colors,
            text=[f"{fr.ic:.4f}" for fr in factor_results],
            textposition="auto",
        ))
        fig_factor.update_layout(
            title=t("因子 IC 排名 (信息系数)"),
            xaxis_title=t("因子"),
            yaxis_title=t("IC 值"),
            height=400,
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_factor, width='stretch')

        # IR 排名柱状图
        st.subheader(t("📈 因子 IR 排名"))
        fig_ir = go.Figure()
        fig_ir.add_trace(go.Bar(
            x=[fr.name for fr in factor_results],
            y=[fr.ir for fr in factor_results],
            marker_color="royalblue",
            text=[f"{fr.ir:.4f}" for fr in factor_results],
            textposition="auto",
        ))
        fig_ir.update_layout(
            title=t("因子 IR 排名 (信息比率 = IC均值/IC标准差)"),
            xaxis_title=t("因子"),
            yaxis_title=t("IR 值"),
            height=350,
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_ir, width='stretch')

        # 使用因子创建策略
        st.divider()
        st.subheader(t("💡 后续操作"))
        effective_factors = [fr.name for fr in factor_results if fr.is_effective]
        if effective_factors:
            st.info(t('发现 {v1} 个有效因子，可前往「📈 策略管理」创建使用这些因子的策略', v1=len(effective_factors))            st.code(t('有效因子: {v1}', v1=', '.join(effective_factors[:10]))        else:
            st.info(t("未发现有效因子，尝试降低 IC 阈值后重新搜索"))


# ═══════════════════════════════════════════════════════
# Tab 5: 训练历史
# ═══════════════════════════════════════════════════════

def _render_training_history():
    """训练历史记录"""
    st.subheader(t("📜 训练历史"))
    st.caption(t("从 MLflow (mlruns/) 和模型文件中提取历史训练记录"))

    # 获取 MLflow 记录
    mlflow_runs = get_mlflow_runs()

    # 获取模型文件列表
    model_files = scan_model_files()

    col_h1, col_h2 = st.columns(2)
    col_h1.metric(t("MLflow 记录数"), str(len(mlflow_runs)))
    col_h2.metric(t("模型文件数"), str(len([f for f in model_files if f["模型类型"] != "元数据"])))

    st.divider()

    # MLflow 记录
    if mlflow_runs:
        st.subheader(t("📋 MLflow 训练记录"))

        # 提取关键指标到表格
        history_data = []
        for run in mlflow_runs:
            row = {
                "实验ID": run["实验ID"],
                "Run ID": run["Run ID"][:12] + "...",
                "创建时间": run["创建时间"],
                "状态": run["状态"],
            }
            # 添加关键指标
            metrics = run.get("指标", {})
            for key in ["IC", "valid_ic", "train_ic", "rank_ic", "valid_rank_ic"]:
                if key in metrics:
                    row[key] = f"{metrics[key]:.4f}"
            # 添加参数
            for key in ["param_model_type", "param_learning_rate"]:
                if key in metrics:
                    row[key.replace("param_", "")] = metrics[key]

            history_data.append(row)

        df_history = pd.DataFrame(history_data)
        st.dataframe(df_history, width='stretch', hide_index=True)

        # 如果有 IC 指标，绘制趋势图
        ic_runs = [r for r in mlflow_runs if "valid_ic" in r.get("指标", {})]
        if ic_runs:
            st.subheader(t("📈 验证集 IC 趋势"))
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(
                x=list(range(len(ic_runs))),
                y=[r["指标"]["valid_ic"] for r in ic_runs],
                mode="lines+markers",
                name=t("验证集 IC"),
                line=dict(color="green", width=2),
            ))
            fig_hist.update_layout(
                title=t("验证集 IC 历史趋势"),
                xaxis_title=t("训练次数"),
                yaxis_title=t("IC 值"),
                height=300,
            )
            st.plotly_chart(fig_hist, width='stretch')

    else:
        st.info(t("📭 未找到 MLflow 训练记录 (mlruns/ 目录为空)"))

    st.divider()

    # 模型文件时间线
    if model_files:
        st.subheader(t("📅 模型文件时间线"))

        # 过滤非元数据文件
        model_files_only = [f for f in model_files if f["模型类型"] != "元数据"]

        if model_files_only:
            # 按时间排序
            df_timeline = pd.DataFrame(model_files_only)
            df_timeline["修改时间"] = pd.to_datetime(df_timeline["修改时间"])
            df_timeline = df_timeline.sort_values("修改时间")

            # 时间线散点图
            fig_timeline = go.Figure()
            colors_map = {"LightGBM": "green", "LSTM": "orange"}
            for mt in df_timeline["模型类型"].unique():
                subset = df_timeline[df_timeline["模型类型"] == mt]
                fig_timeline.add_trace(go.Scatter(
                    x=subset["修改时间"],
                    y=subset["大小 (KB)"],
                    mode="markers",
                    marker=dict(size=12, color=colors_map.get(mt, "blue")),
                    name=mt,
                    text=subset["文件名"],
                    textposition="top center",
                ))
            fig_timeline.update_layout(
                title=t("模型文件创建时间线"),
                xaxis_title=t("时间"),
                yaxis_title=t("文件大小 (KB)"),
                height=350,
            )
            st.plotly_chart(fig_timeline, width='stretch')

    # 清理缓存
    st.divider()
    if st.button(t("🔄 刷新数据"), key="btn_refresh_history"):
        st.cache_data.clear()
        st.rerun()
