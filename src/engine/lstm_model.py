"""
LSTM 深度学习模型 - 时序预测
=====================================================
Phase 6 T6.3: LSTM 模型支持

功能:
- 基于 PyTorch 的 LSTM 网络定义
- 时序序列数据构建 (滑动窗口) — build_sequences()
- 训练流程 (含 early stopping)
- 模型保存/加载 (.pth 格式)
- 验证集 IC 计算

依赖: torch (可选, 缺失时优雅降级)

API 对齐说明:
- build_sequences(features, labels, lookback) -> (X, y)
- LSTMTrainer(lookback=, hidden_size=, ..., params=) 独立参数构造
- LSTMTrainer.train(features, labels, stock_pool) -> ModelResult
- LSTMTrainer.predict(model_path, features) -> np.ndarray  (多序列预测)
- LSTMNetwork.forward(x) -> (batch,) 一维输出
"""

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.models import ModelResult
from src.utils.config import config

# 安全导入 PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch 未安装, LSTM 模型不可用。请安装: pip install torch")


# ═══════════════════════════════════════════════════════
# 序列构建 (不依赖 PyTorch)
# ═══════════════════════════════════════════════════════

def build_sequences(
    features: pd.DataFrame,
    labels: pd.Series,
    lookback: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """将面板数据按滑动窗口切分为序列样本

    对于 T 行数据, 生成 T - lookback 个序列样本:
      - 序列 i 的特征: features.iloc[i : i+lookback]  -> (lookback, F)
      - 序列 i 的标签: labels.iloc[i + lookback]

    Args:
        features: 特征 DataFrame, shape (T, F)
        labels:   标签 Series,   shape (T,)
        lookback: 回看窗口长度

    Returns:
        X: shape (N, lookback, F) 的 numpy 数组
        y: shape (N,) 的 numpy 数组
        若数据不足则返回空数组 (shape (0, lookback, F) 和 (0,))
    """
    T = len(features)
    n_cols = features.shape[1]
    if T <= lookback:
        empty_X = np.empty((0, lookback, n_cols), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.float32)
        return empty_X, empty_y

    feat_arr = features.values.astype(np.float32)
    label_arr = labels.values.astype(np.float32)

    # 向量化滑窗 (等价于逐行循环, 性能提升数十倍)
    # sliding_window_view 沿 axis=0 得到 shape (T-lookback+1, F, lookback)
    windows = np.lib.stride_tricks.sliding_window_view(feat_arr, lookback, axis=0)
    # 转置为 (N, lookback, F), 并去掉最后一个窗口 (其标签越界)
    X = np.ascontiguousarray(windows[:-1].transpose(0, 2, 1))
    y = label_arr[lookback:]

    return X, y


# ═══════════════════════════════════════════════════════
# 网络定义 (需要 PyTorch)
# ═══════════════════════════════════════════════════════

if TORCH_AVAILABLE:

    class LSTMNetwork(nn.Module):
        """LSTM 网络结构

        结构: LSTM(input) -> Dropout -> Linear(hidden, 1) -> squeeze

        forward 输入:  (batch, seq_len, input_size)
        forward 输出:  (batch,)  — 一维预测值
        """

        def __init__(
            self,
            input_size: int,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            """前向传播

            Args:
                x: (batch, seq_len, input_size)
            Returns:
                (batch,) 预测值
            """
            lstm_out, _ = self.lstm(x)            # (batch, seq_len, hidden)
            last_hidden = lstm_out[:, -1, :]       # (batch, hidden)
            out = self.dropout(last_hidden)
            pred = self.fc(out)                    # (batch, 1)
            return pred.squeeze(-1)                # (batch,)


# ═══════════════════════════════════════════════════════
# LSTM 训练器
# ═══════════════════════════════════════════════════════

class LSTMTrainer:
    """LSTM 训练流程封装

    构造参数以独立关键字传入 (与 test/model_manager 调用对齐):
        lookback, hidden_size, num_layers, dropout,
        n_epochs, lr, patience, batch_size, weight_decay
    """

    DEFAULT_PARAMS = {
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "n_epochs": 50,
        "lr": 1e-3,
        "batch_size": 32,
        "lookback": 20,
        "patience": 10,
        "weight_decay": 1e-5,
    }

    def __init__(
        self,
        lookback: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        n_epochs: int = 50,
        lr: float = 1e-3,
        patience: int = 10,
        batch_size: int = 32,
        weight_decay: float = 1e-5,
        params: Optional[dict] = None,
    ):
        """初始化 LSTM 训练器

        Args:
            lookback:      滑动窗口长度
            hidden_size:   LSTM 隐藏层维度
            num_layers:    LSTM 层数
            dropout:       Dropout 比率
            n_epochs:      最大训练轮数
            lr:            学习率
            patience:      Early Stopping 耐心值
            batch_size:    批次大小
            weight_decay:  L2 正则化系数
            params:        可选, 以字典覆盖以上参数 (兼容旧调用)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch 未安装, 无法使用 LSTMTrainer")

        # 优先使用独立参数, params 字典可覆盖
        self.params = {
            "lookback": lookback,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "n_epochs": n_epochs,
            "lr": lr,
            "patience": patience,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
        }
        if params:
            self.params.update(params)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional["LSTMNetwork"] = None
        self.input_size: int = 0
        self.best_val_loss: float = float("inf")

    # ─── 训练 ───

    def train_from_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        stock_pool: Optional[list[str]] = None,
    ) -> ModelResult:
        """从已构建好的序列样本训练 (供多股票拼接数据使用)

        Args:
            X: shape (N, lookback, F) 的序列特征数组
            y: shape (N,) 的标签数组
            stock_pool: 股票池 (仅用于日志)

        Returns:
            ModelResult — 成功时 model_path 非空; 数据不足时 model_path 为空
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch 未安装, 无法使用 LSTMTrainer")

        if len(X) < 20:
            logger.error(
                f"LSTM 训练数据不足: {len(X)} 序列样本 (需 ≥20)"
            )
            return ModelResult(model_type="lstm")

        logger.info(
            f"LSTM 训练 | 序列样本: {len(X)} | lookback={X.shape[1]} | "
            f"特征数: {X.shape[2]} | 股票池: {stock_pool or 'N/A'}"
        )

        # ─── 按时间顺序分割 (前 80% 训练, 后 20% 验证) ───
        split = int(len(X) * 0.8)
        if split < 2 or len(X) - split < 2:
            logger.error(f"LSTM 训练/验证分割后样本不足: train={split}, val={len(X)-split}")
            return ModelResult(model_type="lstm")

        X_train_raw, X_val_raw = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # ─── 标准化 (Z-Score): 仅用训练集统计量, 消除数据泄漏 ───
        feat_mean = X_train_raw.mean(axis=(0, 1))
        feat_std = X_train_raw.std(axis=(0, 1)) + 1e-8
        X_train_norm = (X_train_raw - feat_mean) / feat_std
        X_val_norm = (X_val_raw - feat_mean) / feat_std

        return self._fit_and_save(
            X_train_norm, X_val_norm, y_train, y_val,
            input_size=X.shape[2], lookback=X.shape[1],
            feat_mean=feat_mean, feat_std=feat_std,
            feature_names=None, stock_pool=stock_pool,
        )

    def train(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        stock_pool: Optional[list[str]] = None,
    ) -> ModelResult:
        """训练 LSTM 模型

        Args:
            features:   特征 DataFrame (T, F)
            labels:     标签 Series (T,) — 前向收益
            stock_pool: 股票池 (仅用于日志, 可为 None)

        Returns:
            ModelResult — 成功时 model_path 非空; 数据不足时 model_path 为空
        """
        lookback = self.params["lookback"]

        # 构建序列
        X, y = build_sequences(features, labels, lookback=lookback)

        if len(X) < 20:
            logger.error(
                f"LSTM 训练数据不足: {len(X)} 序列样本 (需 ≥20), "
                f"lookback={lookback}, 原始行数={len(features)}"
            )
            return ModelResult(model_type="lstm")

        logger.info(
            f"LSTM 训练 | 序列样本: {len(X)} | lookback={lookback} | "
            f"特征数: {features.shape[1]} | 股票池: {stock_pool or 'N/A'}"
        )

        # ─── 按时间顺序分割 (前 80% 训练, 后 20% 验证) ───
        # 时序数据禁止随机分割, 否则验证集包含未来信息导致泄漏
        split = int(len(X) * 0.8)
        if split < 2 or len(X) - split < 2:
            logger.error(f"LSTM 训练/验证分割后样本不足: train={split}, val={len(X)-split}")
            return ModelResult(model_type="lstm")

        X_train_raw, X_val_raw = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # ─── 标准化 (Z-Score): 仅用训练集统计量, 消除数据泄漏 ───
        feat_mean = X_train_raw.mean(axis=(0, 1))
        feat_std = X_train_raw.std(axis=(0, 1)) + 1e-8
        X_train_norm = (X_train_raw - feat_mean) / feat_std
        X_val_norm = (X_val_raw - feat_mean) / feat_std

        return self._fit_and_save(
            X_train_norm, X_val_norm, y_train, y_val,
            input_size=features.shape[1], lookback=lookback,
            feat_mean=feat_mean, feat_std=feat_std,
            feature_names=list(features.columns), stock_pool=stock_pool,
        )

    def _fit_and_save(
        self,
        X_train_norm: np.ndarray,
        X_val_norm: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        input_size: int,
        lookback: int,
        feat_mean: np.ndarray,
        feat_std: np.ndarray,
        feature_names: Optional[list],
        stock_pool: Optional[list[str]] = None,
    ) -> ModelResult:
        """核心训练循环 + 模型保存 (train / train_from_sequences 共用)"""
        # 转为 PyTorch Tensor
        X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val_norm, dtype=torch.float32)
        y_tensor_train = torch.tensor(y_train, dtype=torch.float32)
        y_tensor_val = torch.tensor(y_val, dtype=torch.float32)

        train_set = torch.utils.data.TensorDataset(X_train_tensor, y_tensor_train)
        val_set = torch.utils.data.TensorDataset(X_val_tensor, y_tensor_val)

        split = len(X_train_norm)

        train_loader = DataLoader(
            train_set, batch_size=self.params["batch_size"], shuffle=True,
        )
        val_loader = DataLoader(
            val_set, batch_size=self.params["batch_size"], shuffle=False,
        )

        # 构建模型
        self.input_size = input_size
        self.model = LSTMNetwork(
            input_size=self.input_size,
            hidden_size=self.params["hidden_size"],
            num_layers=self.params["num_layers"],
            dropout=self.params["dropout"],
        ).to(self.device)

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.params["lr"],
            weight_decay=self.params["weight_decay"],
        )
        criterion = nn.MSELoss()

        # 训练循环
        patience_counter = 0
        best_state = None

        logger.info(
            f"LSTM 训练开始 | 样本: {split} 训练 + {len(X_val_norm)} 验证 | "
            f"输入维度: {self.input_size} | 设备: {self.device} | 分割: 时序顺序 (无泄漏)"
        )

        for epoch in range(self.params["n_epochs"]):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch_x)

            train_loss /= max(split, 1)

            # 验证阶段
            val_loss, val_pred, val_actual = self._evaluate(
                self.model, val_loader, criterion,
            )

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"  Epoch {epoch+1}/{self.params['n_epochs']} | "
                    f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                    f"Best: {self.best_val_loss:.6f}"
                )

            if patience_counter >= self.params["patience"]:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

        # 恢复最佳模型
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # 计算 IC
        train_pred, train_actual = self._predict_dataset(self.model, train_loader)
        train_ic = self._calc_ic(train_pred, train_actual)
        train_rank_ic = self._calc_ic(train_pred, train_actual, method="spearman")
        val_ic = self._calc_ic(val_pred, val_actual)
        val_rank_ic = self._calc_ic(val_pred, val_actual, method="spearman")

        # 保存模型
        model_dir = config.DATA_DIR / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"

        torch.save({
            "model_state": self.model.state_dict(),
            "input_size": self.input_size,
            "hidden_size": self.params["hidden_size"],
            "num_layers": self.params["num_layers"],
            "dropout": self.params["dropout"],
            "lookback": lookback,
            "feat_mean": feat_mean,
            "feat_std": feat_std,
            "feature_names": feature_names,
            "train_ic": float(train_ic),
            "valid_ic": float(val_ic),
        }, str(model_path))

        logger.info(
            f"LSTM 训练完成 | IC={val_ic:.4f}, RankIC={val_rank_ic:.4f} | "
            f"模型已保存: {model_path.name}"
        )

        return ModelResult(
            model_type="lstm",
            train_ic=float(train_ic),
            valid_ic=float(val_ic),
            train_rank_ic=float(train_rank_ic),
            valid_rank_ic=float(val_rank_ic),
            model_path=str(model_path),
            params=self.params,
        )

    # ─── 预测 ───

    def predict(self, model_path: str, features: pd.DataFrame) -> Optional[np.ndarray]:
        """加载模型并预测

        对输入特征按 lookback 窗口生成所有序列, 返回每个序列的预测值。

        Args:
            model_path: .pth 模型文件路径
            features:   特征 DataFrame, 至少 lookback+1 行

        Returns:
            预测值 numpy 数组, shape (N,) — N = len(features) - lookback
            若数据不足则返回 None
        """
        if not TORCH_AVAILABLE:
            return None

        try:
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False,
            )

            # 重建模型
            model = LSTMNetwork(
                input_size=checkpoint["input_size"],
                hidden_size=checkpoint["hidden_size"],
                num_layers=checkpoint["num_layers"],
                dropout=checkpoint["dropout"],
            ).to(self.device)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()

            lookback = checkpoint["lookback"]
            feat_mean = checkpoint["feat_mean"]
            feat_std = checkpoint["feat_std"]

            # 构建序列
            feat_arr = features.values.astype(np.float32)
            if len(feat_arr) <= lookback:
                logger.warning(
                    f"特征数据不足: {len(feat_arr)} <= lookback {lookback}"
                )
                return None

            N = len(feat_arr) - lookback
            # 向量化滑窗 (等价于逐行循环, 性能提升数十倍)
            windows = np.lib.stride_tricks.sliding_window_view(feat_arr, lookback, axis=0)
            sequences = np.ascontiguousarray(windows[:-1].transpose(0, 2, 1))

            # 标准化
            sequences = (sequences - feat_mean) / feat_std

            # 批量预测
            preds = []
            batch_size = 128
            with torch.no_grad():
                for start in range(0, N, batch_size):
                    batch = torch.tensor(
                        sequences[start : start + batch_size],
                        dtype=torch.float32,
                    ).to(self.device)
                    pred = model(batch)
                    preds.extend(pred.cpu().numpy())

            return np.array(preds)

        except Exception as e:
            logger.error(f"LSTM 预测失败: {e}")
            return None

    # ─── 内部方法 ───

    def _evaluate(self, model, loader, criterion):
        """评估模型"""
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_actuals = []

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                total_loss += loss.item() * len(batch_x)

                all_preds.extend(pred.cpu().numpy())
                all_actuals.extend(batch_y.cpu().numpy())

        return (
            total_loss / max(len(loader.dataset), 1),
            np.array(all_preds),
            np.array(all_actuals),
        )

    def _predict_dataset(self, model, loader):
        """对整个数据集预测"""
        model.eval()
        all_preds = []
        all_actuals = []

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                pred = model(batch_x)
                all_preds.extend(pred.cpu().numpy())
                all_actuals.extend(batch_y.cpu().numpy())

        return np.array(all_preds), np.array(all_actuals)

    @staticmethod
    def _calc_ic(pred, actual, method="pearson") -> float:
        """计算 IC"""
        if len(pred) != len(actual) or len(pred) == 0:
            return 0.0
        pred_s = pd.Series(pred)
        actual_s = pd.Series(actual)
        if method == "spearman":
            return float(pred_s.corr(actual_s, method="spearman"))
        return float(pred_s.corr(actual_s))


# ═══════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════

def is_lstm_available() -> bool:
    """检查 LSTM 是否可用 (PyTorch 是否安装)"""
    return TORCH_AVAILABLE


def build_lstm_features(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """从 K 线数据构建 LSTM 特征矩阵

    使用与 model_manager._calc_features 相同的因子集,
    确保 LSTM 和 LightGBM 使用一致的特征。

    Args:
        df: 原始 K 线数据 (date, open, high, low, close, volume)
        lookback: 用于计算滑动窗口特征 (不影响输出行数)

    Returns:
        特征 DataFrame
    """
    close = df["close"]
    volume = df["volume"]
    returns = close.pct_change()

    feat = pd.DataFrame(index=df.index)
    feat["rsi_14"] = _calc_rsi(close, 14)
    feat["ma_ratio"] = close.rolling(5).mean() / close.rolling(20).mean()
    feat["vol_ratio"] = volume / volume.rolling(20).mean()
    feat["momentum_5"] = close.pct_change(5)
    feat["momentum_20"] = close.pct_change(20)
    feat["volatility_20"] = returns.rolling(20).std()
    feat["price_ma60"] = close / close.rolling(60).mean()
    return feat


def _calc_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """RSI 计算"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    return 100 - (100 / (1 + rs))
