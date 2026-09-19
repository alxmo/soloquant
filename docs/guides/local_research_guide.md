# 本地量化研究环境使用指南

## 📁 目录结构

```
soloquant/
├── scripts/
│   ├── local_setup.py       # 环境配置脚本 (数据初始化)
│   ├── train_pipeline.py    # 训练工作流 (数据→训练→回测→报告)
│   ├── sync_to_cloud.py     # 云端同步工具
│   ├── init_local_data.bat  # 一键初始化数据 (双击运行)
│   ├── train_local.bat      # 一键训练 (双击运行)
│   └── sync_to_cloud.bat    # 一键同步 (双击运行)
├── data/
│   ├── qlib_data/           # Qlib 原始数据 (50只美股)
│   ├── models/              # 训练好的模型文件
│   │   └── best_model.json  # 当前最优模型标记
│   ├── strategies/          # 策略配置 + 回测结果
│   ├── cache/               # 因子缓存
│   └── reports/             # 训练/环境报告
```

## 🚀 快速开始

### 第一步：初始化数据（只需做一次）

双击运行 `scripts/init_local_data.bat`

或命令行执行：
```bash
python scripts/local_setup.py --full
```

这会下载 50 只美股 3 年历史数据并预计算 Alpha158 因子。
预计耗时：5-15 分钟（取决于网络）。

### 第二步：训练模型

双击运行 `scripts/train_local.bat`

或命令行执行：
```bash
# 快速训练 (8只股票，约5-10分钟)
python scripts/train_pipeline.py --quick

# 标准训练 (20只股票，约20-40分钟)
python scripts/train_pipeline.py

# 完整训练 (50只股票，约1-2小时)
python scripts/train_pipeline.py --years 3

# 包含 LSTM 深度学习
python scripts/train_pipeline.py --lstm
```

训练完成后会自动：
- ✅ 因子搜索与筛选
- ✅ 模型训练
- ✅ 回测验证
- ✅ 生成对比报告
- ✅ 标记最优模型 (best_model.json)

### 第三步：同步到云端

**前置条件**：在 `.env` 中配置云端服务器信息：
```
CLOUD_HOST=你的服务器IP
CLOUD_PORT=22
CLOUD_USER=用户名
CLOUD_PASSWORD=密码
CLOUD_PATH=/home/user/soloquant
```

然后双击运行 `scripts/sync_to_cloud.bat`

或命令行执行：
```bash
# 同步最优模型 (推荐)
python scripts/sync_to_cloud.py

# 同步所有模型和策略
python scripts/sync_to_cloud.py --all

# 检查云端连接
python scripts/sync_to_cloud.py --check

# 回滚到上一版本
python scripts/sync_to_cloud.py --rollback
```

## 📊 股票池说明

默认股票池（50只主流美股）：

| 分类 | 股票代码 |
|------|---------|
| 科技巨头 | AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, AMD, NFLX, INTC, ORCL, ADBE, CRM, PYPL, UBER, COIN |
| 金融 | JPM, BAC, GS, MS, V, MA, AXP, BLK |
| 消费 | WMT, COST, HD, MCD, SBUX, NKE, TGT, LOW |
| 医疗 | JNJ, UNH, PFE, ABBV, MRK, TMO, ABT, LLY |
| 能源/工业 | XOM, CVX, BA, CAT, GE, DIS, VZ, T |
| 中概 | BABA, BIDU, JD |

可以在 `scripts/local_setup.py` 中修改 `LOCAL_STOCK_POOL` 自定义股票池。

## 🔄 日常工作流

### 每周策略迭代（推荐流程）

```
周一凌晨/周末:
  1. 更新数据: python scripts/local_setup.py --update
  2. 训练模型: python scripts/train_pipeline.py
  3. 查看报告: data/reports/train_report_xxx.json
  4. 效果满意 → 同步到云端: python scripts/sync_to_cloud.py
  5. 效果不满意 → 调参后重新训练

工作日:
  云端自动运行交易，本地随时可以登录查看
```

### 环境检查

```bash
# 查看当前环境状态
python scripts/local_setup.py --check
```

## ⚙️ 配置优化

### 调整训练线程数

本地高配电脑可以开满线程，默认会自动使用 `CPU核心数 - 1`。

如需手动调整，编辑 `scripts/train_pipeline.py` 中的 `num_threads`。

### 自定义股票池

编辑 `scripts/local_setup.py` 中的 `LOCAL_STOCK_POOL` 列表。

### 调整训练参数

编辑 `src/engine/model_manager.py` 中的默认参数：
- `num_leaves`: 叶子节点数（越大越复杂）
- `max_depth`: 最大深度
- `learning_rate`: 学习率
- `num_threads`: 线程数

## 📈 结果解读

### 策略评级标准

| 评级 | 年化收益 | 夏普比率 | 最大回撤 |
|------|---------|---------|---------|
| A级 | >20% | >1.5 | <10% |
| B级 | >15% | >1.0 | <15% |
| C级 | >10% | >0.5 | <20% |
| D级 | 其他 | 其他 | 其他 |

### 回测指标说明

- **年化收益**: 策略平均每年的收益率
- **夏普比率**: 风险调整后收益，越高越好（>1不错，>1.5优秀）
- **最大回撤**: 历史最大亏损幅度，越小越稳

## 🐛 常见问题

### Q: 数据下载失败怎么办？
A: 检查网络连接，AKShare 数据源来自 Yahoo Finance，国内可能需要代理。可以多试几次，或减少股票数量。

### Q: 训练报错内存不足？
A: 减少股票池数量，或减少训练年限。可以先用 `--quick` 模式测试。

### Q: 云端连接失败？
A: 检查 `.env` 中的 CLOUD_* 配置是否正确，确认服务器 SSH 端口是否开放。

### Q: 如何添加新股票？
A: 在 `LOCAL_STOCK_POOL` 中添加股票代码，然后运行 `python scripts/local_setup.py --update` 更新数据。

## 📝 版本历史

- v1.0 (2026-08-07): 初始版本，支持数据初始化、模型训练、云端同步
