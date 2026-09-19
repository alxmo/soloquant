# 🚀 AI量化系统 - 云端+本地分布式部署指南

## 一、架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                    🌐 云端服务器 (2核2G)                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐    │
│  │ 实时行情  │ │ 交易执行  │ │ 风控引擎  │ │ 信号推理(轻量)│    │
│  │ WebSocket│ │ Alpaca   │ │ RiskGuard│ │ 加载训练好的  │    │
│  └──────────┘ └──────────┘ └──────────┘ │ 模型做预测    │    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ └──────────────┘    │
│  │ Dashboard│ │ 定时调度  │ │ 数据采集  │ ┌──────────────┐    │
│  │ Streamlit│ │ Scheduler│ │ Finnhub/ │ │ 模型同步API   │◄───┼── 手动上传
│  │ :8501    │ │          │ │ FRED/FMP │ │ (FastAPI)     │    │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘    │
│  内存占用: <1.5G                                            │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ HTTP API (模型上传/状态查询)
                              │
┌─────────────────────────────┼───────────────────────────────┐
│                💻 本地高配电脑 (Windows + GPU)                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐    │
│  │ 模型训练  │ │ 因子挖掘  │ │ 超参优化  │ │ 历史回测      │    │
│  │ LightGBM │ │ Factor   │ │ AutoOpt  │ │ Backtest     │    │
│  │ LSTM(GPU)│ │ Search   │ │          │ │              │    │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘    │
│  ┌──────────┐ ┌──────────┐                                    │
│  │ 模型打包  │ │ 上传脚本  │──── HTTP 上传到云端 ─────────────►│
│  └──────────┘ └──────────┘                                    │
└───────────────────────────────────────────────────────────────┘
```

---

## 二、云端部署步骤

### 2.1 服务器准备
1. 购买 2核2G 云服务器 (推荐 Ubuntu 22.04)
2. SSH 登录服务器，安装基础环境:
   ```bash
   sudo apt update && sudo apt install -y python3 python3-pip python3-venv git
   ```

### 2.2 项目部署
```bash
# 克隆项目 (或使用 SCP 上传)
git clone <your-repo> ~/soloquant
cd ~/soloquant

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装云端依赖 (轻量级, 不含 torch/qlib)
pip install -r documents/requirements-cloud.txt
# 或: pip install -e ".[cloud]"
```

### 2.3 配置环境变量
```bash
# 从模板创建配置
cp .env.cloud.example .env

# 编辑 .env, 填入实际 API Key
nano .env
```
关键配置项:
- `DEPLOY_MODE=cloud` (必须!)
- `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` (交易)
- `FINNHUB_API_KEY` (行情数据)
- `MODEL_API_KEY` (自己设定一个密钥, 本地上传时需要)
- `CLOUD_API_PORT=8000`

### 2.4 启动服务

#### 方式一: 一键启动 (推荐)
```bash
bash scripts/start_cloud.sh
```

#### 方式二: systemd 服务 (生产推荐)
```bash
# 复制服务配置
sudo cp scripts/cloud/ai-quant-api.service /etc/systemd/system/
sudo cp scripts/cloud/ai-quant-scheduler.service /etc/systemd/system/
sudo cp scripts/cloud/ai-quant-dashboard.service /etc/systemd/system/

# 修改 service 文件中的路径 (User, WorkingDirectory, ExecStart)
sudo nano /etc/systemd/system/ai-quant-api.service

# 启动并设为开机自启
sudo systemctl daemon-reload
sudo systemctl enable ai-quant-api ai-quant-scheduler ai-quant-dashboard
sudo systemctl start ai-quant-api ai-quant-scheduler ai-quant-dashboard

# 查看状态
sudo systemctl status ai-quant-api
```

#### 方式三: Python 命令启动
```bash
# 启动全部云端服务
python main.py cloud

# 或仅启动 API
python main.py api
```

### 2.5 验证云端
```bash
# 健康检查
curl http://localhost:8000/api/health

# 应返回类似:
# {"status":"ok","deploy_mode":"cloud","models_count":0,"memory_usage_mb":120.5}
```

### 2.6 防火墙放行
```bash
# 放行 API 和 Dashboard 端口
sudo ufw allow 8000/tcp
sudo ufw allow 8501/tcp
```

---

## 三、本地配置步骤

### 3.1 安装本地依赖
```bash
# 在项目根目录
cd C:\Users\me\Desktop\soloquant

# 安装完整依赖 (含训练/GPU)
pip install -r documents/requirements-local.txt
# 或: pip install -e ".[local]"

# 如果有 NVIDIA GPU, 安装 CUDA 版 PyTorch:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 3.2 配置环境变量
```bash
# 从模板创建配置 (本地)
copy .env.local.example .env

# 编辑 .env, 填入:
# - CLOUD_API_URL=http://你的云服务器IP:8000
# - MODEL_API_KEY=与云端相同的密钥
# - 各 API Key (Alpaca, Finnhub 等)
# - LLM 配置 (本地用)
```

### 3.3 验证本地连接云端
```bash
# 检查云端连接
python scripts/package_model.py --check

# 查看云端状态
python scripts/package_model.py --status

# 列出云端模型
python scripts/package_model.py --list
```

---

## 四、日常使用流程

### 4.1 训练 + 上传 (本地)
```bash
# 一键训练并上传
scripts\train_and_upload.bat

# 或分步执行:
# 步骤 1: 训练
python scripts/train_pipeline.py --quick

# 步骤 2: 上传
python scripts/package_model.py
```

### 4.2 云端自动运行
云端调度器会自动执行:
- 每天 06:00 (美东) 执行每日交易循环
- 每周日 20:00 执行策略迭代
- 每 5 分钟执行风控快照

### 4.3 查看云端状态
```bash
# 本地查看云端交易状态
python scripts/package_model.py --status

# 或浏览器访问
http://你的云服务器IP:8501    # Dashboard
http://你的云服务器IP:8000/docs  # API 文档
```

---

## 五、模型管理

### 5.1 查看模型
```bash
# 列出云端模型
python scripts/package_model.py --list

# 查看当前激活模型
curl http://你的云服务器IP:8000/api/models/active
```

### 5.2 切换模型
```bash
# 通过 API 切换
curl -X POST http://你的云服务器IP:8000/api/models/activate \
  -H "X-API-Key: your_key" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "lightgbm_20250115_120000.txt"}'
```

### 5.3 模型热加载
云端 API 接收新模型后:
1. 模型文件保存到 `data/models/`
2. 元数据写入 `best_model.json`
3. 推理加载器每 30 秒检查一次，自动加载新模型
4. **无需重启服务**

---

## 六、运维指南

### 6.1 日志查看
```bash
# 云端日志目录
ls ~/soloquant/logs/
# api.log        - API 服务日志
# scheduler.log  - 调度器日志
# dashboard.log  - Dashboard 日志

# 实时查看日志
tail -f ~/soloquant/logs/api.log
```

### 6.2 服务重启
```bash
# systemd 方式
sudo systemctl restart ai-quant-api
sudo systemctl restart ai-quant-scheduler

# 脚本方式
bash scripts/stop_cloud.sh
bash scripts/start_cloud.sh
```

### 6.3 内存监控
```bash
# 查看内存使用
free -h

# 查看进程内存
ps aux --sort=-%mem | head -10

# API 健康检查 (含内存)
curl http://localhost:8000/api/health
```

---

## 七、常见问题

### Q1: 云端内存不够?
- 确保没有安装 torch / pyqlib
- Dashboard 和 API 可以分时运行
- 考虑增加 swap: `sudo fallocate -l 2G /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile`

### Q2: 模型上传失败?
1. 检查 `CLOUD_API_URL` 是否正确
2. 检查 `MODEL_API_KEY` 是否匹配
3. 检查防火墙是否放行 8000 端口
4. 运行 `python scripts/package_model.py --check` 诊断

### Q3: 云端模型不生效?
- 推理加载器每 30 秒检查一次，请等待
- 检查 `data/models/best_model.json` 是否存在
- 查看 API 日志: `tail -f logs/api.log`

### Q4: 如何使用 Nginx 反向代理?
```bash
sudo cp scripts/cloud/nginx-ai-quant.conf /etc/nginx/sites-available/
sudo ln -s /etc/nginx/sites-available/nginx-ai-quant.conf /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

---

## 八、文件清单

### 新增文件
| 文件 | 说明 |
|------|------|
| `src/api/__init__.py` | API 模块初始化 |
| `src/api/model_sync.py` | 模型同步 FastAPI 服务 |
| `src/engine/model_inference.py` | 云端模型推理加载器 |
| `scripts/package_model.py` | 本地模型打包上传脚本 |
| `scripts/train_and_upload.bat` | 一键训练+上传批处理 |
| `scripts/start_cloud.sh` | 云端一键启动脚本 |
| `scripts/stop_cloud.sh` | 云端停止脚本 |
| `scripts/cloud/*.service` | systemd 服务配置 |
| `scripts/cloud/nginx-ai-quant.conf` | Nginx 反向代理配置 |
| `.env.cloud.example` | 云端配置模板 |
| `.env.local.example` | 本地配置模板 |
| `documents/requirements-cloud.txt` | 云端依赖清单 |
| `documents/requirements-local.txt` | 本地依赖清单 |

### 修改文件
| 文件 | 修改内容 |
|------|---------|
| `src/utils/config.py` | 添加 DEPLOY_MODE 等部署配置 |
| `src/engine/signal_generator.py` | 云端模式使用 model_inference 推理 |
| `src/ui/cli.py` | 新增 `cloud` 和 `api` 命令 |
| `main.py` | 更新用法文档 |
| `pyproject.toml` | 添加 cloud/local 可选依赖组 |
