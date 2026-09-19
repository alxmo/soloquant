# 贡献指南

感谢你对 Soloquant 的关注！欢迎任何形式的贡献 💚

## 🚀 如何贡献

### 报告问题（Issue）

- 提交前请先搜索 [已有 Issues](https://github.com/alxmo/soloquant/issues)，避免重复
- 提 Bug 时请包含：**环境信息**（Python 版本 / 操作系统）、**复现步骤**、**完整报错日志**
- 提功能建议时请描述：**使用场景**、**期望效果**

### 提交代码（Pull Request）

1. Fork 本仓库
2. 创建你的分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m "feat: 添加某个功能"`
4. 推送分支：`git push origin feature/amazing-feature`
5. 发起 Pull Request

### 提交信息规范

| 前缀 | 用途 |
|------|------|
| `feat:` | 新功能 |
| `fix:` | 修复 Bug |
| `docs:` | 文档更新 |
| `refactor:` | 代码重构 |
| `test:` | 测试相关 |
| `chore:` | 构建/工具变更 |

## 🛠️ 开发环境搭建

```bash
# 1. 克隆仓库
git clone https://github.com/alxmo/soloquant.git
cd soloquant

# 2. 安装依赖（本地完整版）
pip install -e ".[local,dev]"

# 3. 复制配置模板并填入你的 API Key
cp .env.example .env

# 4. 运行测试
pytest

# 5. 启动仪表盘
streamlit run src/ui/dashboard.py
```

## 📋 代码规范

- Python 3.10+，遵循 [PEP 8](https://peps.python.org/pep-0008/)（行宽上限 120）
- 使用 `ruff` 进行代码检查：`ruff check .`
- 新功能请附带测试用例（`tests/` 目录）
- 注释使用中文，关键模块需有 docstring

## ⚠️ 重要提醒

- **绝对不要提交 `.env` 文件或任何 API Key**
- 涉及交易逻辑的修改请格外谨慎，并说明测试情况
- 默认所有交易均在 **Paper Trading（模拟盘）** 模式下测试

## 📄 许可证

提交贡献即表示你同意将代码以 [MIT License](LICENSE) 开源。
