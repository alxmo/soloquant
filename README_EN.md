<div align="center">

<img src="docs/images/logo.png" width="120" alt="Soloquant Logo"/>

# Soloquant

### 🤖 AI-Powered Fully Automated US Stock Quant Trading System

**Tell it what you want in plain language — let AI handle the rest**

From market research → smart screening → strategy generation → backtesting → paper trading → position monitoring, fully automated

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-v0.8.0-orange.svg)](#-roadmap)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Trading](https://img.shields.io/badge/Mode-Paper%20Trading-success.svg)](https://alpaca.markets/)

[简体中文](README.md) | **English**

**📖 Quick Links** · [Quick Start](#-quick-start) · [Features](#-key-features) · [Architecture](#-architecture) · [User Guide](docs/guides/user_guide.md) · [Deployment Guide](docs/guides/deployment_guide.md)

</div>

---

## 📸 Screenshots

<table>
<tr>
<td width="50%">

**📊 Dashboard Overview**

![Dashboard](docs/images/dashboard_overview.png)

*Account assets, position allocation, daily P&L, and risk alerts at a glance*

</td>
<td width="50%">

**💬 AI Assistant**

![AI Assistant](docs/images/ai_assistant.png)

*Analyze the market like chatting: AI summarizes news sentiment, spots opportunities, and suggests actions*

</td>
</tr>
<tr>
<td width="50%">

**📈 Market Analysis**

![Market Analysis](docs/images/market_analysis.png)

*Professional candlestick charts with MA/MACD/RSI/Bollinger indicators, one-click strategy generation*

</td>
<td width="50%">

**🎯 Smart Screening**

![Smart Screening](docs/images/smart_screening.png)

*Four-dimension scoring radar (Technical/Fundamental/Sentiment/Momentum) with side-by-side comparison*

</td>
</tr>
<tr>
<td width="50%">

**🔍 Market Research**

![Market Research](docs/images/market_research.png)

*Auto-fetches news coverage and market sentiment for your watchlist, generates research reports*

</td>
<td width="50%">

**₿ Crypto**

![Crypto](docs/images/crypto.png)

*Real-time price board for BTC/ETH/SOL and more, 24/7 non-stop*

</td>
</tr>
</table>

---

## 💡 What is Soloquant?

Soloquant is an **AI quant trading system built for everyone**. No coding skills or financial jargon required — just chat with it naturally:

> *"Pick 5 promising stocks around $10 for me"*
> *"How's the market lately? Any opportunities?"*
> *"Generate a trading strategy for NVIDIA"*

It handles everything else automatically:

```
🔍 Research → 📊 Screening → 📈 Strategy → 🧪 Backtest → 💰 Paper Trade → 🛡️ Risk Control
```

**Runs entirely on Alpaca Paper Trading — zero capital risk, experiment freely.**

---

## ✨ Why Soloquant?

### 🗣️ True Natural Language Interaction
Not simple keyword matching, but **LLM-driven full intent understanding**: multi-turn conversation memory, context awareness (it knows which page you're on), and access to all system functions. The AI assistant proactively analyzes positions, interprets news, and flags risks.

### 🧠 Complete Quant Research Pipeline
Built on **Microsoft Qlib**, an industry-grade quant framework: automated factor mining → LightGBM/LSTM model training → signal generation → historical backtesting. Every strategy must pass **A/B/C/D grading** before going live — failing grades are blocked automatically.

### 🛡️ Seven-Layer Risk Control
Per-position size caps, daily loss stops, max drawdown circuit breakers, Top-K signal filtering... 7 risk rules execute automatically, preventing impulsive decisions at the system level.

### 🏗️ Cloud + Local Distributed Architecture
Data collection and model training run on your cloud server, while the local dashboard stays lightweight and fast. Pure local mode is also supported — switch freely between deployment modes.

### 🌍 Bilingual UI (English / 中文)
Built-in i18n internationalization — switch the dashboard between English and Chinese with one click.

### 🖥️ Ready-to-Use Desktop App
Electron-packaged desktop installer — **double-click to run, no Python or dependencies required** (see [Releases](../../releases)).

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔍 **Smart Screening** | Four-dimension scoring (Technical/Fundamental/Sentiment/Momentum), screen candidates in one sentence, watchlist management |
| 🤖 **AI Assistant** | LLM-driven natural language interaction: intent understanding, full system access, multi-turn memory, context awareness |
| 📈 **Strategy Generation** | Microsoft Qlib-powered factor mining, model training (LightGBM/LSTM), trading signal generation |
| 🧪 **Backtesting** | Historical validation with A/B/C/D grading — failing strategies never go live |
| 💰 **Paper Trading** | Alpaca Paper Trading integration for realistic simulated execution |
| 🛡️ **Risk Control** | 7 automated rules: position caps, daily loss stops, max drawdown circuit breakers, etc. |
| 📊 **Visual Dashboard** | 10 pages: market analysis, research, macro indicators, options data, crypto, model training, and more |
| 📋 **Position Monitoring** | Real-time P&L tracking with automatic anomaly alerts |
| ⏰ **Scheduler** | Built-in scheduler: pre-market research, post-market review, fully automated |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────┐
│  Interaction Layer    Natural Language / Streamlit UI │
├──────────────────────────────────────────────────────┤
│  Agent Orchestration  Research → Strategy → Backtest  │
│                       → Execution → Monitor           │
├──────────────────────────────────────────────────────┤
│  Quant Engine         Factor Mining → Training →      │
│                       Signals → Backtesting           │
├──────────────────────────────────────────────────────┤
│  Data Layer           AKShare (quotes) + Finnhub      │
│                       (news/fundamentals) + Alpaca    │
│                       (account) + FRED (macro)        │
├──────────────────────────────────────────────────────┤
│  Execution Layer      Orders → Risk Control → Position│
│                       Management → Stop Loss/Take     │
└──────────────────────────────────────────────────────┘
```

**Tech Stack**: Python · Streamlit · Microsoft Qlib · LightGBM · PyTorch · Alpaca API · Finnhub · FastAPI · Electron · Multi-Agent Architecture

---

## 🚀 Quick Start

### Option 1: Desktop App (Recommended for Beginners)

Download the installer from [GitHub Releases](../../releases), double-click to run — **no Python or dependencies needed**.

> Windows version available. macOS / Linux users please use the source code option.

### Option 2: Run from Source (All Platforms)

#### Requirements

- Python 3.10+
- Windows / macOS / Linux

#### Three Steps to Launch

```bash
# 1️⃣ Clone the repository
git clone https://github.com/alxmo/soloquant.git
cd soloquant

# 2️⃣ Install dependencies
pip install -e ".[local]"     # Full version (training/GPU)
# or pip install -e ".[cloud]"  # Lightweight version (cloud deployment only)

# 3️⃣ One-click launch (auto-checks environment, generates config, opens browser)
python run.py
```

> 💡 **macOS / Linux**: run `./launch.sh` — it automatically creates a virtual environment, installs dependencies, and starts the app.

### Get Free API Keys

| Service | Purpose | Sign Up |
|---------|---------|---------|
| [Alpaca](https://app.alpaca.markets/) | Paper trading | Free — Paper Key on registration |
| [Finnhub](https://finnhub.io/register) | News/fundamentals data | Free tier is enough for personal use |
| LLM (optional) | AI Assistant | Any OpenAI-compatible API |

> 💡 **No LLM key needed for core features**: screening, backtesting, and paper trading work without LLM. The AI assistant is an optional enhancement.

---

## 📁 Project Structure

```
soloquant/
├── src/
│   ├── agents/        # Multi-agent architecture (Research/Strategy/Execution/Monitor)
│   ├── api/           # FastAPI cloud API
│   ├── data/          # Data collection & processing
│   ├── engine/        # Quant engine (factors/models/backtest)
│   ├── execution/     # Trade execution & risk control
│   ├── llm/           # LLM integration layer
│   ├── scheduler/     # Scheduled tasks
│   ├── ui/            # Streamlit dashboard
│   └── workflow/      # Workflow orchestration
├── app/               # Electron desktop app
├── tests/             # Test cases
├── documents/         # User guides & deployment docs
├── docs/images/       # Screenshots
├── run.py             # Cross-platform launcher (Windows/macOS/Linux)
├── launch.sh          # macOS/Linux one-click launcher
└── launch.bat         # Windows one-click launcher
```

---

## 📖 Documentation

- 📕 [User Guide](docs/guides/user_guide.md) — Features, configuration, FAQ
- 📗 [Beginner's Guide (Chinese)](docs/guides/beginner_guide.md) — Zero-to-hero onboarding (Chinese)
- 📘 [Deployment Guide](documents/部署指南-云端本地分布式.md) — Cloud + local distributed deployment (Chinese)
- 📙 [Local Research Environment (Chinese)](docs/guides/local_research_guide.md) — Local training setup (Chinese)

---

## ⚠️ Disclaimer

> This project is for **learning and research purposes only** and does **not constitute investment advice**.
>
> - Stock trading involves risk; past backtest performance does not guarantee future returns
> - All trades default to Paper Trading mode — do not switch to live trading lightly
> - Any profits or losses from using this project are the user's sole responsibility

---

## 🗺️ Roadmap

- [x] v0.3.0 — Dashboard + scheduler
- [x] v0.4.0 — LLM intelligent conversation (intent recognition + multi-turn dialogue)
- [x] v0.5.0 — Cloud + local distributed architecture
- [x] v0.8.0 — Macro indicators, options data, crypto section
- [x] English README
- [ ] More strategy templates
- [ ] Live trading guide (cautiously)

---

<div align="center">

### If this project helps you, please give it a ⭐ Star!

**Stars keep the development going** 🚀

[![Star History Chart](https://api.star-history.com/svg?repos=alxmo/soloquant&type=Date)](https://star-history.com/#alxmo/soloquant&Date)

</div>
