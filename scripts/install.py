#!/usr/bin/env python3
"""
AI 全自动美股量化交易系统 - 一键依赖安装脚本 (跨平台 Python 版)
用法: python install.py

依赖将自动安装到项目目录下的 python_env 虚拟环境中。
"""

import subprocess
import sys
import os
import time
import platform

# ═══════════════════════════════════════════════════════════════
# 配置区
# ═══════════════════════════════════════════════════════════════

# 项目根目录（脚本所在目录）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# python_env 虚拟环境路径
PYTHON_ENV_DIR = os.path.join(PROJECT_ROOT, "python_env")

# 根据平台确定 python_env 中的 Python 可执行文件路径
if platform.system() == "Windows":
    ENV_PYTHON = os.path.join(PYTHON_ENV_DIR, "python.exe")
    ENV_PIP = os.path.join(PYTHON_ENV_DIR, "Scripts", "pip.exe")
else:
    ENV_PYTHON = os.path.join(PYTHON_ENV_DIR, "bin", "python")
    ENV_PIP = os.path.join(PYTHON_ENV_DIR, "bin", "pip")

# 分阶段安装列表 (顺序很重要)
INSTALL_PHASES = [
    {
        "name": "基础科学计算",
        "desc": "NumPy 数值计算 (pandas/lightgbm/qlib 的底层依赖)",
        "packages": [
            "numpy>=1.26,<2.0",
        ],
    },
    {
        "name": "数据层 (多源数据采集)",
        "desc": "AKShare + Finnhub + yfinance + FMP + FRED 金融数据采集与处理",
        "packages": [
            "pandas>=2.0,<3.0",
            "pyarrow>=15.0",
            "openpyxl>=3.1",
            "akshare>=1.14",
            "finnhub-python>=2.4.20",
            "yfinance>=0.2.37",
            "requests>=2.31",
            "fredapi>=0.5",
        ],
    },
    {
        "name": "量化引擎 (Qlib + ML)",
        "desc": "Microsoft Qlib 量化框架 + 机器学习模型",
        "packages": [
            "pyqlib>=0.9.6",
            "lightgbm>=4.3",
            "scikit-learn>=1.5",
            "matplotlib>=3.8",
            "plotly>=5.20",
        ],
    },
    {
        "name": "交易执行层 (Alpaca)",
        "desc": "美股交易 API + 实时行情 + 定时任务 + 日志",
        "packages": [
            "alpaca-py>=0.30",
            "websockets>=12.0",
            "python-dotenv>=1.0",
            "schedule>=1.2",
            "loguru>=0.7",
        ],
    },
    {
        "name": "API 服务层 (FastAPI)",
        "desc": "智能助手聊天服务 + 数据API + 模型同步API",
        "packages": [
            "fastapi>=0.110",
            "uvicorn>=0.29",
            "python-multipart>=0.0.9",
        ],
    },
    {
        "name": "UI 与工具",
        "desc": "Streamlit 仪表盘 + HTTP 客户端 + 数据校验 + 云端同步",
        "packages": [
            "streamlit>=1.35",
            "httpx>=0.27",
            "pydantic>=2.5",
            "paramiko>=3.0",
        ],
    },
]

# 验证导入映射
VERIFY_IMPORTS = {
    # 基础科学计算
    "numpy": "NumPy (数值计算)",
    # 数据层
    "pandas": "Pandas (数据处理)",
    "pyarrow": "PyArrow (Parquet缓存)",
    "openpyxl": "OpenPyXL (Excel读写)",
    "akshare": "AKShare (金融数据采集)",
    "finnhub": "Finnhub (新闻/情绪数据)",
    "yfinance": "yfinance (Yahoo Finance数据)",
    "requests": "Requests (HTTP客户端)",
    # 量化引擎
    "qlib": "Microsoft Qlib (量化引擎)",
    "lightgbm": "LightGBM (梯度提升模型)",
    "sklearn": "scikit-learn (机器学习)",
    "matplotlib": "Matplotlib (图表)",
    "plotly": "Plotly (交互图表)",
    # 交易执行
    "alpaca": "Alpaca-py (交易API)",
    "websockets": "WebSockets (实时行情)",
    "dotenv": "python-dotenv (环境变量)",
    "schedule": "Schedule (定时任务)",
    "loguru": "Loguru (日志系统)",
    # API服务
    "fastapi": "FastAPI (API框架)",
    "uvicorn": "Uvicorn (ASGI服务器)",
    "multipart": "python-multipart (文件上传)",
    # UI与工具
    "streamlit": "Streamlit (仪表盘)",
    "httpx": "httpx (异步HTTP)",
    "pydantic": "Pydantic (数据校验)",
    "paramiko": "Paramiko (SSH/SFTP)",
    # 可选
    "torch": "PyTorch (深度学习, 可选)",
}

# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def print_banner():
    print()
    print("  ╔═══════════════════════════════════════════════════════════════╗")
    print("  ║                                                               ║")
    print("  ║   AI 全自动美股量化交易系统 - 一键依赖安装脚本                  ║")
    print("  ║   Soloquant Trading System - Dependency Installer              ║")
    print("  ║                                                               ║")
    print("  ║   技术栈: Qlib + Alpaca + Finnhub + AKShare + SOLOOMO         ║")
    print("  ║   安装目标: python_env 虚拟环境                                ║")
    print("  ║                                                               ║")
    print("  ╚═══════════════════════════════════════════════════════════════╝")
    print()
    print(f"  当前 Python (启动器): {sys.version}")
    print(f"  安装目标 Python:     {ENV_PYTHON}")
    print(f"  Platform: {platform.platform()}")
    print()


def run_cmd(cmd: list[str], timeout: int = 600) -> tuple[bool, str]:
    """运行命令（静默模式，捕获输出），返回 (成功与否, 输出)"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def run_cmd_streaming(cmd: list[str], timeout: int = 600, prefix: str = "     ") -> tuple[bool, str]:
    """运行命令（实时流式输出模式），打印安装过程和进度，返回 (成功与否, 全部输出)"""
    import threading

    output_lines = []
    proc = None
    timed_out = False

    def target():
        nonlocal proc
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # 行缓冲
            )
            for line in proc.stdout:
                line = line.rstrip()
                output_lines.append(line)
                if line.strip():
                    # 过滤掉过多空行，高亮关键信息
                    print(f"{prefix}{line}", flush=True)
            proc.wait()
        except Exception as e:
            output_lines.append(str(e))

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        timed_out = True
        if proc:
            proc.kill()
        thread.join()
        return False, "\n".join(output_lines) + "\nTimeout"

    if proc and proc.returncode == 0:
        return True, "\n".join(output_lines)
    return False, "\n".join(output_lines)


def pip_install(packages: list[str], extra_args: list[str] = None) -> bool:
    """pip 安装一组包（安装到 python_env，实时显示进度）"""
    cmd = [ENV_PYTHON, "-m", "pip", "install"] + packages
    if extra_args:
        cmd += extra_args
    # 不加 --quiet，让 pip 输出下载进度条

    total = len(packages)
    print(f"     共 {total} 个包，开始安装...")
    print()

    ok, output = run_cmd_streaming(cmd, timeout=600, prefix="     ")
    if ok:
        print()
        return True

    # 批量安装失败，回退到逐个安装
    print()
    print(f"  ⚠️ 批量安装失败，尝试逐个安装...")
    print()
    all_ok = True
    for idx, pkg in enumerate(packages, 1):
        pkg_name = pkg.split(">=")[0].split("<")[0].split("==")[0].split("!=")[0].split("~=")[0]
        print(f"     [{idx}/{total}] 安装 {pkg}...", flush=True)
        ok2, out2 = run_cmd_streaming(
            [ENV_PYTHON, "-m", "pip", "install", pkg],
            timeout=300,
            prefix="       ",
        )
        if ok2:
            print(f"       ✅ {pkg_name} 安装成功")
        else:
            print(f"       ❌ {pkg_name} 安装失败")
            print(f"        可稍后手动安装: {ENV_PYTHON} -m pip install {pkg}")
            all_ok = False
        print()
    return all_ok


def verify_import(module_name: str, display_name: str) -> bool:
    """验证模块是否可以导入（在 python_env 中验证）"""
    try:
        # 使用 python_env 中的 Python 来验证导入
        result = subprocess.run(
            [ENV_PYTHON, "-c", f"import {module_name}; m=__import__('{module_name}'); print(getattr(m, '__version__', ''))"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            if version:
                print(f"  ✅ {display_name} {version}")
            else:
                print(f"  ✅ {display_name}")
            return True
        else:
            err = result.stderr.strip().split('\n')[-1] if result.stderr.strip() else "导入失败"
            print(f"  ❌ {display_name} - {err}")
            return False
    except Exception as e:
        print(f"  ⚠️  {display_name} - 异常: {e}")
        return False


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def fix_embedded_python():
    """修复嵌入式 Python 环境（Windows embeddable 版默认禁用 site，导致 pip 不可用）"""
    if platform.system() != "Windows":
        return True

    # 检查 _pth 文件 (如 python312._pth)
    pth_files = [f for f in os.listdir(PYTHON_ENV_DIR) if f.endswith("._pth")]
    if not pth_files:
        return True  # 非嵌入式 Python，无需修复

    pth_path = os.path.join(PYTHON_ENV_DIR, pth_files[0])
    with open(pth_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 检查是否需要修复
    needs_fix = False
    lines = content.splitlines()
    new_lines = []
    has_lib = False
    has_site_packages = False
    has_site_import = False

    for line in lines:
        stripped = line.strip()
        if stripped == "Lib":
            has_lib = True
        if stripped == "Lib\\site-packages" or stripped == "Lib/site-packages":
            has_site_packages = True
        if stripped == "import site" and not stripped.startswith("#"):
            has_site_import = True

    if not has_lib or not has_site_packages or not has_site_import:
        needs_fix = True

    if needs_fix:
        print("  🔧 检测到嵌入式 Python 环境，正在修复 _pth 配置...")
        # 重写 _pth 文件
        new_content = "python312.zip\n.\nLib\nLib\\site-packages\n\n# Uncomment to run site.main() automatically\nimport site\n"
        # 尝试匹配实际的 zip 文件名
        for f in os.listdir(PYTHON_ENV_DIR):
            if f.endswith(".zip") and f.startswith("python"):
                new_content = new_content.replace("python312.zip", f)
                break
        with open(pth_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        # 确保 Lib\site-packages 目录存在
        sp_dir = os.path.join(PYTHON_ENV_DIR, "Lib", "site-packages")
        os.makedirs(sp_dir, exist_ok=True)
        print("  ✅ _pth 配置已修复")

        # 如果 pip 不存在，安装 pip
        ok, _ = run_cmd([ENV_PYTHON, "-m", "pip", "--version"], timeout=10)
        if not ok:
            print("  📦 pip 未安装，正在安装 pip...")
            get_pip_path = os.path.join(PROJECT_ROOT, "get-pip.py")
            if os.path.isfile(get_pip_path):
                ok, out = run_cmd_streaming([ENV_PYTHON, get_pip_path], timeout=120, prefix="     ")
                if ok:
                    print("  ✅ pip 安装完成")
                else:
                    print(f"  ⚠️ pip 安装失败")
                    return False
            else:
                # 尝试用 ensurepip
                print("     尝试 ensurepip...")
                ok, out = run_cmd_streaming([ENV_PYTHON, "-m", "ensurepip", "--upgrade"], timeout=60, prefix="     ")
                if not ok:
                    print(f"  ❌ pip 安装失败，请手动运行: {ENV_PYTHON} get-pip.py")
                    return False
                print("  ✅ pip 安装完成 (ensurepip)")
    return True


def main():
    print_banner()

    # Step 0: 检查 python_env 是否存在
    print("  [Step 0/9] 检查 python_env 虚拟环境...")
    if not os.path.isfile(ENV_PYTHON):
        print(f"  ❌ 未找到 python_env 环境: {ENV_PYTHON}")
        print(f"     请确保 python_env 目录存在并包含 Python 解释器")
        sys.exit(1)

    # 获取 python_env 的版本信息
    ok, ver_output = run_cmd([ENV_PYTHON, "--version"], timeout=10)
    if ok:
        print(f"  ✅ 找到 python_env: {ver_output.strip()}")
    else:
        print(f"  ⚠️ 无法获取 python_env 版本信息，继续...")

    # 修复嵌入式 Python 环境（Windows）
    if not fix_embedded_python():
        print("  ❌ python_env 环境修复失败，无法继续安装")
        sys.exit(1)

    # 检查 python_env 的 Python 版本
    ok, ver_check = run_cmd(
        [ENV_PYTHON, "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"],
        timeout=10,
    )
    if ok:
        env_version = tuple(map(int, ver_check.strip().split(".")))
        if env_version < (3, 10):
            print(f"  ❌ python_env 中的 Python 版本需要 3.10+，当前: {ver_check.strip()}")
            sys.exit(1)
        print(f"  ✅ Python 版本检查通过 ({ver_check.strip()})")
    print()

    # Step 1: 升级 pip
    print("  [Step 1/9] 升级 python_env 中的 pip 和打包工具...")
    print("     正在升级 pip, setuptools, wheel...")
    print()
    ok, _ = run_cmd_streaming(
        [ENV_PYTHON, "-m", "pip", "install", "--upgrade",
         "pip", "setuptools", "wheel"],
        timeout=120,
        prefix="     ",
    )
    print()
    print("  ✅ pip 升级完成" if ok else "  ⚠️ pip 升级失败，继续安装...")
    print()

    # Step 2-7: 分阶段安装
    for i, phase in enumerate(INSTALL_PHASES):
        step_num = i + 2
        print(f"  [Step {step_num}/9] 安装{phase['name']}...")
        print(f"     {phase['desc']}")
        pip_install(phase["packages"])
        print(f"  ✅ {phase['name']}安装完成")
        print()

    # Step 8: PyTorch (可选)
    print("  [Step 8/9] 安装 PyTorch (CPU版, 可选)...")
    print("     用于 LSTM/Transformer 深度学习模型")
    print("     CPU版约 200MB，不含 CUDA")
    print()

    response = input("  是否安装 PyTorch CPU 版? [Y/n]: ").strip().lower()
    if response in ("", "y", "yes"):
        print("  正在安装 PyTorch CPU 版 (可能需要几分钟，请耐心等待)...")
        print()
        ok, output = run_cmd_streaming(
            [ENV_PYTHON, "-m", "pip", "install", "torch",
             "--index-url", "https://download.pytorch.org/whl/cpu"],
            timeout=600,
            prefix="     ",
        )
        print()
        if ok:
            print("  ✅ PyTorch CPU 版安装完成")
        else:
            print("  ⚠️ PyTorch 安装失败，可稍后手动安装:")
            print(f"     {ENV_PYTHON} -m pip install torch --index-url https://download.pytorch.org/whl/cpu")
    else:
        print("  ⏭️ 跳过 PyTorch 安装")
    print()

    # Step 9: 验证安装
    print("  [Step 9/9] 验证安装结果...")
    print("  ═══════════════════════════════════════════════════════")
    print("  🔍 正在验证 python_env 中的安装结果...")
    print("  ═══════════════════════════════════════════════════════")
    print()

    success_count = 0
    total_count = 0
    for module_name, display_name in VERIFY_IMPORTS.items():
        total_count += 1
        if verify_import(module_name, display_name):
            success_count += 1

    print()
    print("  ═══════════════════════════════════════════════════════")
    print()
    print(f"  📦 安装完成！({success_count}/{total_count} 个包验证成功)")
    print(f"  📍 所有依赖已安装到: {PYTHON_ENV_DIR}")
    print()
    print("  下一步:")
    print("    1. 复制 .env.example 为 .env，填入你的 API Key")
    print(f"    2. 运行验证脚本: {ENV_PYTHON} scripts/verify_integration.py")
    print(f"    3. 启动仪表盘: {ENV_PYTHON} -m streamlit run src/ui/dashboard.py")
    print()
    return 0 if success_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())
