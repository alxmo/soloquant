# -*- coding: utf-8 -*-
"""
AI 量化交易系统 - 一键打包脚本
=====================================================
用法: python build_exe.py

功能:
  1. 自动检查环境
  2. 安装 PyInstaller
  3. 执行 onedir 打包
  4. 复制额外文件
  5. 生成启动脚本和说明
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent
DIST = ROOT / "dist" / "soloquant"
BUILD = ROOT / "build"


def run(cmd, **kwargs):
    """执行命令"""
    print(f"  > {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=ROOT, **kwargs)
    if result.returncode != 0:
        print(f"❌ 命令执行失败: {cmd}")
        sys.exit(1)
    return result


def step(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def main():
    os.chdir(ROOT)
    
    print("""
╔══════════════════════════════════════════════════════════╗
║   📦 AI 量化交易系统 - PyInstaller 打包工具 (onedir)     ║
╠══════════════════════════════════════════════════════════╣
║   模式: onedir (目录模式，启动快)                         ║
║   策略: 轻量依赖打包 + 大依赖首次运行自动安装              ║
║   排除: torch (~2GB), pyqlib (~500MB)                    ║
╚══════════════════════════════════════════════════════════╝
""")

    # 1. 检查 Python
    step("1/6 🔍 检查 Python 环境")
    print(f"  Python: {sys.version}")
    assert sys.version_info >= (3, 10), "需要 Python 3.10+"

    # 2. 安装 PyInstaller
    step("2/6 📦 安装/更新打包依赖")
    run(f"{sys.executable} -m pip install --upgrade pyinstaller packaging --quiet")

    # 3. 清理旧产物
    step("3/6 🧹 清理旧产物")
    for d in [DIST, BUILD]:
        if d.exists():
            shutil.rmtree(d)
            print(f"  已清理: {d}")

    # 4. 执行打包
    step("4/6 🚀 开始打包 (这可能需要几分钟)...")
    run(f"{sys.executable} -m PyInstaller soloquant.spec --clean --noconfirm")

    if not DIST.exists():
        print("❌ 打包失败：未找到输出目录")
        sys.exit(1)

    # 5. 复制额外文件
    step("5/6 📋 生成启动脚本和说明文件")

    # 主启动脚本
    launch_bat = DIST / "启动AI量化.bat"
    launch_bat.write_text(
        '@echo off\n'
        'chcp 65001 >nul\n'
        'cd /d "%~dp0"\n'
        'echo.\n'
        'echo  ═══════════════════════════════════════════════════════════\n'
        'echo     🤖 AI 全自动美股量化交易系统 v0.5.0\n'
        'echo  ═══════════════════════════════════════════════════════════\n'
        'echo.\n'
        'soloquant.exe %*\n'
        'echo.\n'
        'pause\n',
        encoding='utf-8'
    )

    # Dashboard 启动脚本
    dash_bat = DIST / "启动仪表盘.bat"
    dash_bat.write_text(
        '@echo off\n'
        'chcp 65001 >nul\n'
        'cd /d "%~dp0"\n'
        'echo 📈 启动 Streamlit 仪表盘...\n'
        'echo    浏览器会自动打开，如未打开请访问 http://localhost:8501\n'
        'echo.\n'
        'soloquant.exe dashboard\n'
        'pause\n',
        encoding='utf-8'
    )

    # 云端服务启动脚本
    cloud_bat = DIST / "启动云端服务.bat"
    cloud_bat.write_text(
        '@echo off\n'
        'chcp 65001 >nul\n'
        'cd /d "%~dp0"\n'
        'echo ☁️  启动云端 API 服务...\n'
        'echo    API 地址: http://localhost:8000\n'
        'echo.\n'
        'soloquant.exe cloud\n'
        'pause\n',
        encoding='utf-8'
    )

    # 使用说明
    readme = DIST / "使用说明.txt"
    readme.write_text(
        '═══════════════════════════════════════════════════════════\n'
        '   🤖 AI 全自动美股量化交易系统 - 便携版 v0.5.0\n'
        '═══════════════════════════════════════════════════════════\n\n'
        '📂 文件说明:\n'
        '   启动AI量化.bat      - 主程序入口（推荐）\n'
        '   启动仪表盘.bat      - 可视化仪表盘\n'
        '   启动云端服务.bat    - 云端API服务\n'
        '   soloquant.exe       - 核心程序\n'
        '   _internal/         - 运行依赖（请勿删除）\n\n'
        '🚀 快速开始:\n'
        '   1. 双击 "启动AI量化.bat"\n'
        '   2. 首次运行会自动检测并安装大依赖（torch/qlib）\n'
        '   3. 按提示配置 API Key (.env 文件)\n\n'
        '📝 命令行用法:\n'
        '   soloquant.exe chat       - 对话模式（默认）\n'
        '   soloquant.exe menu       - 菜单模式\n'
        '   soloquant.exe dashboard  - 启动仪表盘\n'
        '   soloquant.exe auto       - 全自动交易\n'
        '   soloquant.exe cloud      - 云端服务\n'
        '   soloquant.exe status     - 系统状态\n'
        '   soloquant.exe health     - 健康检查\n\n'
        '⚠️  注意:\n'
        '   - 首次启动需要联网\n'
        '   - 如需 GPU 加速，请手动安装 CUDA 版 torch:\n'
        '     pip install torch --index-url https://download.pytorch.org/whl/cu121\n',
        encoding='utf-8'
    )

    # 复制 .env 模板
    for f in ['.env.example', '.env.cloud.example', '.env.local.example']:
        src = ROOT / f
        if src.exists():
            shutil.copy2(src, DIST / f)

    # 复制 .env 文件（如果有）
    env_file = ROOT / '.env'
    if env_file.exists():
        shutil.copy2(env_file, DIST / '.env')
        print(f"  已复制 .env 配置文件")
    else:
        # 没有 .env，用 .env.example 作为模板
        env_example = ROOT / '.env.example'
        if env_example.exists():
            shutil.copy2(env_example, DIST / '.env')
            print(f"  已从 .env.example 生成 .env 模板")
        print(f"  ⚠️  请编辑 .env 文件配置 API Key")

    # 6. 统计
    step("6/6 ✅ 打包完成！")

    # 计算目录大小
    total_size = sum(f.stat().st_size for f in DIST.rglob('*') if f.is_file())
    size_mb = total_size / 1024 / 1024

    print(f"""
  📂 输出目录: {DIST}
  📦 打包大小: {size_mb:.1f} MB

  🚀 运行方式:
     cd {DIST}
     启动AI量化.bat

  💡 提示:
     - 打包已排除 torch/pyqlib（节省约 2.5GB）
     - 首次运行会自动 pip install 缺失依赖
     - 如需制作分发包，将 dist/soloquant 目录压缩为 zip 即可
""")


if __name__ == "__main__":
    main()
