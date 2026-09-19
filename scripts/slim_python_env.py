# -*- coding: utf-8 -*-
"""
python_env 瘦身脚本 - 为 Electron 打包 Release 准备
=====================================================
用途: 清理 python_env 中运行时不需要的包, 将安装包从 ~2GB 压缩到 ~900MB

安全策略:
- 只删除"确认不被 src/ 引用"的包 (torch/mlflow/jupyter/debugpy 等)
- torch 已在 lstm_model.py 中做优雅降级 (TORCH_AVAILABLE=False 时禁用 LSTM)
- 默认预览模式, --apply 才实际删除

用法:
    python slim_python_env.py            # 预览模式 (只打印, 不删除)
    python slim_python_env.py --apply    # 实际执行删除
"""

import os
import shutil
import sys
import argparse

# ─── 待清理清单 (已验证 src/ 中无引用或已优雅降级) ───
REMOVE_PACKAGES = [
    # 深度学习 (490MB) - 仅 lstm_model.py 使用, 已优雅降级
    "torch",
    # ML 实验追踪 (68MB) - 源码零引用
    "mlflow",
    "databricks",
    # Jupyter 全家桶 (~100MB) - 开发工具
    "jupyterlab",
    "notebook",
    "jupyter_server",
    "jupyter_client",
    "jupyter_core",
    "jupyter_events",
    "jupyter_lsp",
    "ipykernel",
    "ipywidgets",
    "nbclient",
    "nbconvert",
    "nbformat",
    "nbclassic",
    "qtconsole",
    # 调试器 (31MB) - 开发工具
    "debugpy",
    # 符号计算 (66MB) - 仅 torch 依赖
    "sympy",
    "mpmath",
    # 其他开发/可选依赖
    "py_mini_racer",   # 38MB - qlib 可选依赖
    "networkx",         # 15MB - 仅 torch 依赖
    "winpty",           # 6.6MB - 仅 jupyter 依赖
    "babel",            # 30MB - 仅 jupyter 依赖
    "pydeck",           # 23MB - streamlit 可选地图组件
    "fontTools",        # 15MB - matplotlib 可选
    "jedi",             # 14MB - IDE 补全
    "parso",
    "win32",            # 5MB - Windows COM, 运行时不需要
    "win32com",
    "win32exts",
    "pythonwin",
    "pywin32_system32",
    "adodbapi",
    "isort",
    # 打包/测试工具
    "pip",
    "setuptools",
    "wheel",
    "pytest",
    "_pytest",
    "pytest_asyncio",
    "iniconfig",
    "pluggy",
]

# dist-info 元数据目录匹配前缀
REMOVE_DISTINFO_PREFIXES = [
    "torch-", "mlflow-", "databricks-", "jupyter", "notebook", "ipykernel",
    "ipywidgets", "nbclient", "nbconvert", "nbformat", "nbclassic",
    "qtconsole", "debugpy-", "sympy-", "mpmath-", "py_mini_racer-",
    "networkx-", "winpty-", "pip-", "setuptools-", "wheel-", "pytest-",
    "_pytest", "pytest_asyncio", "iniconfig-", "pluggy-", "babel-",
    "pydeck-", "fonttools-", "jedi-", "parso-", "pywin32-", "adodbapi-",
    "isort-",
]


def get_size(path: str) -> int:
    """递归计算路径大小"""
    total = 0
    if os.path.isfile(path):
        return os.path.getsize(path)
    for dp, dn, fn in os.walk(path):
        for f in fn:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="实际执行删除 (默认只预览)")
    args = parser.parse_args()

    sp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "python_env", "Lib", "site-packages")
    sp = os.path.abspath(sp)
    if not os.path.exists(sp):
        print(f"❌ 未找到 site-packages: {sp}")
        sys.exit(1)

    total_before = get_size(sp)
    print(f"清理前大小: {total_before/1024/1024:.0f} MB\n")

    removed_bytes = 0
    found = []

    # 1. 统计待删除的包目录
    for pkg in REMOVE_PACKAGES:
        p = os.path.join(sp, pkg)
        if os.path.exists(p):
            sz = get_size(p)
            removed_bytes += sz
            found.append((pkg, sz))

    # 2. 统计 dist-info 元数据
    distinfo_bytes = 0
    for name in os.listdir(sp):
        if any(name.startswith(prefix) for prefix in REMOVE_DISTINFO_PREFIXES) \
                and (name.endswith(".dist-info") or name.endswith(".egg-info")):
            distinfo_bytes += get_size(os.path.join(sp, name))

    # 3. 统计 __pycache__
    pycache_bytes = 0
    for dp, dn, fn in os.walk(sp):
        for d in dn:
            if d == "__pycache__":
                pycache_bytes += get_size(os.path.join(dp, d))

    # ─── 输出预览 ───
    print("待删除的包:")
    for pkg, sz in sorted(found, key=lambda x: -x[1]):
        print(f"  🗑  {pkg:25s} {sz/1024/1024:7.1f} MB")
    print(f"  🗑  {'(dist-info 元数据)':25s} {distinfo_bytes/1024/1024:7.1f} MB")
    print(f"  🗑  {'(所有 __pycache__)':25s} {pycache_bytes/1024/1024:7.1f} MB")

    total_removed = removed_bytes + distinfo_bytes + pycache_bytes
    print(f"\n{'═'*50}")
    print(f"可释放空间: {total_removed/1024/1024:.0f} MB")
    print(f"清理后预计: {(total_before-total_removed)/1024/1024:.0f} MB")

    # ─── 实际执行 ───
    if args.apply:
        print("\n开始删除...")
        for pkg, sz in found:
            p = os.path.join(sp, pkg)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                try:
                    os.remove(p)
                except OSError:
                    pass
            print(f"  ✅ 已删除 {pkg}")

        for name in os.listdir(sp):
            if any(name.startswith(prefix) for prefix in REMOVE_DISTINFO_PREFIXES) \
                    and (name.endswith(".dist-info") or name.endswith(".egg-info")):
                shutil.rmtree(os.path.join(sp, name), ignore_errors=True)

        for dp, dn, fn in os.walk(sp):
            for d in list(dn):
                if d == "__pycache__":
                    shutil.rmtree(os.path.join(dp, d), ignore_errors=True)

        total_after = get_size(sp)
        print(f"\n🎉 完成! 清理后实际大小: {total_after/1024/1024:.0f} MB")
    else:
        print("\n⚠️  当前为预览模式, 确认无误后执行:")
        print("    python scripts/slim_python_env.py --apply")


if __name__ == "__main__":
    main()
