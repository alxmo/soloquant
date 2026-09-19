#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔═══════════════════════════════════════════════════════════════╗
║           AI 量化系统 - 云端部署管理工具 v1.0                   ║
╠═══════════════════════════════════════════════════════════════╣
║  功能: 上传代码、安装Python、安装依赖、启动/停止/重启服务        ║
║  用法: python scripts/deploy_manager.py <命令> [选项]          ║
╚═══════════════════════════════════════════════════════════════╝

命令列表:
  upload          上传项目代码到服务器
  install-python  安装 Python 3.11（CentOS）
  install-deps    安装项目依赖（cloud 模式）
  start           启动所有服务（API + 调度器 + Dashboard）
  stop            停止所有服务
  restart         重启所有服务
  status          查看服务状态
  logs            查看实时日志
  health          健康检查
  deploy          一键完整部署（上传+安装+启动）
  sync-model      同步最优模型到云端

配置:
  在 .env 中设置以下变量:
    CLOUD_HOST=服务器IP
    CLOUD_PORT=22
    CLOUD_USER=root
    CLOUD_PASSWORD=密码 (或使用密钥)
    CLOUD_PATH=/opt/soloquant
    CLOUD_KEY_PATH=私钥路径 (可选, 优先使用密钥)
"""

import argparse
import os
import sys
import time
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from loguru import logger


# ─── 配置 ───────────────────────────────────────────────────────

CLOUD_HOST = os.getenv("CLOUD_HOST", "")
CLOUD_PORT = int(os.getenv("CLOUD_PORT", "22"))
CLOUD_USER = os.getenv("CLOUD_USER", "root")
CLOUD_PASSWORD = os.getenv("CLOUD_PASSWORD", "")
CLOUD_PATH = os.getenv("CLOUD_PATH", "/opt/soloquant")
CLOUD_KEY_PATH = os.getenv("CLOUD_KEY_PATH", "")

# 排除列表（上传时跳过）
EXCLUDE_PATTERNS = [
    '.bak', '.pytest_cache', '__pycache__', '.git', '.svn',
    'mlruns', 'data/qlib_data', '.coverage', '*.pyc',
    'node_modules', '.DS_Store', '*.log',
]

# 服务定义
SERVICES = {
    'api': {
        'name': '模型同步 API',
        'cmd': 'cd {path} && nohup python3.11 -m uvicorn src.api.model_sync:app --host 0.0.0.0 --port 8000 > logs/api.log 2>&1 &',
        'grep': 'uvicorn.*model_sync',
        'port': 8000,
        'log': 'logs/api.log',
    },
    'scheduler': {
        'name': '交易调度器',
        'cmd': 'cd {path} && nohup python3.11 main.py scheduler > logs/scheduler.log 2>&1 &',
        'grep': 'main.py scheduler',
        'port': None,
        'log': 'logs/scheduler.log',
    },
    'dashboard': {
        'name': 'Dashboard 仪表盘',
        'cmd': 'cd {path} && nohup python3.11 -m streamlit run src/ui/dashboard.py --server.port 8501 --server.headless true --server.address 0.0.0.0 > logs/dashboard.log 2>&1 &',
        'grep': 'streamlit.*dashboard',
        'port': 8501,
        'log': 'logs/dashboard.log',
    },
}


# ─── SSH 连接管理 ──────────────────────────────────────────────

class SSHManager:
    """SSH 连接管理器"""

    def __init__(self):
        self.client = None
        self.sftp = None
        self._connect()

    def _connect(self):
        """建立 SSH 连接"""
        try:
            import paramiko
        except ImportError:
            logger.error("paramiko 未安装，请运行: pip install paramiko")
            sys.exit(1)

        if not CLOUD_HOST:
            logger.error("请先在 .env 中配置 CLOUD_HOST")
            sys.exit(1)

        try:
            logger.info(f"🔌 连接服务器: {CLOUD_USER}@{CLOUD_HOST}:{CLOUD_PORT}")
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            connect_kwargs = {
                'hostname': CLOUD_HOST,
                'port': CLOUD_PORT,
                'username': CLOUD_USER,
                'timeout': 30,
            }

            # 优先使用密钥
            if CLOUD_KEY_PATH and Path(CLOUD_KEY_PATH).exists():
                logger.info(f"  使用密钥: {CLOUD_KEY_PATH}")
                connect_kwargs['key_filename'] = CLOUD_KEY_PATH
            elif CLOUD_PASSWORD:
                connect_kwargs['password'] = CLOUD_PASSWORD
            else:
                # 尝试默认密钥
                logger.info("  尝试使用默认 SSH 密钥...")

            self.client.connect(**connect_kwargs)
            self.sftp = self.client.open_sftp()
            logger.info("✅ 连接成功")
        except Exception as e:
            logger.error(f"❌ 连接失败: {e}")
            sys.exit(1)

    def exec(self, command: str, timeout: int = 60) -> tuple:
        """执行远程命令，返回 (stdout, stderr, exit_code)"""
        logger.debug(f"  $ {command}")
        stdin, stdout, stderr = self.client.exec_command(command, timeout=timeout)
        out = stdout.read().decode('utf-8', errors='replace')
        err = stderr.read().decode('utf-8', errors='replace')
        code = stdout.channel.recv_exit_status()
        return out, err, code

    def exec_print(self, command: str, timeout: int = 60, show_output: bool = False) -> int:
        """执行命令并打印输出"""
        out, err, code = self.exec(command, timeout)
        if show_output and out:
            print(out)
        if err and code != 0:
            logger.warning(f"  stderr: {err.strip()[:200]}")
        return code

    def upload_file(self, local_path: Path, remote_path: str):
        """上传单个文件"""
        remote_dir = str(Path(remote_path).parent)
        self.exec(f"mkdir -p {remote_dir}")
        self.sftp.put(str(local_path), remote_path)

    def close(self):
        """关闭连接"""
        if self.sftp:
            self.sftp.close()
        if self.client:
            self.client.close()
        logger.info("🔌 连接已关闭")


# ─── 部署功能 ──────────────────────────────────────────────────

def check_config():
    """检查配置是否完整"""
    required = ['CLOUD_HOST', 'CLOUD_USER', 'CLOUD_PATH']
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        logger.error(f"缺少必要配置: {', '.join(missing)}")
        logger.info("请在 .env 文件中添加以下配置:")
        logger.info("  CLOUD_HOST=你的服务器IP")
        logger.info("  CLOUD_PORT=22")
        logger.info("  CLOUD_USER=root")
        logger.info("  CLOUD_PASSWORD=密码 (或配置 CLOUD_KEY_PATH)")
        logger.info("  CLOUD_PATH=/opt/soloquant")
        return False
    return True


def cmd_upload(ssh: SSHManager):
    """上传项目代码到服务器"""
    logger.info("📤 上传项目代码...")

    # 先打包（排除不需要的文件）
    import tarfile
    import tempfile

    tar_path = Path(tempfile.gettempdir()) / f"Soloquant_deploy_{int(time.time())}.tar.gz"

    logger.info("  正在打包项目...")
    with tarfile.open(tar_path, 'w:gz') as tar:
        for item in ['src', 'scripts', 'main.py', 'pyproject.toml', '.env.cloud.example']:
            local_item = PROJECT_ROOT / item
            if not local_item.exists():
                continue

            if local_item.is_file():
                tar.add(local_item, arcname=item)
            else:
                for f in local_item.rglob('*'):
                    if f.is_file():
                        # 检查排除模式
                        rel_path = str(f.relative_to(PROJECT_ROOT))
                        skip = False
                        for pat in EXCLUDE_PATTERNS:
                            if pat in rel_path or f.name.endswith(pat):
                                skip = True
                                break
                        if not skip:
                            tar.add(f, arcname=str(f.relative_to(PROJECT_ROOT)))

    logger.info(f"  打包完成: {tar_path.stat().st_size / 1024:.1f} KB")

    # 上传 tar 包
    remote_tar = f"/tmp/Soloquant_deploy.tar.gz"
    logger.info("  上传中...")
    ssh.upload_file(tar_path, remote_tar)

    # 解压
    logger.info("  解压到服务器...")
    ssh.exec(f"mkdir -p {CLOUD_PATH}")
    code = ssh.exec_print(f"cd {CLOUD_PATH} && tar -xzf {remote_tar}")

    # 创建必要目录
    ssh.exec(f"cd {CLOUD_PATH} && mkdir -p data/models data/strategies logs")

    # 清理
    ssh.exec(f"rm -f {remote_tar}")
    tar_path.unlink(missing_ok=True)

    if code == 0:
        logger.info("✅ 代码上传完成")
    else:
        logger.error("❌ 上传失败")
    return code == 0


def cmd_install_python(ssh: SSHManager):
    """安装 Python 3.11"""
    logger.info("🐍 检查/安装 Python 3.11...")

    # 检查是否已安装
    out, _, code = ssh.exec("python3.11 --version 2>&1")
    if code == 0 and "Python 3.11" in out:
        logger.info(f"  ✅ 已安装: {out.strip()}")
        return True

    logger.info("  开始安装 Python 3.11...")

    # 检测系统类型
    out, _, _ = ssh.exec("cat /etc/os-release | head -3")
    os_info = out.lower()

    if 'centos' in os_info or 'rhel' in os_info or 'fedora' in os_info:
        # CentOS/RHEL
        code = ssh.exec_print("dnf install -y python3.11 python3.11-pip 2>&1 | tail -5", timeout=300)
    elif 'ubuntu' in os_info or 'debian' in os_info:
        # Ubuntu/Debian
        code = ssh.exec_print(
            "apt-get update && apt-get install -y python3.11 python3.11-pip python3.11-venv 2>&1 | tail -5",
            timeout=300
        )
    else:
        logger.error(f"  不支持的系统: {out.strip()}")
        return False

    if code == 0:
        # 升级 pip
        ssh.exec_print("python3.11 -m pip install --upgrade pip 2>&1 | tail -3", timeout=120)
        out, _, _ = ssh.exec("python3.11 --version")
        logger.info(f"  ✅ Python 安装完成: {out.strip()}")
        return True
    else:
        logger.error("  ❌ Python 安装失败")
        return False


def cmd_install_deps(ssh: SSHManager):
    """安装项目依赖"""
    logger.info("📦 安装项目依赖（cloud 模式）...")

    # 检查 Python
    _, _, code = ssh.exec("python3.11 --version")
    if code != 0:
        logger.error("  Python 3.11 未安装，请先运行 install-python")
        return False

    # 安装
    logger.info("  正在安装依赖（可能需要几分钟）...")
    code = ssh.exec_print(
        f"cd {CLOUD_PATH} && python3.11 -m pip install '.[cloud]' 2>&1 | tail -10",
        timeout=600
    )

    if code == 0:
        # 验证
        out, _, _ = ssh.exec(
            "python3.11 -c \"import streamlit, lightgbm, fastapi, pandas; print('OK')\" 2>&1"
        )
        if 'OK' in out:
            logger.info("  ✅ 依赖安装成功")
            return True

    logger.error("  ❌ 依赖安装失败")
    return False


def cmd_start(ssh: SSHManager):
    """启动所有服务"""
    logger.info("🚀 启动所有服务...")

    # 确保日志目录存在
    ssh.exec(f"mkdir -p {CLOUD_PATH}/logs")

    started = []
    for svc_id, svc in SERVICES.items():
        # 检查是否已在运行
        out, _, _ = ssh.exec(f"pgrep -f '{svc['grep']}' | wc -l")
        if int(out.strip()) > 0:
            logger.info(f"  ⏭️  {svc['name']} 已在运行")
            started.append(svc_id)
            continue

        # 启动服务
        cmd = svc['cmd'].format(path=CLOUD_PATH)
        ssh.exec(cmd)
        started.append(svc_id)
        logger.info(f"  ✅ {svc['name']} 已启动")

    # 等待几秒后检查状态
    logger.info("  等待服务就绪...")
    time.sleep(5)

    # 检查端口
    for svc_id, svc in SERVICES.items():
        if svc['port']:
            out, _, code = ssh.exec(f"curl -s -o /dev/null -w '%{{http_code}}' http://localhost:{svc['port']}/")
            if code == 0 and out.strip() in ['200', '301', '302', '404']:
                logger.info(f"  🌐 {svc['name']}: http://{CLOUD_HOST}:{svc['port']}")
            else:
                logger.warning(f"  ⚠️  {svc['name']} 端口 {svc['port']} 未响应")

    logger.info("✅ 所有服务启动完成")
    return True


def cmd_stop(ssh: SSHManager):
    """停止所有服务"""
    logger.info("⏹️  停止所有服务...")

    stopped = 0
    for svc_id, svc in SERVICES.items():
        out, _, _ = ssh.exec(f"pgrep -f '{svc['grep']}'")
        pids = out.strip().split('\n') if out.strip() else []

        if pids and pids[0]:
            ssh.exec(f"pkill -f '{svc['grep']}'")
            time.sleep(1)
            # 强制杀掉
            ssh.exec(f"pkill -9 -f '{svc['grep']}' 2>/dev/null")
            logger.info(f"  ✅ {svc['name']} 已停止 ({len(pids)} 个进程)")
            stopped += 1
        else:
            logger.info(f"  ⏭️  {svc['name']} 未运行")

    logger.info(f"✅ 已停止 {stopped} 个服务")
    return True


def cmd_restart(ssh: SSHManager):
    """重启所有服务"""
    logger.info("🔄 重启所有服务...")
    cmd_stop(ssh)
    time.sleep(2)
    cmd_start(ssh)
    return True


def cmd_status(ssh: SSHManager):
    """查看服务状态"""
    logger.info("📊 服务状态")
    print("─" * 60)

    all_ok = True
    for svc_id, svc in SERVICES.items():
        out, _, _ = ssh.exec(f"pgrep -f '{svc['grep']}' | wc -l")
        count = int(out.strip())

        if count > 0:
            status = "✅ 运行中"
            # 获取内存使用
            out_mem, _, _ = ssh.exec(
                f"ps aux | grep '{svc['grep']}' | grep -v grep | "
                "awk '{sum+=$6} END {printf \"%.1f\", sum/1024}'"
            )
            mem = f" ({out_mem.strip()} MB)" if out_mem.strip() else ""
        else:
            status = "❌ 已停止"
            mem = ""
            all_ok = False

        port_info = f":{svc['port']}" if svc['port'] else "   "
        print(f"  {svc['name']:<15} {port_info:<8} {status}{mem}")

    print("─" * 60)

    # 磁盘/内存
    out, _, _ = ssh.exec("df -h / | tail -1 | awk '{print $5}'")
    disk_usage = out.strip()
    out, _, _ = ssh.exec("free -m | grep Mem | awk '{printf \"%.1f%%\", $3/$2*100}'")
    mem_usage = out.strip()

    print(f"  磁盘使用: {disk_usage}  |  内存使用: {mem_usage}")
    print()

    return all_ok


def cmd_logs(ssh: SSHManager, service: str = None, lines: int = 50):
    """查看日志"""
    if service and service in SERVICES:
        log_file = SERVICES[service]['log']
        logger.info(f"📋 {SERVICES[service]['name']} 日志 (最近 {lines} 行)")
        print("─" * 60)
        out, _, _ = ssh.exec(f"tail -n {lines} {CLOUD_PATH}/{log_file}")
        print(out)
    else:
        logger.info("📋 所有服务日志摘要")
        for svc_id, svc in SERVICES.items():
            print(f"\n── {svc['name']} ({svc['log']}) ──")
            out, _, _ = ssh.exec(f"tail -n 5 {CLOUD_PATH}/{svc['log']} 2>/dev/null || echo '(无日志)'")
            print(out.strip())
    return True


def cmd_health(ssh: SSHManager):
    """健康检查"""
    logger.info("💊 健康检查")

    # API 健康检查
    out, _, code = ssh.exec("curl -s http://localhost:8000/api/health 2>&1")
    if code == 0 and 'status' in out:
        import json
        try:
            data = json.loads(out)
            print(f"  API 状态:   ✅ {data.get('status')}")
            print(f"  部署模式:   {data.get('deploy_mode')}")
            print(f"  模型数量:   {data.get('models_count')}")
            print(f"  活跃模型:   {data.get('active_model') or '无'}")
            print(f"  内存使用:   {data.get('memory_usage_mb', 0):.1f} MB")
        except:
            print(f"  API 响应: {out[:200]}")
    else:
        print("  API 状态:   ❌ 未响应")

    # Dashboard 检查
    out, _, code = ssh.exec("curl -s -o /dev/null -w '%{http_code}' http://localhost:8501/")
    if code == 0 and out.strip() == '200':
        print(f"  Dashboard:  ✅ 正常")
    else:
        print(f"  Dashboard:  ❌ 异常 (HTTP {out.strip()})")

    # 调度器检查
    out, _, _ = ssh.exec("pgrep -f 'main.py scheduler' | wc -l")
    if int(out.strip()) > 0:
        print(f"  调度器:     ✅ 运行中")
    else:
        print(f"  调度器:     ❌ 未运行")

    return True


def cmd_deploy(ssh: SSHManager):
    """一键完整部署"""
    logger.info("🚀 ===== 一键完整部署开始 =====")

    steps = [
        ("上传代码", cmd_upload),
        ("安装 Python", cmd_install_python),
        ("安装依赖", cmd_install_deps),
        ("启动服务", cmd_start),
    ]

    for name, func in steps:
        logger.info(f"\n📌 步骤: {name}")
        if not func(ssh):
            logger.error(f"❌ 部署失败: {name}")
            return False

    logger.info("\n🎉 ===== 部署完成 =====")
    logger.info(f"  Dashboard: http://{CLOUD_HOST}:8501")
    logger.info(f"  API:       http://{CLOUD_HOST}:8000/api/health")
    return True


def cmd_sync_model(ssh: SSHManager):
    """同步最优模型到云端"""
    logger.info("📦 同步最优模型到云端...")

    # 找到最优模型
    best_marker = PROJECT_ROOT / "data" / "models" / "best_model.json"
    if not best_marker.exists():
        logger.error("  未找到最优模型标记，请先在本地训练模型")
        return False

    import json
    with open(best_marker) as f:
        best = json.load(f)

    strategy_id = best.get("strategy_id", "")
    if not strategy_id:
        logger.error("  最优模型信息不完整")
        return False

    logger.info(f"  策略ID: {strategy_id}")
    logger.info(f"  模型类型: {best.get('model_type')}")
    logger.info(f"  夏普比率: {best.get('sharpe_ratio', 0):.2f}")
    logger.info(f"  年化收益: {best.get('annual_return', 0):.2%}")

    # 收集要上传的文件
    files_to_sync = []

    # 策略配置
    strategy_file = PROJECT_ROOT / "data" / "strategies" / f"{strategy_id}.json"
    if strategy_file.exists():
        files_to_sync.append((strategy_file, f"data/strategies/{strategy_id}.json"))

    # 回测结果
    bt_file = PROJECT_ROOT / "data" / "strategies" / f"{strategy_id}_backtest.json"
    if bt_file.exists():
        files_to_sync.append((bt_file, f"data/strategies/{strategy_id}_backtest.json"))

    # 模型文件
    model_dir = PROJECT_ROOT / "data" / "models"
    for f in model_dir.glob(f"*{strategy_id}*"):
        if f.is_file():
            files_to_sync.append((f, f"data/models/{f.name}"))

    if not files_to_sync:
        logger.error("  没有找到要同步的文件")
        return False

    # 上传
    logger.info(f"  上传 {len(files_to_sync)} 个文件...")
    for local_path, remote_rel in files_to_sync:
        remote_path = f"{CLOUD_PATH.rstrip('/')}/{remote_rel}"
        ssh.upload_file(local_path, remote_path)
        logger.info(f"    ↑ {local_path.name}")

    # 更新 best_model.json
    best_remote = f"{CLOUD_PATH.rstrip('/')}/data/models/best_model.json"
    ssh.upload_file(best_marker, best_remote)

    # 通知 API 重新加载
    logger.info("  通知 API 重新加载模型...")
    ssh.exec("curl -s -X POST http://localhost:8000/api/reload 2>/dev/null || true")

    logger.info("✅ 模型同步完成!")
    return True


# ─── 菜单模式 ──────────────────────────────────────────────────

def print_menu():
    """打印主菜单"""
    print()
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║           AI 量化系统 - 云端部署管理工具 v1.0                   ║")
    print("║           Cloud Deploy Manager - Menu Mode                    ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()
    print(f"  服务器: {CLOUD_USER}@{CLOUD_HOST}:{CLOUD_PORT}")
    print(f"  部署路径: {CLOUD_PATH}")
    print()
    print("  ── 部署类 ──────────────────────────────────────────────")
    print()
    print("    [1] 一键完整部署 (上传+安装+启动)")
    print("    [2] 上传项目代码")
    print("    [3] 安装 Python 3.11")
    print("    [4] 安装项目依赖")
    print()
    print("  ── 服务管理 ────────────────────────────────────────────")
    print()
    print("    [5] 启动所有服务")
    print("    [6] 停止所有服务")
    print("    [7] 重启所有服务")
    print("    [8] 查看服务状态")
    print()
    print("  ── 监控维护 ────────────────────────────────────────────")
    print()
    print("    [9] 查看实时日志")
    print("   [10] 健康检查")
    print("   [11] 同步最优模型到云端")
    print()
    print("  ── 其他 ────────────────────────────────────────────────")
    print()
    print("    [0] 退出")
    print()
    print("═" * 63)


def menu_logs(ssh: SSHManager):
    """日志查看子菜单"""
    while True:
        print()
        print("── 日志查看 ──")
        print("  [1] API 日志")
        print("  [2] 调度器日志")
        print("  [3] Dashboard 日志")
        print("  [4] 所有服务日志摘要")
        print("  [0] 返回主菜单")
        print()
        choice = input("请选择: ").strip()

        if choice == '1':
            lines = input("查看行数 (默认50): ").strip()
            lines = int(lines) if lines.isdigit() else 50
            cmd_logs(ssh, 'api', lines)
        elif choice == '2':
            lines = input("查看行数 (默认50): ").strip()
            lines = int(lines) if lines.isdigit() else 50
            cmd_logs(ssh, 'scheduler', lines)
        elif choice == '3':
            lines = input("查看行数 (默认50): ").strip()
            lines = int(lines) if lines.isdigit() else 50
            cmd_logs(ssh, 'dashboard', lines)
        elif choice == '4':
            cmd_logs(ssh)
        elif choice == '0':
            break
        else:
            print("  ❌ 无效选项")

        input("\n按回车键继续...")


def menu_mode():
    """交互式菜单模式"""
    # 检查配置
    if not check_config():
        sys.exit(1)

    # 连接
    ssh = SSHManager()

    try:
        while True:
            print_menu()
            choice = input("请输入选项 [0-11]: ").strip()

            if choice == '0':
                print()
                print("👋 再见!")
                break

            print()

            try:
                if choice == '1':
                    confirm = input("⚠️  将执行完整部署，确认继续? (y/N): ").strip().lower()
                    if confirm == 'y':
                        cmd_deploy(ssh)
                    else:
                        print("  已取消")
                elif choice == '2':
                    cmd_upload(ssh)
                elif choice == '3':
                    cmd_install_python(ssh)
                elif choice == '4':
                    cmd_install_deps(ssh)
                elif choice == '5':
                    cmd_start(ssh)
                elif choice == '6':
                    cmd_stop(ssh)
                elif choice == '7':
                    cmd_restart(ssh)
                elif choice == '8':
                    cmd_status(ssh)
                elif choice == '9':
                    menu_logs(ssh)
                    continue  # 子菜单自己处理暂停，这里跳过
                elif choice == '10':
                    cmd_health(ssh)
                elif choice == '11':
                    cmd_sync_model(ssh)
                else:
                    print("❌ 无效选项，请重新选择")
                    continue
            except KeyboardInterrupt:
                print("\n\n⏹️  操作已中断")
            except Exception as e:
                logger.error(f"操作失败: {e}")

            if choice != '9':  # 子菜单自己处理暂停
                input("\n按回车键返回菜单...")

    finally:
        ssh.close()


# ─── 主入口 ────────────────────────────────────────────────────

def main():
    # 如果没有参数或参数是 menu，进入菜单模式
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] == 'menu'):
        menu_mode()
        return

    # 命令行参数模式
    parser = argparse.ArgumentParser(
        description="AI 量化系统 - 云端部署管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/deploy_manager.py menu            # 交互式菜单模式
  python scripts/deploy_manager.py deploy          # 一键完整部署
  python scripts/deploy_manager.py upload          # 仅上传代码
  python scripts/deploy_manager.py restart         # 重启所有服务
  python scripts/deploy_manager.py status          # 查看服务状态
  python scripts/deploy_manager.py logs api        # 查看 API 日志
  python scripts/deploy_manager.py sync-model      # 同步最优模型
        """
    )
    parser.add_argument(
        'command',
        choices=['menu', 'upload', 'install-python', 'install-deps', 'start', 'stop',
                 'restart', 'status', 'logs', 'health', 'deploy', 'sync-model'],
        help='执行的命令 (menu 进入交互式菜单)'
    )
    parser.add_argument('--service', help='指定服务 (api/scheduler/dashboard)')
    parser.add_argument('--lines', type=int, default=50, help='日志行数 (默认50)')

    args = parser.parse_args()

    if args.command == 'menu':
        menu_mode()
        return

    # 检查配置
    if not check_config():
        sys.exit(1)

    # 连接
    ssh = SSHManager()

    try:
        # 执行命令
        cmd_map = {
            'upload': lambda: cmd_upload(ssh),
            'install-python': lambda: cmd_install_python(ssh),
            'install-deps': lambda: cmd_install_deps(ssh),
            'start': lambda: cmd_start(ssh),
            'stop': lambda: cmd_stop(ssh),
            'restart': lambda: cmd_restart(ssh),
            'status': lambda: cmd_status(ssh),
            'logs': lambda: cmd_logs(ssh, args.service, args.lines),
            'health': lambda: cmd_health(ssh),
            'deploy': lambda: cmd_deploy(ssh),
            'sync-model': lambda: cmd_sync_model(ssh),
        }

        func = cmd_map.get(args.command)
        if func:
            result = func()
            sys.exit(0 if result else 1)
        else:
            logger.error(f"未知命令: {args.command}")
            sys.exit(1)

    finally:
        ssh.close()


if __name__ == "__main__":
    main()
