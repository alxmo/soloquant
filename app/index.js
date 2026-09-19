const { app, BrowserWindow, Menu, dialog, ipcMain, Tray, nativeImage } = require('electron');
const url = require('url');
const path = require('path');
const { spawn, exec } = require('child_process');
const http = require('http');

// ========== 单例锁 ==========
const gotTheLock = app.requestSingleInstanceLock();

if (!gotTheLock) {
  // 新实例：静默退出，不弹窗
  app.quit();
  process.exit(0);
}

// ========== 全局变量 ==========
let mainWindow = null;
let tray = null;
let isQuitting = false;
let backendProc = null;      // Python 后端进程
let backendStarted = false;  // 后端是否已启动
const threads = {};

// ========== 常量 ==========
const DASHBOARD_PORT = 8501;   // Streamlit 仪表盘端口
const CHAT_PORT = 8502;        // 智能助手聊天服务端口
const BACKEND_TIMEOUT = 120000; // 后端启动超时 (120秒)
const POLL_INTERVAL = 1000;     // 端口轮询间隔 (1秒)

// ========== 路径 ==========
const PROJECT_ROOT = path.resolve(__dirname, '..');           // 项目根目录 (resources/)
const PYTHON_EXE = path.join(PROJECT_ROOT, 'python_env', 'python.exe');
const MAIN_PY = path.join(PROJECT_ROOT, 'main.py');
const LOG_DIR = path.join(PROJECT_ROOT, 'logs');

// ========== 激活已有窗口 ==========
const focusMainWindow = () => {
  if (!mainWindow) return;
  if (mainWindow.isMinimized()) mainWindow.restore();
  mainWindow.show();
  mainWindow.focus();
};

// ========== 端口检查 ==========
function isPortInUse(port) {
  return new Promise((resolve) => {
    const req = http.get({ host: 'localhost', port, path: '/', timeout: 1500 }, (res) => {
      res.destroy();
      resolve(true);
    });
    req.on('error', () => resolve(false));
    req.on('timeout', () => { req.destroy(); resolve(false); });
  });
}

// ========== 等待端口就绪 ==========
async function waitForPort(port, timeout = BACKEND_TIMEOUT) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (await isPortInUse(port)) return true;
    await new Promise(r => setTimeout(r, POLL_INTERVAL));
  }
  return false;
}

// ========== 启动 Python 后端 ==========
function startBackend() {
  if (backendStarted && backendProc && backendProc.pid) {
    console.log('[backend] 后端已在运行, 跳过启动');
    return;
  }

  // 启动前先清理上次异常退出可能残留的端口进程
  killPortProcesses(DASHBOARD_PORT);
  killPortProcesses(CHAT_PORT);

  // 确保日志目录存在
  try {
    const fs = require('fs');
    if (!fs.existsSync(LOG_DIR)) fs.mkdirSync(LOG_DIR, { recursive: true });
  } catch (e) { /* 忽略 */ }

  const logFile = path.join(LOG_DIR, 'electron_backend.log');
  const fs = require('fs');
  const outFd = fs.openSync(logFile, 'a');  // 打开日志文件获取 fd

  console.log('[backend] 启动 Python 后端: ' + PYTHON_EXE + ' main.py dashboard');
  console.log('[backend] 工作目录: ' + PROJECT_ROOT);
  console.log('[backend] 日志: ' + logFile);

  try {
    backendProc = spawn(PYTHON_EXE, ['main.py', 'dashboard'], {
      cwd: PROJECT_ROOT,
      env: { ...process.env, PYTHONPATH: PROJECT_ROOT },
      stdio: ['ignore', outFd, outFd],
      windowsHide: true,
    });
    backendStarted = true;

    backendProc.on('error', (err) => {
      console.error('[backend] 启动失败:', err.message);
      backendStarted = false;
    });

    backendProc.on('exit', (code, signal) => {
      console.log(`[backend] 后端进程退出 (code=${code}, signal=${signal})`);
      backendStarted = false;
      backendProc = null;
    });
  } catch (err) {
    console.error('[backend] 启动异常:', err.message);
    backendStarted = false;
  }
}

// ========== 同步清理占用端口的进程 (供退出时兜底) ==========
function killPortProcesses(port) {
  try {
    const { execSync } = require('child_process');
    const stdout = execSync(`netstat -ano | findstr :${port} | findstr LISTENING`, {
      windowsHide: true,
      encoding: 'utf8',
      timeout: 5000,
    });
    if (!stdout) return;
    const pids = new Set();
    stdout.split('\n').forEach((line) => {
      const m = line.trim().match(/(\d+)\s*$/);
      if (m) pids.add(m[1]);
    });
    pids.forEach((pid) => {
      try {
        execSync(`taskkill /pid ${pid} /T /F`, { windowsHide: true, timeout: 5000 });
        console.log(`[backend] 已清理端口 ${port} 的进程 PID ${pid}`);
      } catch (e) { /* 进程可能已退出 */ }
    });
  } catch (e) { /* 无占用或命令失败 */ }
}

// ========== 停止 Python 后端 (释放 8501/8502 端口) ==========
function stopBackend() {
  console.log('[backend] 正在停止后端服务...');

  // 1. 终止后端进程树 (taskkill /T 覆盖子进程: streamlit + uvicorn)
  if (backendProc && backendProc.pid) {
    const pid = backendProc.pid;
    try {
      exec(`taskkill /pid ${pid} /T /F`, { windowsHide: true }, (err) => {
        if (err) {
          console.log(`[backend] taskkill 返回: ${err.message}`);
        } else {
          console.log(`[backend] 已终止进程树 (PID ${pid})`);
        }
      });
    } catch (e) {
      console.error('[backend] taskkill 异常:', e.message);
    }
    backendProc = null;
    backendStarted = false;
  }

  // 2. 兜底: 同步清理占用 8501/8502 端口的残留进程
  killPortProcesses(DASHBOARD_PORT);
  killPortProcesses(CHAT_PORT);
}

// ========== 创建主窗口 ==========
const createWindow = () => {
  // ----- macOS 菜单设置 -----
  if (process.platform === 'darwin') {
    const template = [{
      label: "程序",
      submenu: [
        {
          label: '重载',
          accelerator: 'CmdOrCtrl+R',
          click: function (item, focusedWindow) {
            if (focusedWindow) {
              if (focusedWindow.id === 1) {
                BrowserWindow.getAllWindows().forEach(function (win) {
                  if (win.id > 1) win.close();
                });
              }
              focusedWindow.reload();
            }
          }
        }, {
          label: "退出",
          accelerator: "Command+Q",
          click: function () {
            isQuitting = true;
            app.quit();
          }
        }
      ]
    },
    {
      label: "编辑",
      submenu: [
        { label: '撤销', accelerator: 'CmdOrCtrl+Z', selector: 'undo' },
        { label: '重做', accelerator: 'Shift+CmdOrCtrl+Z', selector: 'redo' },
        { label: "复制", accelerator: "CmdOrCtrl+C", selector: "copy:" },
        { label: "粘贴", accelerator: "CmdOrCtrl+V", selector: "paste:" },
        { label: "剪切", accelerator: "CmdOrCtrl+X", selector: "cut:" },
        { label: '全选', accelerator: 'CmdOrCtrl+A', selector: 'selectall' }
      ]
    }, {
      label: '查看',
      submenu: [
        {
          label: '全屏',
          accelerator: process.platform === 'darwin' ? 'Ctrl+Command+F' : 'F11',
          click: function (item, focusedWindow) {
            if (focusedWindow) {
              focusedWindow.setFullScreen(!focusedWindow.isFullScreen());
            }
          }
        }, {
          label: '开发者工具',
          accelerator: process.platform === 'darwin' ? 'Alt+Command+I' : 'Ctrl+Shift+I',
          click: function (item, focusedWindow) {
            if (focusedWindow) focusedWindow.toggleDevTools();
          }
        }
      ]
    }];
    Menu.setApplicationMenu(Menu.buildFromTemplate(template));
  } else {
    Menu.setApplicationMenu(null);
  }

  // ----- 创建主窗口 -----
  mainWindow = new BrowserWindow({
    width: 1600,
    height: 900,
    frame: true,
    webPreferences: {
      nodeIntegration: true,
      webSecurity: false,
      contextIsolation: false,
      enableRemoteModule: true
    }
  });

  // ----- 先加载本地加载页，等待后端就绪后跳转 -----
  mainWindow.loadFile(path.join(__dirname, 'loading.html'));

  // 启动后端并等待就绪
  startBackend();
  waitForPort(DASHBOARD_PORT).then((ready) => {
    if (!mainWindow) return;
    if (ready) {
      console.log('[backend] 仪表盘就绪, 加载 http://localhost:8501');
      mainWindow.loadURL(`http://localhost:${DASHBOARD_PORT}`);
    } else {
      console.error('[backend] 等待仪表盘超时');
      mainWindow.loadFile(path.join(__dirname, 'loading.html'));
      // 超时后仍尝试加载，若后端最终可用则自动恢复
      setTimeout(() => {
        if (mainWindow) mainWindow.loadURL(`http://localhost:${DASHBOARD_PORT}`);
      }, 5000);
    }
  });

  // ----- 拦截外部链接在新窗口打开 -----
  mainWindow.webContents.on('will-navigate', (event, url) => {
    if (url.startsWith('http://') || url.startsWith('https://')) {
      event.preventDefault();
      const childWindow = new BrowserWindow({
        width: 800,
        height: 600,
        webPreferences: {
          nodeIntegration: true,
          webSecurity: false,
          contextIsolation: false,
          enableRemoteModule: true
        }
      });
      childWindow.loadURL(url);
    }
  });

  // ----- 窗口关闭：直接隐藏到托盘，不弹窗 -----
  mainWindow.on('close', (event) => {
    if (isQuitting) return;           // 如果是退出，允许关闭
    event.preventDefault();           // 阻止关闭
    mainWindow.hide();               // 直接隐藏到托盘
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  createTray();
};

// ========== 创建系统托盘 ==========
function createTray() {
  const iconPath = path.join(__dirname, 'icon.png');
  let icon = nativeImage.createFromPath(iconPath);
  if (icon.isEmpty()) {
    icon = nativeImage.createEmpty();
  }
  tray = new Tray(icon);
  tray.setToolTip('SOLOQUANT');

  const contextMenu = Menu.buildFromTemplate([
    {
      label: '显示SOLOQUANT窗口',
      click: focusMainWindow
    },
    {
      label: '退出',
      click: () => {
        isQuitting = true;
        app.quit();
      }
    }
  ]);
  tray.setContextMenu(contextMenu);

  tray.on('click', () => {
    if (mainWindow) {
      if (mainWindow.isVisible()) {
        mainWindow.hide();
      } else {
        focusMainWindow();
      }
    }
  });

  tray.on('double-click', focusMainWindow);
}

// ========== app 生命周期 ==========
app.on('ready', createWindow);

// ========== 监听第二个实例：直接激活已有窗口，无提示 ==========
app.on('second-instance', (event, commandLine, workingDirectory) => {
  focusMainWindow();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  } else {
    focusMainWindow();
  }
});

// ========== 退出前清理后端 ==========
app.on('before-quit', () => {
  isQuitting = true;
  stopBackend();
});

// ========== 进程退出兜底: 无论何种方式退出都清理后端 ==========
// 注意: 'exit' 事件中只能执行同步操作
process.on('exit', () => {
  try {
    killPortProcesses(DASHBOARD_PORT);
    killPortProcesses(CHAT_PORT);
  } catch (e) { /* 忽略 */ }
});

app.allowRendererProcessReuse = true;

// ========== IPC 事件 ==========
ipcMain.on('window-min', () => {
  if (mainWindow) mainWindow.minimize();
});

ipcMain.on('window-max', () => {
  if (!mainWindow) return;
  if (!mainWindow.isMaximized()) mainWindow.maximize();
  else mainWindow.unmaximize();
});

// 自定义关闭按钮 → 直接隐藏到托盘（与系统关闭行为一致）
ipcMain.on('window-close', () => {
  if (mainWindow) {
    mainWindow.hide();
  }
});

ipcMain.on('toggledevtools', () => {
  if (mainWindow) mainWindow.toggleDevTools();
});

ipcMain.on('opendir', (e, data) => {
  if (!data) data = {};
  const d = dialog.showOpenDialogSync(mainWindow, {
    properties: [data.title || 'openFile', 'openDirectory']
  });
  if (d && d[0]) {
    mainWindow.send('opendir', d[0]);
  }
});

ipcMain.on('openfile', (e, data) => {
  if (!data) data = {};
  const d = dialog.showOpenDialogSync(mainWindow, {
    properties: [data.title || 'openFile'],
    filters: [{ name: 'All Files', extensions: ['*'] }]
  });
  if (d && d[0]) {
    mainWindow.send('openfile', d[0]);
  }
});

// ----- 线程管理 -----
ipcMain.on('create-thread', (e, data) => {
  const win = new BrowserWindow({
    width: ~~data.width,
    height: ~~data.height,
    show: data.visible,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      enableRemoteModule: true,
    }
  });
  if (data.openDevTools) win.webContents.openDevTools();
  win.loadURL(url.format({
    pathname: path.join(__dirname, data.url),
    protocol: 'file:',
    slashes: true
  }));
  threads[data.id] = win;
});

ipcMain.on('thread-message', (e, data) => {
  const win = threads[data.id];
  if (!win) return;
  try {
    win.send(data.head, data.data);
  } catch (_) { }
});

ipcMain.on('close-thread', (e, data) => {
  if (!threads[data.id]) return;
  threads[data.id].close();
  delete threads[data.id];
});
