"""BestShots.exe: запуск интерфейса двойным щелчком, без консоли.

Находит папку проекта (где лежит exe), запускает review.py через .venv\\Scripts\\pythonw.exe,
ждёт ответа сервера на 127.0.0.1:8765 и открывает интерфейс в окне pywebview
(если pywebview недоступен — в браузере по умолчанию).
Если сервер уже запущен, второй не стартует. При закрытии окна запущенный нами сервер останавливается.
"""
import ctypes
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from datetime import datetime
from pathlib import Path

PORT = 8765
URL = f"http://127.0.0.1:{PORT}/"
START_TIMEOUT = 90


def app_dir():
    return Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent


ROOT = app_dir()
LOG_FILE = ROOT / "logs" / "bestshots.log"


def log(text):
    try:
        LOG_FILE.parent.mkdir(exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S,000} INFO launcher: {text}\n")
    except OSError:
        pass


def message(text, icon=0x10):
    """Окно сообщения Windows (0x10 — ошибка, 0x40 — информация)."""
    ctypes.windll.user32.MessageBoxW(0, text, "BestShots", icon)


def request(path, data=None, timeout=2.0):
    headers = {"Origin": f"http://127.0.0.1:{PORT}"}
    body = None
    if data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(URL + path.lstrip("/"), data=body, headers=headers)
    return urllib.request.urlopen(req, timeout=timeout).read()


def port_state():
    """free — порт свободен, ours — отвечает BestShots, foreign — занят другой программой."""
    with socket.socket() as s:
        s.settimeout(1.0)
        if s.connect_ex(("127.0.0.1", PORT)) != 0:
            return "free"
    try:
        return "ours" if "events" in json.loads(request("api/events")) else "foreign"
    except Exception:
        return "foreign"


def start_server():
    py = ROOT / ".venv" / "Scripts" / "pythonw.exe"
    review = ROOT / "review.py"
    if not py.exists():
        message(f"Не найдено окружение Python: {py}\n\nСоздайте его по инструкции в README.md "
                f"(раздел «Установка»): python -m venv .venv и pip install -r requirements.txt.")
        return None
    if not review.exists():
        message(f"Не найден файл {review}.\nBestShots.exe должен лежать в папке проекта рядом с review.py.")
        return None
    proc = subprocess.Popen([str(py), str(review), "--no-browser", "--port", str(PORT)], cwd=str(ROOT),
                            creationflags=subprocess.CREATE_NO_WINDOW)
    log(f"сервер запущен, pid {proc.pid}")
    deadline = time.time() + START_TIMEOUT
    while time.time() < deadline:
        if proc.poll() is not None:
            message(f"Сервер BestShots не запустился (код {proc.returncode}).\n"
                    f"Возможно, порт {PORT} занят.\n\nПодробности: {LOG_FILE}")
            return None
        if port_state() == "ours":
            return proc
        time.sleep(0.4)
    proc.kill()
    message(f"Сервер BestShots не ответил за {START_TIMEOUT} с.\n\nПодробности: {LOG_FILE}")
    return None


def stop_server(proc):
    try:
        request("api/shutdown", {}, timeout=3)
        proc.wait(timeout=10)
        return
    except Exception:
        pass
    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True,
                   creationflags=subprocess.CREATE_NO_WINDOW)


def open_window(owned):
    try:
        import webview
    except Exception as e:
        log(f"pywebview недоступен ({e}), открываю браузер")
        webbrowser.open(URL)
        return False
    window = webview.create_window("BestShots", URL, width=1400, height=900, resizable=True, min_size=(900, 600))
    if owned:
        def watch():  # сервер остановили из интерфейса («Выход») — закрываем окно
            while owned.poll() is None:
                time.sleep(1)
            try:
                window.destroy()
            except Exception:
                pass
        threading.Thread(target=watch, daemon=True).start()
    webview.start(private_mode=False, storage_path=str(ROOT / "logs" / "webview"))
    return True


def main():
    os.chdir(ROOT)
    state = port_state()
    if state == "foreign":
        message(f"Порт {PORT} занят другой программой, BestShots не может запуститься.\n"
                f"Закройте эту программу и повторите.\n\nПодробности: {LOG_FILE}")
        return
    owned = None
    if state == "free":
        owned = start_server()
        if owned is None:
            return
    else:
        log("сервер уже запущен, открываю окно")
    try:
        shown = open_window(owned)
    except Exception as e:
        log(f"ошибка окна: {type(e).__name__}: {e}")
        webbrowser.open(URL)
        shown = False
    if owned and shown:
        stop_server(owned)
        log("сервер остановлен")
    elif owned:
        message("Интерфейс открыт в браузере. Чтобы остановить BestShots, нажмите «Выход» в интерфейсе.", 0x40)


if __name__ == "__main__":
    main()
