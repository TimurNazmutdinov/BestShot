"""BestShots: энергосбережение Windows для длинных заданий (анализ, поиск дублей).

Что происходит на больших архивах (десятки тысяч фото, часы работы): ноутбук уходит в «современный режим
ожидания» (Modern Standby, S0) — например, при погасшем экране или закрытой крышке, даже если обычный «сон»
отключён. В этом режиме Windows замораживает настольные программы и будит их лишь раз в ~15 минут:
вместо тысячи кадров в час обрабатывается несколько десятков. Кроме того, фоновые процессы без окна получают
режим «эффективности» (EcoQoS) и работают медленнее.

prefer_performance() вызывается в каждом рабочем процессе: отключает EcoQoS и запрашивает у системы «выполнение
необходимо» (PowerRequestExecutionRequired + SystemRequired) на всё время жизни процесса; такой запрос
не замораживается в Modern Standby. keep_awake_while() делает то же для процесса сервера (SetThreadExecutionState).
Закрытая крышка по-прежнему усыпляет ноутбук, если действие крышки в Windows — «Сон».
"""
import sys
import threading
import time

_request = {"handle": None}


def prefer_performance():
    """Отключает EcoQoS и держит запрос «выполнение необходимо» до конца процесса. Вне Windows ничего не делает."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p

        class PowerThrottling(ctypes.Structure):
            _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]

        state = PowerThrottling(1, 1, 0)  # управляем EXECUTION_SPEED и выключаем троттлинг
        k32.SetProcessInformation(ctypes.c_void_p(k32.GetCurrentProcess()), 4, ctypes.byref(state), ctypes.sizeof(state))

        class ReasonContext(ctypes.Structure):
            _fields_ = [("Version", ctypes.c_ulong), ("Flags", wintypes.DWORD), ("Reason", ctypes.c_wchar_p)]

        ctx = ReasonContext(0, 0x1, "BestShots: идёт анализ фотографий")  # POWER_REQUEST_CONTEXT_SIMPLE_STRING
        k32.PowerCreateRequest.restype = ctypes.c_void_p
        handle = k32.PowerCreateRequest(ctypes.byref(ctx))
        if handle:
            for kind in (1, 3):  # PowerRequestSystemRequired, PowerRequestExecutionRequired
                k32.PowerSetRequest(ctypes.c_void_p(handle), kind)
            _request["handle"] = handle  # запрос снимется сам при завершении процесса
    except Exception:
        pass


def keep_awake_while(is_active, poll=5.0):
    """Пока is_active() истинно, система не уходит в сон (SetThreadExecutionState из отдельного потока)."""
    if sys.platform != "win32":
        return

    def run():
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            es_continuous, es_system_required = 0x80000000, 0x00000001
            k32.SetThreadExecutionState(es_continuous | es_system_required)
            while is_active():
                time.sleep(poll)
        except Exception:
            pass
        finally:
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)  # снова разрешаем сон
            except Exception:
                pass

    threading.Thread(target=run, daemon=True, name="keep-awake").start()
