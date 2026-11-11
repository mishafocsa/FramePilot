"""Narrow Windows UAC helper launcher. Payloads contain names/state, never shell commands or paths."""

import ctypes
import subprocess
import sys

from .windows_camera_names import CameraNameError


class ShellExecuteInfo(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_uint32),
        ("fMask", ctypes.c_uint32),
        ("hwnd", ctypes.c_void_p),
        ("lpVerb", ctypes.c_wchar_p),
        ("lpFile", ctypes.c_wchar_p),
        ("lpParameters", ctypes.c_wchar_p),
        ("lpDirectory", ctypes.c_wchar_p),
        ("nShow", ctypes.c_int32),
        ("hInstApp", ctypes.c_void_p),
        ("lpIDList", ctypes.c_void_p),
        ("lpClass", ctypes.c_wchar_p),
        ("hkeyClass", ctypes.c_void_p),
        ("dwHotKey", ctypes.c_uint32),
        ("hIcon", ctypes.c_void_p),
        ("hProcess", ctypes.c_void_p),
    ]


def helper_arguments(token, frozen=None):
    frozen = getattr(sys, "frozen", False) if frozen is None else frozen
    args = [] if frozen else ["-B", "-m", "framepilot"]
    return [*args, "--camera-name-task", token]


def is_admin():
    if sys.platform != "win32":
        return False
    shell = ctypes.WinDLL("shell32", use_last_error=True)
    shell.IsUserAnAdmin.argtypes = []
    shell.IsUserAnAdmin.restype = ctypes.c_int32
    return bool(shell.IsUserAnAdmin())


def run_elevated(token, hwnd=0):
    if sys.platform != "win32":
        raise CameraNameError(3, "Изменение имени камеры доступно только в Windows.")
    shell = ctypes.WinDLL("shell32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    ole = ctypes.WinDLL("ole32", use_last_error=True)
    ole.CoInitializeEx.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    ole.CoInitializeEx.restype = ctypes.c_int32
    ole.CoUninitialize.argtypes = []
    ole.CoUninitialize.restype = None
    shell.ShellExecuteExW.argtypes = [ctypes.POINTER(ShellExecuteInfo)]
    shell.ShellExecuteExW.restype = ctypes.c_int32
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.WaitForSingleObject.restype = ctypes.c_uint32
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    kernel.GetExitCodeProcess.restype = ctypes.c_int32
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = ctypes.c_int32
    info = ShellExecuteInfo()
    info.cbSize = ctypes.sizeof(info)
    info.fMask = 0x40 | 0x100  # NOCLOSEPROCESS | NOASYNC; retain process handle for its exit code.
    info.hwnd, info.lpVerb, info.lpFile = hwnd, "runas", sys.executable
    info.lpParameters = subprocess.list2cmdline(helper_arguments(token))
    info.nShow = 0
    hr = ole.CoInitializeEx(None, 0x2 | 0x4)  # Apartment threaded; disable OLE1 DDE.
    if hr < 0 and hr != -2147417850:  # RPC_E_CHANGED_MODE: keep the existing COM apartment.
        raise CameraNameError(5, "Не удалось подготовить запрос прав Windows.")
    try:
        if not shell.ShellExecuteExW(ctypes.byref(info)):
            error = ctypes.get_last_error()
            raise CameraNameError(1223 if error == 1223 else 5)
        if not info.hProcess:
            raise CameraNameError(5, "Windows не вернула процесс изменения имени камеры.")
        while True:
            status = kernel.WaitForSingleObject(info.hProcess, 1000)
            if status == 0:
                break
            if status != 258:
                raise CameraNameError(5, "Не удалось получить результат изменения имени камеры.")
        code = ctypes.c_uint32()
        if not kernel.GetExitCodeProcess(info.hProcess, ctypes.byref(code)):
            raise CameraNameError(5)
        return int(code.value)
    finally:
        if info.hProcess:
            kernel.CloseHandle(info.hProcess)
        if hr >= 0:
            ole.CoUninitialize()
