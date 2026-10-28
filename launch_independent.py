"""Launch a detached Windows process outside the caller's job, if permitted.

Usage: python launch_independent.py LOG_FILE PROGRAM [ARG ...]
The process writes both stdout and stderr to LOG_FILE. This is used only to
launch the already resumable summer training supervisor.
"""

import ctypes
from ctypes import wintypes
import msvcrt
import os
from pathlib import Path
import subprocess
import sys


class STARTUPINFO(ctypes.Structure):
    _fields_ = [
        ('cb', wintypes.DWORD), ('lpReserved', wintypes.LPWSTR),
        ('lpDesktop', wintypes.LPWSTR), ('lpTitle', wintypes.LPWSTR),
        ('dwX', wintypes.DWORD), ('dwY', wintypes.DWORD),
        ('dwXSize', wintypes.DWORD), ('dwYSize', wintypes.DWORD),
        ('dwXCountChars', wintypes.DWORD), ('dwYCountChars', wintypes.DWORD),
        ('dwFillAttribute', wintypes.DWORD), ('dwFlags', wintypes.DWORD),
        ('wShowWindow', wintypes.WORD), ('cbReserved2', wintypes.WORD),
        ('lpReserved2', ctypes.POINTER(ctypes.c_byte)),
        ('hStdInput', wintypes.HANDLE), ('hStdOutput', wintypes.HANDLE),
        ('hStdError', wintypes.HANDLE),
    ]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ('hProcess', wintypes.HANDLE), ('hThread', wintypes.HANDLE),
        ('dwProcessId', wintypes.DWORD), ('dwThreadId', wintypes.DWORD),
    ]


def main():
    if os.name != 'nt' or len(sys.argv) < 4:
        raise SystemExit('Windows only: LOG_FILE PROGRAM [ARG ...]')
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.CreateProcessW.argtypes = [
        wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.c_void_p, ctypes.c_void_p,
        wintypes.BOOL, wintypes.DWORD, ctypes.c_void_p, wintypes.LPCWSTR,
        ctypes.POINTER(STARTUPINFO), ctypes.POINTER(PROCESS_INFORMATION),
    ]
    kernel32.CreateProcessW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE,
                                       ctypes.POINTER(wintypes.BOOL)]
    kernel32.IsProcessInJob.restype = wintypes.BOOL

    in_job = wintypes.BOOL()
    if not kernel32.IsProcessInJob(kernel32.GetCurrentProcess(), None,
                                   ctypes.byref(in_job)):
        raise ctypes.WinError(ctypes.get_last_error())

    log_path = Path(sys.argv[1]).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(os.devnull, 'rb') as stdin, log_path.open('ab', buffering=0) as log:
        stdin_handle = msvcrt.get_osfhandle(stdin.fileno())
        log_handle = msvcrt.get_osfhandle(log.fileno())
        os.set_handle_inheritable(stdin_handle, True)
        os.set_handle_inheritable(log_handle, True)
        startup = STARTUPINFO()
        startup.cb = ctypes.sizeof(startup)
        startup.dwFlags = 0x00000100  # STARTF_USESTDHANDLES
        startup.hStdInput = stdin_handle
        startup.hStdOutput = log_handle
        startup.hStdError = log_handle
        info = PROCESS_INFORMATION()
        command = ctypes.create_unicode_buffer(subprocess.list2cmdline(sys.argv[2:]))
        flags = 0x01000000 | 0x08000000  # BREAKAWAY_FROM_JOB | NO_WINDOW
        created = kernel32.CreateProcessW(
            None, command, None, None, True, flags, None, str(Path.cwd()),
            ctypes.byref(startup), ctypes.byref(info))
        if not created:
            raise ctypes.WinError(ctypes.get_last_error())
        kernel32.CloseHandle(info.hThread)
        kernel32.CloseHandle(info.hProcess)
        print(f'pid={info.dwProcessId} launcher_in_job={bool(in_job.value)}')


if __name__ == '__main__':
    main()
