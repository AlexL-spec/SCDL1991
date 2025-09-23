# src/common/io_utils.py
from pathlib import Path, PureWindowsPath
import os
import platform

def normalize(path_str: str, verbose=False) -> str:
    path_str = path_str.strip().strip('"').strip("'")
    path_str = path_str.replace('\\', '/')

    if platform.system() == 'Linux' and len(path_str) > 2 and path_str[1:3] == ':/':
        drive_letter = path_str[0].lower()
        rest = path_str[2:]
        path_str = f"/mnt/{drive_letter}{rest}"

    path = Path(path_str).resolve()
    if verbose:
        print(f"Normalized path: {path_str} -> {path}")
    return str(path)
