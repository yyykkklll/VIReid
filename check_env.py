import socket
import platform
import sys
import os


def check_environment():
    hostname = socket.gethostname()
    os_type = platform.system()
    python_path = sys.executable
    cwd = os.getcwd()

    print("=" * 30)
    print("       运行环境检测")
    print("=" * 30)

    # 1. 检查主机名
    print(f"[主机名]     : {hostname}")

    # 2. 检查操作系统
    print(f"[操作系统]   : {os_type} ({platform.release()})")

    # 3. 检查 Python 解释器位置
    print(f"[Python位置] : {python_path}")

    # 4. 检查当前工作目录
    print(f"[当前目录]   : {cwd}")

    print("-" * 30)

    # 智能判断逻辑 (基于你提供的信息)
    if os_type == "Linux" and ("anaconda" in python_path or "miniconda" in python_path):
        print(">>> 结论: 正在【远程服务器】上运行 (Linux + Conda)")
    elif os_type == "Windows":
        print(">>> 结论: 正在【本地电脑】上运行 (Windows)")
    elif os_type == "Darwin":
        print(">>> 结论: 正在【本地电脑】上运行 (MacOS)")
    else:
        print(">>> 结论: 无法自动判断，请查看上方详细信息")


if __name__ == "__main__":
    check_environment()
