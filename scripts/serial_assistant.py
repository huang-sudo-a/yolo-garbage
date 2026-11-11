"""Chinese receiver UI: local simulation without drivers, or real COM reception."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import queue
import subprocess
import sys
import time
import tkinter as tk
from tkinter import messagebox, ttk

from serial.tools import list_ports

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from invc_vision.protocol import CATEGORY_NAME
from invc_vision.serial_receiver import SerialReceiver


class Assistant:
    def __init__(self, root: tk.Tk, listen: int = 8765) -> None:
        self.root, self.listen = root, listen
        self.receiver = None
        self.events = queue.Queue(maxsize=300)
        self.sender = None
        self._sender_log = None
        self._stop_file = None
        self._last_received_at = 0
        self._closing = False
        self.output = PROJECT / "outputs/serial" / datetime.now().strftime("assistant_%Y%m%d_%H%M%S_%f")
        self.output.mkdir(parents=True, exist_ok=True)
        root.title("INVC 串口接收助手")
        root.geometry("1120x770")
        root.minsize(980, 660)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TLabel", font=("Microsoft YaHei UI", 10))
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=6)
        style.configure("Treeview", font=("Microsoft YaHei UI", 10), rowheight=28)
        body = ttk.Frame(root, padding=16)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="INVC 通信接收助手", font=("Microsoft YaHei UI", 19, "bold")).pack(anchor="w")
        ttk.Label(body, text="本机模拟无需下位机；COM 接收需要实际连接的另一端。两种模式使用同一套协议。", foreground="#996000").pack(anchor="w", pady=(4, 12))
        settings = ttk.Frame(body)
        settings.pack(fill="x")
        self.mode = tk.StringVar(value="local")
        ttk.Radiobutton(settings, text="本机模拟", variable=self.mode, value="local").pack(side="left")
        ttk.Radiobutton(settings, text="COM 接收", variable=self.mode, value="serial").pack(side="left", padx=(12, 4))
        self.port = ttk.Combobox(settings, values=[p.device for p in list_ports.comports()], width=10)
        self.port.pack(side="left")
        self.port.set("COM1")
        ttk.Label(settings, text="波特率").pack(side="left", padx=(14, 4))
        self.baud = ttk.Combobox(settings, values=[115200, 230400, 460800, 921600], width=10)
        self.baud.set("460800")
        self.baud.pack(side="left")
        ttk.Button(settings, text="开启接收", command=self.start_receiver).pack(side="left", padx=10)
        ttk.Button(settings, text="关闭接收", command=self.stop_receiver).pack(side="left")
        controls = ttk.Frame(body)
        controls.pack(fill="x", pady=12)
        ttk.Button(controls, text="启动摄像头识别", command=lambda: self.start_sender("camera")).pack(side="left")
        ttk.Button(controls, text="回放历史识别结果", command=lambda: self.start_sender("replay")).pack(side="left", padx=10)
        ttk.Button(controls, text="停止发送", command=self.stop_sender).pack(side="left")
        self.status = tk.StringVar(value="等待开启接收")
        ttk.Label(body, textvariable=self.status, foreground="#145a7c", wraplength=1060).pack(anchor="w")
        self.source = tk.StringVar(value="数据来源：尚未收到数据")
        ttk.Label(body, textvariable=self.source, font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w", pady=(8, 4))
        self.metrics = tk.StringVar(value="有效接收 0 帧　校验错误 0　通信序号缺口 0")
        ttk.Label(body, textvariable=self.metrics).pack(anchor="w", pady=(0, 10))
        columns = ("id", "class", "category", "xy", "wh", "conf", "speed")
        self.table = ttk.Treeview(body, columns=columns, show="headings", height=7)
        for key, title, width in zip(columns, ("目标 ID", "物品", "垃圾大类", "左上角 (像素)", "宽 × 高 (像素)", "置信度", "速度"), (75, 95, 120, 145, 145, 90, 190)):
            self.table.heading(key, text=title)
            self.table.column(key, width=width, anchor="center")
        self.table.pack(fill="x")
        ttk.Label(body, text="最近接收的数据（CRC 校验通过后显示）").pack(anchor="w", pady=(12, 4))
        self.raw = tk.Text(body, height=8, wrap="char", font=("Consolas", 10), bg="#f3f6f8")
        self.raw.pack(fill="both", expand=True)
        self.raw.configure(state="disabled")
        ttk.Label(body, text="位置以画面左上角为原点；速度尚未接入。超过 2 秒未收到数据时，清空当前目标。").pack(anchor="w", pady=(8, 2))
        ttk.Label(body, text=f"日志：{self.output}", font=("Microsoft YaHei UI", 9)).pack(anchor="w")
        self.names = {0: "瓷片", 1: "石子", 2: "砖块", 3: "纸杯", 4: "土豆", 5: "塑料瓶"}
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self.poll)
        self.start_receiver()

    def event(self, kind, payload):
        try:
            self.events.put_nowait((kind, payload))
        except queue.Full:
            # All packets are still written to disk by the receiver worker.
            try:
                self.events.get_nowait()
            except queue.Empty:
                pass
            self.events.put_nowait((kind, payload))

    def start_receiver(self):
        if self.receiver is not None:
            self.status.set("接收已开启；切换模式前请先关闭接收")
            return
        try:
            baudrate = int(self.baud.get())
            if baudrate <= 0:
                raise ValueError("波特率必须为正")
            stamp = datetime.now().strftime("receive_%H%M%S_%f")
            self.receiver = SerialReceiver(self.output / stamp, self.event,
                                           listen_port=self.listen if self.mode.get() == "local" else None,
                                           port=None if self.mode.get() == "local" else self.port.get(),
                                           baudrate=baudrate)
            self.receiver.start()
        except (ValueError, OSError) as exc:
            self.receiver = None
            messagebox.showerror("接收未开启", str(exc))

    def stop_receiver(self):
        if self.receiver:
            self.receiver.close()
            self.receiver = None
        self.clear_targets()
        self._last_received_at = 0
        self.status.set("接收已关闭")

    def start_sender(self, kind):
        if self.sender is not None and self.sender.poll() is None:
            self.status.set("已有发送程序运行，请先点击“停止发送”")
            return
        if not self.receiver or self.receiver.listen_port is None:
            self.status.set("演示按钮需要先开启本机模拟接收。COM 模式请从实际对接的另一端发送。")
            return
        if not self.receiver.ready.is_set() or not self.receiver._thread.is_alive():
            self.status.set("接收端尚未就绪，请检查是否端口被其他助手占用")
            return
        destination = f"socket://127.0.0.1:{self.receiver.listen_port}"
        stamp = datetime.now().strftime("%H%M%S_%f")
        self._stop_file = self.output / f"stop_{stamp}.flag"
        common = ["--stop-file", str(self._stop_file)]
        if kind == "camera":
            command = [sys.executable, "-X", "utf8", "-u", str(PROJECT / "scripts/run_live.py"), "--serial-port", destination,
                       "--log", str(self.output / f"camera_{stamp}.jsonl"), *common]
            self.source.set("数据来源：正在启动当前 USB 摄像头识别")
        else:
            command = [sys.executable, "-X", "utf8", "-u", str(PROJECT / "scripts/replay_serial.py"), "--port", destination,
                       "--loop", "--log", str(self.output / f"replay_{stamp}.jsonl"), *common]
            self.source.set("数据来源：历史识别日志回放（不是当前摄像头）")
        if self._sender_log:
            self._sender_log.close()
        self._sender_log = (self.output / f"sender_{stamp}.log").open("w", encoding="utf-8")
        self.sender = subprocess.Popen(command, cwd=PROJECT, stdout=self._sender_log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        self._last_received_at = 0
        self.status.set("正在启动发送程序，首次加载模型需要稍等")

    def stop_sender(self):
        if self.sender is not None and self.sender.poll() is None:
            self._stop_file.touch()
            try:
                self.sender.wait(timeout=7)
            except subprocess.TimeoutExpired:
                # Only our own child is terminated if its capture/driver does not return.
                self.sender.terminate()
                self.sender.wait(timeout=3)
        self.sender = None
        if self._sender_log:
            self._sender_log.close()
            self._sender_log = None
        self.status.set("发送已停止")

    def clear_targets(self):
        for row in self.table.get_children():
            self.table.delete(row)

    def show_frame(self, p):
        self._last_received_at = time.monotonic()
        self.source.set("数据来源："+("历史识别日志回放（不是当前摄像头）" if p["src"] == "recording_replay" else "当前识别程序输出"))
        self.clear_targets()
        for d in p["d"]:
            speed = "未启用" if p["u"] is None else f"瞬时 {d[8]} / 平均 {d[9]} {p['u']}"
            self.table.insert("", "end", values=(d[0] if d[0] is not None else "—", self.names.get(d[1], f"类别{d[1]}"),
                                                 CATEGORY_NAME[d[2]], f"{d[3]}, {d[4]}", f"{d[5]} × {d[6]}", f"{d[7]:.1%}", speed))
        self.raw.configure(state="normal")
        self.raw.insert("end", json.dumps(p, ensure_ascii=False, separators=(",", ":"))+"\n")
        if int(self.raw.index("end-1c").split(".")[0]) > 100:
            self.raw.delete("1.0", "20.0")
        self.raw.see("end")
        self.raw.configure(state="disabled")

    def poll(self):
        if self._closing:
            return
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "frame":
                    self.show_frame(payload)
                    self.status.set(f"正在接收 · 通信帧 {payload['s']} / 识别帧 {payload['q']} · 目标 {len(payload['d'])} 个 · FPS {payload['f']} · 推理 {payload['ms']} ms")
                else:
                    self.status.set(payload)
        except queue.Empty:
            pass
        if self.receiver:
            s = self.receiver.snapshot()
            self.metrics.set(f"有效接收 {s['received']} 帧　CRC 错误 {s['bad_crc']}　格式错误 {s['bad_format']}　通信序号缺口 {s['missing']}　重复 {s['duplicates']}　乱序 {s['out_of_order']}")
        if self._last_received_at and time.monotonic()-self._last_received_at > 2:
            self.clear_targets()
            self.status.set("超过 2 秒未收到新数据，已清空当前目标；最近数据仅供查看")
        if self.sender is not None and self.sender.poll() is not None:
            code = self.sender.returncode
            self.sender = None
            self.status.set(f"发送程序已结束，返回码 {code}。详细信息见日志文件。")
        self.root.after(100, self.poll)

    def close(self):
        self._closing = True
        self.stop_sender()
        self.stop_receiver()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="INVC 中文串口接收助手")
    parser.add_argument("--listen", type=int, default=8765)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--replay", action="store_true")
    mode.add_argument("--camera", action="store_true")
    args = parser.parse_args()
    root = tk.Tk()
    assistant = Assistant(root, args.listen)
    if args.replay or args.camera:
        root.after(700, lambda: assistant.start_sender("camera" if args.camera else "replay"))
    root.mainloop()


if __name__ == "__main__":
    main()
