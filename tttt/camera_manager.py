"""
camera_manager.py - Robust Windows Camera Detection & Video Streaming Engine
Provides auto-discovery of all attached cameras (laptop integrated + USB webcams)
using DirectShow enumeration and dual-engine capture (DirectShow pipe + OpenCV fallback).
"""

from typing import Dict, List, Optional, Tuple
import os
import re
import subprocess
import threading
import time
import cv2
import imageio_ffmpeg
import numpy as np


class CameraDevice:
    def __init__(self, name: str, device_type: str, identifier: str):
        self.name = name
        self.device_type = device_type  # "dshow_name" or "cv2_index"
        self.identifier = identifier

    def __repr__(self):
        return f"CameraDevice(name='{self.name}', type='{self.device_type}', id='{self.identifier}')"


class CameraManager:
    @staticmethod
    def get_ffmpeg_exe() -> str:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return "ffmpeg"

    @classmethod
    def list_cameras(cls) -> List[CameraDevice]:
        """
        Discovers all available camera hardware on Windows:
        1. Enumerates DirectShow video input devices via FFmpeg.
        2. Probes OpenCV standard camera indices 0-3.
        """
        devices: List[CameraDevice] = []
        seen_names = set()

        # 1. Enumerate via FFmpeg DirectShow
        try:
            ffmpeg_exe = cls.get_ffmpeg_exe()
            cmd = [ffmpeg_exe, "-list_devices", "true", "-f", "dshow", "-i", "dummy"]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3)
            output = proc.stderr

            # Parse lines like: [dshow @ ...] "ACER HD User Facing" (video)
            matches = re.findall(r'\[dshow\s*@\s*[^\]]+\]\s*"([^"]+)"\s*\(video\)', output)
            for dev_name in matches:
                if dev_name not in seen_names:
                    seen_names.add(dev_name)
                    devices.append(CameraDevice(
                        name=dev_name,
                        device_type="dshow_name",
                        identifier=f"video={dev_name}"
                    ))
        except Exception as e:
            print(f"Error enumerating DirectShow devices: {e}")

        # 2. Probe OpenCV indices if no DirectShow devices found, or as fallback
        for idx in range(3):
            name_label = f"Camera Index {idx}"
            # Check if this index was already matched
            if any(str(idx) in d.identifier for d in devices):
                continue
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, _ = cap.read()
                    cap.release()
                    if ret:
                        devices.append(CameraDevice(
                            name=f"USB Camera (Index {idx})",
                            device_type="cv2_index",
                            identifier=str(idx)
                        ))
            except Exception:
                pass

        return devices


class CameraStream:
    """
    High-performance video streaming worker.
    Uses DirectShow subprocess pipe when given a device name (e.g. video=ACER HD User Facing),
    or OpenCV VideoCapture for numeric index.
    """
    def __init__(self):
        self.proc: Optional[subprocess.Popen] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.width = 640
        self.height = 480
        self.fps = 30.0
        self.current_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.device: Optional[CameraDevice] = None

    def start(self, device: CameraDevice, width: int = 640, height: int = 480) -> bool:
        self.stop()
        self.device = device
        self.width = width
        self.height = height

        if device.device_type == "dshow_name":
            return self._start_dshow_pipe(device.identifier, width, height)
        else:
            return self._start_cv2_cap(int(device.identifier), width, height)

    def _start_cv2_cap(self, index: int, width: int, height: int) -> bool:
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.is_running = True
        self.thread = threading.Thread(target=self._cv2_worker, daemon=True)
        self.thread.start()
        return True

    def _cv2_worker(self):
        while self.is_running and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.current_frame = frame
            time.sleep(0.01)

    def _start_dshow_pipe(self, identifier: str, width: int, height: int) -> bool:
        ffmpeg_exe = CameraManager.get_ffmpeg_exe()
        # DirectShow raw BGR frame pipe
        cmd = [
            ffmpeg_exe,
            "-f", "dshow",
            "-video_size", f"{width}x{height}",
            "-i", identifier,
            "-f", "image2pipe",
            "-pix_fmt", "bgr24",
            "-vcodec", "rawvideo",
            "-"
        ]
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=width * height * 3 * 2
            )
            self.is_running = True
            self.thread = threading.Thread(target=self._pipe_worker, daemon=True)
            self.thread.start()

            # Wait up to 1.5 seconds for first frame
            start_t = time.time()
            while time.time() - start_t < 1.5:
                if self.current_frame is not None:
                    return True
                time.sleep(0.05)
            return self.current_frame is not None or self.is_running
        except Exception as e:
            print(f"Error starting DirectShow pipe: {e}")
            return False

    def _pipe_worker(self):
        frame_bytes = self.width * self.height * 3
        while self.is_running and self.proc is not None and self.proc.stdout is not None:
            raw_data = self.proc.stdout.read(frame_bytes)
            if len(raw_data) == frame_bytes:
                frame = np.frombuffer(raw_data, dtype=np.uint8).reshape((self.height, self.width, 3))
                with self.lock:
                    self.current_frame = frame
            else:
                break
        self.is_running = False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if self.current_frame is not None:
                return True, self.current_frame.copy()
            return False, None

    def is_opened(self) -> bool:
        return self.is_running

    def stop(self):
        self.is_running = False
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=0.5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.5)
            self.thread = None

        with self.lock:
            self.current_frame = None
