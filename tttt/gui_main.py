"""
gui_main.py - Modern Scientific Desktop GUI for MazeTrack v2.0
Built with PyQt5, featuring:
1. Interactive Video Canvas with 8 drag/resize handles for effortless Arena positioning.
2. Custom Free Zones (Arbitrary Polygons, Rectangles, Circles) with names and colors.
3. Robust Camera Auto-Discovery & DirectShow/OpenCV streaming connection dialog.
4. User-Controlled Tracking Workflow (Setup Mode -> Start Tracking -> Pause/Stop/Export).
5. Open Field Grid division (N x M), 1-click Auto-labeling, and interactive cell painting.
6. 2-point scale calibration tool (pixels to real cm).
7. Live scientific analytics HUD, real-time zone statistics table, and export to CSV/PNG.
"""

from typing import Dict, List, Optional, Tuple
import math
import os
import sys
import cv2
import numpy as np

from PyQt5.QtCore import Qt, QTimer, QPoint, QRectF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QCursor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QSlider, QSpinBox, QDoubleSpinBox,
    QComboBox, QRadioButton, QButtonGroup, QCheckBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QFileDialog, QInputDialog, QMessageBox,
    QGroupBox, QFrame, QSplitter, QStatusBar, QToolBar, QAction, QDialog,
    QColorDialog, QTabWidget, QListWidget, QListWidgetItem
)

from arena_grid import TrackingArena, OpenFieldGridArena, LABEL_COLORS_HEX, LABEL_COLORS_BGR
from arena_presets import MorrisWaterMazeArena, ElevatedPlusMazeArena
from tracker_engine import AnimalTracker
from simulation import RodentSimulation
from export_manager import ExportManager
from custom_zones import CustomZone, CustomZoneManager
from camera_manager import CameraManager, CameraStream, CameraDevice


DARK_STYLESHEET = """
QMainWindow {
    background-color: #0b0f19;
}
QWidget {
    background-color: #0b0f19;
    color: #e2e8f0;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 12px;
}
QGroupBox {
    background-color: #131b2e;
    border: 1px solid #1e293b;
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 10px;
    font-weight: bold;
    color: #38bdf8;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
}
QTabWidget::pane {
    border: 1px solid #1e293b;
    background-color: #131b2e;
    border-radius: 6px;
}
QTabBar::tab {
    background: #0f172a;
    color: #94a3b8;
    padding: 6px 12px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-weight: 600;
}
QTabBar::tab:selected {
    background: #1e293b;
    color: #38bdf8;
    border-bottom: 2px solid #38bdf8;
}
QPushButton {
    background-color: #1e293b;
    color: #f8fafc;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #334155;
    border-color: #38bdf8;
}
QPushButton:pressed {
    background-color: #0284c7;
    border-color: #0284c7;
}
QPushButton#startBtn {
    background-color: #059669;
    border-color: #10b981;
    color: #ffffff;
    font-size: 13px;
    font-weight: bold;
    padding: 8px 16px;
}
QPushButton#startBtn:hover {
    background-color: #047857;
}
QPushButton#pauseBtn {
    background-color: #d97706;
    border-color: #f59e0b;
    color: #ffffff;
    font-size: 13px;
    font-weight: bold;
    padding: 8px 16px;
}
QPushButton#pauseBtn:hover {
    background-color: #b45309;
}
QPushButton#stopBtn {
    background-color: #be123c;
    border-color: #f43f5e;
    color: #ffffff;
    font-size: 13px;
    font-weight: bold;
    padding: 8px 16px;
}
QPushButton#stopBtn:hover {
    background-color: #9f1239;
}
QPushButton#primaryBtn {
    background-color: #0284c7;
    border-color: #38bdf8;
    color: #ffffff;
}
QPushButton#primaryBtn:hover {
    background-color: #0369a1;
}
QPushButton#exportBtn {
    background-color: #047857;
    border-color: #10b981;
    color: #ffffff;
}
QPushButton#exportBtn:hover {
    background-color: #065f46;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #1e293b;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #0284c7;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #38bdf8;
    width: 14px;
    margin-top: -4px;
    margin-bottom: -4px;
    border-radius: 7px;
}
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 4px 6px;
    color: #f8fafc;
}
QTableWidget, QListWidget {
    background-color: #131b2e;
    border: 1px solid #1e293b;
    border-radius: 6px;
    gridline-color: #1e293b;
}
QHeaderView::section {
    background-color: #1e293b;
    color: #94a3b8;
    padding: 4px 6px;
    border: 1px solid #0f172a;
    font-weight: 600;
}
QTableWidget::item, QListWidget::item {
    padding: 4px;
    border-bottom: 1px solid #1e293b;
}
QTableWidget::item:selected, QListWidget::item:selected {
    background-color: #0284c7;
    color: #ffffff;
}
QRadioButton {
    spacing: 6px;
    font-weight: 500;
}
QRadioButton::indicator {
    width: 14px;
    height: 14px;
    border-radius: 7px;
    border: 2px solid #64748b;
    background: #1e293b;
}
QRadioButton::indicator:checked {
    background: #38bdf8;
    border-color: #38bdf8;
}
QCheckBox {
    spacing: 6px;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border-radius: 3px;
    border: 1px solid #64748b;
    background: #1e293b;
}
QCheckBox::indicator:checked {
    background: #0284c7;
    border-color: #38bdf8;
}
"""


class CameraSelectDialog(QDialog):
    """Clean modal dialog for discovering and selecting cameras."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Koneksi Kamera / Webcam")
        self.setFixedWidth(460)
        self.setStyleSheet(DARK_STYLESHEET)
        self.selected_device: Optional[CameraDevice] = None
        self.selected_width = 640
        self.selected_height = 480

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        layout.addWidget(QLabel("Pilih Kamera yang Terpasang:"))
        self.combo_cams = QComboBox()
        layout.addWidget(self.combo_cams)

        # Resolution row
        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("Resolusi:"))
        self.combo_res = QComboBox()
        self.combo_res.addItems(["640 x 480 (Standar / Cepat)", "1280 x 720 (HD)"])
        res_row.addWidget(self.combo_res)
        layout.addLayout(res_row)

        self.btn_refresh = QPushButton("🔄 Pindai Ulang Kamera (Refresh)")
        self.btn_refresh.clicked.connect(self.scan_cameras)
        layout.addWidget(self.btn_refresh)

        btn_row = QHBoxLayout()
        self.btn_connect = QPushButton("Sambungkan Kamera")
        self.btn_connect.setObjectName("primaryBtn")
        self.btn_connect.clicked.connect(self.on_connect)
        self.btn_cancel = QPushButton("Batal")
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_connect)
        btn_row.addWidget(self.btn_cancel)
        layout.addLayout(btn_row)

        self.scan_cameras()

    def scan_cameras(self):
        self.combo_cams.clear()
        self.devices = CameraManager.list_cameras()
        if not self.devices:
            self.combo_cams.addItem("Tidak ada kamera terdeteksi")
            self.btn_connect.setEnabled(False)
        else:
            for dev in self.devices:
                self.combo_cams.addItem(f"📷 {dev.name} ({dev.device_type})", dev)
            self.btn_connect.setEnabled(True)

    def on_connect(self):
        if not self.devices:
            return
        idx = self.combo_cams.currentIndex()
        if 0 <= idx < len(self.devices):
            self.selected_device = self.devices[idx]
            if "1280" in self.combo_res.currentText():
                self.selected_width, self.selected_height = 1280, 720
            else:
                self.selected_width, self.selected_height = 640, 480
            self.accept()


class VideoCanvas(QWidget):
    """
    High-performance interactive video canvas with:
    - 8 Drag handles on the Arena overlay for intuitive mouse resizing & moving
    - Interactive Arena Drawing: Rectangle ("DRAW_ARENA_RECT") & Ellipse ("DRAW_ARENA_ELLIPSE")
    - Custom Free Zones drawing (Polygon, Rectangle, Circle)
    - Scale Calibration tool
    - Cell painting for Open Field Grid
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.setMouseTracking(True)

        self.current_frame_bgr: Optional[np.ndarray] = None
        self.pixmap: Optional[QPixmap] = None

        # Interaction mode: "ADJUST_ARENA", "DRAW_ARENA_RECT", "DRAW_ARENA_ELLIPSE", "PAINT_CELL",
        # "DRAW_CUSTOM_RECT", "DRAW_CUSTOM_CIRCLE", "DRAW_CUSTOM_POLY", "CALIBRATE", "NONE"
        self.interaction_mode = "ADJUST_ARENA"
        self.active_paint_label = "Corner"

        # Active handle being dragged: None, "TL", "TR", "BL", "BR", "T", "B", "L", "R", "CENTER"
        self.drag_handle = None
        self.drag_start_norm = None
        self.original_bbox_norm = None

        # Drawing state for shapes
        self.draw_p1_norm: Optional[Tuple[float, float]] = None
        self.draw_p2_norm: Optional[Tuple[float, float]] = None
        self.poly_points_norm: List[Tuple[float, float]] = []

        # Scale calibration
        self.calib_p1_px: Optional[Tuple[int, int]] = None
        self.calib_p2_px: Optional[Tuple[int, int]] = None

        # Callbacks
        self.on_arena_bbox_changed = None
        self.on_cell_clicked = None
        self.on_calib_completed = None
        self.on_custom_zone_drawn = None

        # Reference to arena object and custom zones manager (set by parent)
        self.grid_arena: Optional[TrackingArena] = None
        self.custom_zones_mgr: Optional[CustomZoneManager] = None

    def set_frame(self, frame_bgr: Optional[np.ndarray]):
        if frame_bgr is None:
            self.current_frame_bgr = None
            self.pixmap = None
            self.update()
            return

        self.current_frame_bgr = frame_bgr
        h, w, ch = frame_bgr.shape
        bytes_per_line = ch * w
        q_img = QImage(frame_bgr.data, w, h, bytes_per_line, QImage.Format_BGR888)
        self.pixmap = QPixmap.fromImage(q_img)
        self.update()

    def get_image_coords(self, mouse_pos: QPoint) -> Optional[Tuple[int, int, float, float]]:
        """Returns (px_x, px_y, norm_x, norm_y) or None."""
        if self.current_frame_bgr is None or self.pixmap is None:
            return None

        img_h, img_w = self.current_frame_bgr.shape[:2]
        w_w = self.width()
        w_h = self.height()

        scale = min(w_w / img_w, w_h / img_h)
        disp_w = int(img_w * scale)
        disp_h = int(img_h * scale)
        offset_x = (w_w - disp_w) // 2
        offset_y = (w_h - disp_h) // 2

        mx = mouse_pos.x() - offset_x
        my = mouse_pos.y() - offset_y

        if 0 <= mx <= disp_w and 0 <= my <= disp_h:
            orig_x = int(mx / scale)
            orig_y = int(my / scale)
            norm_x = max(0.0, min(1.0, orig_x / img_w))
            norm_y = max(0.0, min(1.0, orig_y / img_h))
            return orig_x, orig_y, norm_x, norm_y
        return None

    def get_handle_under_mouse(self, norm_x: float, norm_y: float) -> Optional[str]:
        """Detects if cursor is hovering over any of the 8 bounding box resize handles."""
        if self.grid_arena is None or not getattr(self.grid_arena, "is_defined", True):
            return None

        bx, by, bw, bh = self.grid_arena.norm_bbox
        tol_x = 0.025
        tol_y = 0.025

        cx = bx + bw / 2
        cy = by + bh / 2

        # 4 Corners
        if abs(norm_x - bx) < tol_x and abs(norm_y - by) < tol_y:
            return "TL"
        if abs(norm_x - (bx + bw)) < tol_x and abs(norm_y - by) < tol_y:
            return "TR"
        if abs(norm_x - bx) < tol_x and abs(norm_y - (by + bh)) < tol_y:
            return "BL"
        if abs(norm_x - (bx + bw)) < tol_x and abs(norm_y - (by + bh)) < tol_y:
            return "BR"

        # 4 Edges
        if abs(norm_x - cx) < tol_x and abs(norm_y - by) < tol_y:
            return "T"
        if abs(norm_x - cx) < tol_x and abs(norm_y - (by + bh)) < tol_y:
            return "B"
        if abs(norm_x - bx) < tol_x and abs(norm_y - cy) < tol_y:
            return "L"
        if abs(norm_x - (bx + bw)) < tol_x and abs(norm_y - cy) < tol_y:
            return "R"

        # Inside Move
        if bx < norm_x < bx + bw and by < norm_y < by + bh:
            return "CENTER"

        return None

    def mousePressEvent(self, event):
        coords = self.get_image_coords(event.pos())
        if coords is None:
            return
        px_x, px_y, norm_x, norm_y = coords

        # 1. ADJUST_ARENA: Drag & resize handles
        if self.interaction_mode == "ADJUST_ARENA":
            handle = self.get_handle_under_mouse(norm_x, norm_y)
            if handle is not None:
                self.drag_handle = handle
                self.drag_start_norm = (norm_x, norm_y)
                self.original_bbox_norm = list(self.grid_arena.norm_bbox)
                return

        # 2. DRAW_ARENA_RECT / DRAW_ARENA_ELLIPSE: Drag to define arena from scratch
        elif self.interaction_mode in ("DRAW_ARENA_RECT", "DRAW_ARENA_ELLIPSE"):
            self.draw_p1_norm = (norm_x, norm_y)
            self.draw_p2_norm = (norm_x, norm_y)
            self.update()
            return

        # 3. PAINT_CELL: Click to assign cell label
        elif self.interaction_mode == "PAINT_CELL":
            if self.on_cell_clicked:
                self.on_cell_clicked(px_x, px_y, self.active_paint_label)
            return

        # 4. DRAW_CUSTOM_RECT / CIRCLE:
        elif self.interaction_mode in ("DRAW_CUSTOM_RECT", "DRAW_CUSTOM_CIRCLE"):
            self.draw_p1_norm = (norm_x, norm_y)
            self.draw_p2_norm = (norm_x, norm_y)
            self.update()
            return

        # 5. DRAW_CUSTOM_POLY:
        elif self.interaction_mode == "DRAW_CUSTOM_POLY":
            if event.button() == Qt.LeftButton:
                self.poly_points_norm.append((norm_x, norm_y))
                self.update()
            elif event.button() == Qt.RightButton:
                # Finish polygon on right click
                if len(self.poly_points_norm) >= 3:
                    if self.on_custom_zone_drawn:
                        self.on_custom_zone_drawn("polygon", self.poly_points_norm.copy())
                self.poly_points_norm.clear()
                self.interaction_mode = "ADJUST_ARENA"
                self.update()
            return

        # 6. CALIBRATE: 2-point ruler
        elif self.interaction_mode == "CALIBRATE":
            if self.calib_p1_px is None:
                self.calib_p1_px = (px_x, px_y)
                self.calib_p2_px = None
            else:
                self.calib_p2_px = (px_x, px_y)
                if self.on_calib_completed:
                    self.on_calib_completed(self.calib_p1_px, self.calib_p2_px)
                self.calib_p1_px = None
                self.calib_p2_px = None
                self.interaction_mode = "ADJUST_ARENA"
            self.update()
            return

    def mouseMoveEvent(self, event):
        coords = self.get_image_coords(event.pos())
        if coords is None:
            self.setCursor(Qt.ArrowCursor)
            return
        px_x, px_y, norm_x, norm_y = coords

        # Update cursor when hovering over arena handles
        if self.interaction_mode == "ADJUST_ARENA":
            if self.drag_handle is None:
                h = self.get_handle_under_mouse(norm_x, norm_y)
                if h in ("TL", "BR"):
                    self.setCursor(Qt.SizeFDiagCursor)
                elif h in ("TR", "BL"):
                    self.setCursor(Qt.SizeBDiagCursor)
                elif h in ("T", "B"):
                    self.setCursor(Qt.SizeVerCursor)
                elif h in ("L", "R"):
                    self.setCursor(Qt.SizeHorCursor)
                elif h == "CENTER":
                    self.setCursor(Qt.SizeAllCursor)
                else:
                    self.setCursor(Qt.ArrowCursor)
            else:
                # Actively dragging handle
                dx = norm_x - self.drag_start_norm[0]
                dy = norm_y - self.drag_start_norm[1]
                ox, oy, ow, oh = self.original_bbox_norm

                nx, ny, nw, nh = ox, oy, ow, oh
                if self.drag_handle == "CENTER":
                    nx = max(0.0, min(1.0 - ow, ox + dx))
                    ny = max(0.0, min(1.0 - oh, oy + dy))
                elif self.drag_handle == "TL":
                    nx = min(ox + ow - 0.05, ox + dx)
                    ny = min(oy + oh - 0.05, oy + dy)
                    nw = (ox + ow) - nx
                    nh = (oy + oh) - ny
                elif self.drag_handle == "BR":
                    nw = max(0.05, min(1.0 - ox, ow + dx))
                    nh = max(0.05, min(1.0 - oy, oh + dy))
                elif self.drag_handle == "TR":
                    ny = min(oy + oh - 0.05, oy + dy)
                    nw = max(0.05, min(1.0 - ox, ow + dx))
                    nh = (oy + oh) - ny
                elif self.drag_handle == "BL":
                    nx = min(ox + ow - 0.05, ox + dx)
                    nw = (ox + ow) - nx
                    nh = max(0.05, min(1.0 - oy, oh + dy))
                elif self.drag_handle == "T":
                    ny = min(oy + oh - 0.05, oy + dy)
                    nh = (oy + oh) - ny
                elif self.drag_handle == "B":
                    nh = max(0.05, min(1.0 - oy, oh + dy))
                elif self.drag_handle == "L":
                    nx = min(ox + ow - 0.05, ox + dx)
                    nw = (ox + ow) - nx
                elif self.drag_handle == "R":
                    nw = max(0.05, min(1.0 - ox, ow + dx))

                self.grid_arena.norm_bbox = [
                    max(0.0, min(0.95, nx)),
                    max(0.0, min(0.95, ny)),
                    max(0.05, min(1.0, nw)),
                    max(0.05, min(1.0, nh))
                ]
                if self.on_arena_bbox_changed:
                    self.on_arena_bbox_changed()
                self.update()

        # Update rubber-band drawing
        elif self.interaction_mode in ("DRAW_ARENA_RECT", "DRAW_ARENA_ELLIPSE", "DRAW_CUSTOM_RECT", "DRAW_CUSTOM_CIRCLE"):
            if self.draw_p1_norm is not None:
                self.draw_p2_norm = (norm_x, norm_y)
                self.update()

        elif self.interaction_mode == "CALIBRATE" and self.calib_p1_px is not None:
            self.calib_p2_px = (px_x, px_y)
            self.update()

    def mouseReleaseEvent(self, event):
        coords = self.get_image_coords(event.pos())
        if coords is None:
            self.drag_handle = None
            self.draw_p1_norm = None
            return
        _, _, norm_x, norm_y = coords

        if self.interaction_mode == "ADJUST_ARENA" and self.drag_handle is not None:
            self.drag_handle = None
            self.drag_start_norm = None
            self.original_bbox_norm = None
            self.setCursor(Qt.ArrowCursor)
            self.update()

        elif self.interaction_mode in ("DRAW_ARENA_RECT", "DRAW_ARENA_ELLIPSE"):
            if self.draw_p1_norm is not None and self.grid_arena is not None:
                x1, y1 = self.draw_p1_norm
                x2, y2 = norm_x, norm_y
                bx = min(x1, x2)
                by = min(y1, y2)
                bw = max(0.05, abs(x2 - x1))
                bh = max(0.05, abs(y2 - y1))
                self.grid_arena.norm_bbox = [bx, by, bw, bh]
                self.grid_arena.shape_type = "rectangle" if self.interaction_mode == "DRAW_ARENA_RECT" else "ellipse"
                self.grid_arena.is_defined = True
                if self.on_arena_bbox_changed:
                    self.on_arena_bbox_changed()
            self.draw_p1_norm = None
            self.draw_p2_norm = None
            self.interaction_mode = "ADJUST_ARENA"
            self.update()

        elif self.interaction_mode in ("DRAW_CUSTOM_RECT", "DRAW_CUSTOM_CIRCLE"):
            if self.draw_p1_norm is not None:
                shape = "rectangle" if self.interaction_mode == "DRAW_CUSTOM_RECT" else "circle"
                pts = [self.draw_p1_norm, (norm_x, norm_y)]
                if self.on_custom_zone_drawn:
                    self.on_custom_zone_drawn(shape, pts)
            self.draw_p1_norm = None
            self.draw_p2_norm = None
            self.interaction_mode = "ADJUST_ARENA"
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#080c14"))

        # Empty state if no pixmap is loaded yet
        if self.pixmap is None or self.pixmap.isNull():
            painter.setPen(QColor("#64748b"))
            painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
            painter.drawText(
                self.rect(),
                Qt.AlignCenter,
                "Belum Ada Sumber Video Aktif\n\nSilakan pilih di toolbar atas:\n📹 Sambungkan Kamera  |  📂 Buka Video  |  🧪 Mode Simulasi"
            )
            return

        img_w = self.pixmap.width()
        img_h = self.pixmap.height()
        w_w = self.width()
        w_h = self.height()

        scale = min(w_w / img_w, w_h / img_h)
        disp_w = int(img_w * scale)
        disp_h = int(img_h * scale)
        offset_x = (w_w - disp_w) // 2
        offset_y = (w_h - disp_h) // 2

        scaled_pixmap = self.pixmap.scaled(
            disp_w, disp_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        painter.drawPixmap(offset_x, offset_y, scaled_pixmap)

        # 1. Draw 8 Resize Handles if in ADJUST_ARENA mode and arena is defined
        if self.interaction_mode == "ADJUST_ARENA" and self.grid_arena is not None and getattr(self.grid_arena, "is_defined", True):
            bx_n, by_n, bw_n, bh_n = self.grid_arena.norm_bbox
            bx = offset_x + int(bx_n * disp_w)
            by = offset_y + int(by_n * disp_h)
            bw = int(bw_n * disp_w)
            bh = int(bh_n * disp_h)

            # Outer guideline
            pen_guide = QPen(QColor("#38bdf8"), 2, Qt.DashLine)
            painter.setPen(pen_guide)
            painter.setBrush(Qt.NoBrush)
            if getattr(self.grid_arena, "shape_type", "rectangle") == "ellipse":
                painter.drawEllipse(bx, by, bw, bh)
                pen_bbox = QPen(QColor("#334155"), 1, Qt.DotLine)
                painter.setPen(pen_bbox)
                painter.drawRect(bx, by, bw, bh)
            else:
                painter.drawRect(bx, by, bw, bh)

            # 8 Handles coordinates
            handles = [
                (bx, by), (bx + bw // 2, by), (bx + bw, by),
                (bx, by + bh // 2), (bx + bw, by + bh // 2),
                (bx, by + bh), (bx + bw // 2, by + bh), (bx + bw, by + bh)
            ]
            handle_radius = 6
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.setBrush(QColor("#0284c7"))
            for hx, hy in handles:
                painter.drawEllipse(hx - handle_radius, hy - handle_radius, handle_radius * 2, handle_radius * 2)

        # 2. Draw live rubber-band shapes while drawing
        if self.draw_p1_norm is not None and self.draw_p2_norm is not None:
            p1x = offset_x + int(self.draw_p1_norm[0] * disp_w)
            p1y = offset_y + int(self.draw_p1_norm[1] * disp_h)
            p2x = offset_x + int(self.draw_p2_norm[0] * disp_w)
            p2y = offset_y + int(self.draw_p2_norm[1] * disp_h)

            rx = min(p1x, p2x)
            ry = min(p1y, p2y)
            rw = abs(p2x - p1x)
            rh = abs(p2y - p1y)

            pen_draw = QPen(QColor("#38bdf8"), 2, Qt.DashLine)
            painter.setPen(pen_draw)
            painter.setBrush(QColor(56, 189, 248, 50))

            if self.interaction_mode in ("DRAW_ARENA_RECT", "DRAW_CUSTOM_RECT"):
                painter.drawRect(rx, ry, rw, rh)
            elif self.interaction_mode == "DRAW_ARENA_ELLIPSE":
                painter.drawEllipse(rx, ry, rw, rh)
            elif self.interaction_mode == "DRAW_CUSTOM_CIRCLE":
                rad = int(math.hypot(p2x - p1x, p2y - p1y))
                painter.drawEllipse(p1x - rad, p1y - rad, rad * 2, rad * 2)

        # 3. Prompt banner if arena is not yet drawn
        if self.grid_arena is not None and not getattr(self.grid_arena, "is_defined", True) and self.interaction_mode not in ("DRAW_ARENA_RECT", "DRAW_ARENA_ELLIPSE"):
            guide_w = min(470, disp_w - 40)
            guide_h = 42
            gx = offset_x + (disp_w - guide_w) // 2
            gy = offset_y + 16
            painter.setPen(QPen(QColor("#38bdf8"), 1))
            painter.setBrush(QColor(15, 23, 42, 220))
            painter.drawRoundedRect(gx, gy, guide_w, guide_h, 8, 8)
            painter.setPen(QColor("#f8fafc"))
            painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
            painter.drawText(QRectF(gx, gy, guide_w, guide_h), Qt.AlignCenter, "💡 Silakan gambar batas arena (Klik '🔲 Gambar Kotak' atau '⭕ Gambar Elips')")

        # 4. Draw polygon vertices while clicking
        if self.interaction_mode == "DRAW_CUSTOM_POLY" and self.poly_points_norm:
            pen_poly = QPen(QColor("#ec4899"), 2, Qt.SolidLine)
            painter.setPen(pen_poly)
            painter.setBrush(QColor("#ec4899"))
            disp_pts = []
            for nx, ny in self.poly_points_norm:
                dx = offset_x + int(nx * disp_w)
                dy = offset_y + int(ny * disp_h)
                disp_pts.append(QPoint(dx, dy))
                painter.drawEllipse(dx - 4, dy - 4, 8, 8)
            for i in range(1, len(disp_pts)):
                painter.drawLine(disp_pts[i - 1], disp_pts[i])

        # 5. Draw calibration ruler
        if self.calib_p1_px is not None and self.calib_p2_px is not None:
            p1_disp = QPoint(offset_x + int(self.calib_p1_px[0] * scale), offset_y + int(self.calib_p1_px[1] * scale))
            p2_disp = QPoint(offset_x + int(self.calib_p2_px[0] * scale), offset_y + int(self.calib_p2_px[1] * scale))
            pen = QPen(QColor("#f43f5e"), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawLine(p1_disp, p2_disp)
            painter.setBrush(QColor("#f43f5e"))
            painter.drawEllipse(p1_disp, 4, 4)
            painter.drawEllipse(p2_disp, 4, 4)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MazeTrack Desktop v2.0 - Scientific Animal Behavior Tracking")
        self.resize(1380, 860)

        # Core tracking & arena instances
        self.tracker = AnimalTracker()
        self.grid_arena = TrackingArena(shape_type="rectangle", grid_type="square", real_length_cm=50.0)
        self.grid_arena.is_defined = False  # NO predefined zone at startup!
        self.mwm_arena = MorrisWaterMazeArena()
        self.epm_arena = ElevatedPlusMazeArena()
        self.custom_zones_mgr = CustomZoneManager()
        self.active_arena_type = "Tracking Arena"

        # Video source state
        self.video_cap: Optional[cv2.VideoCapture] = None
        self.cam_stream: Optional[CameraStream] = None
        self.simulation: Optional[RodentSimulation] = None
        self.source_mode: str = "NONE"  # "FILE", "CAMERA", "SIMULATION"
        self.video_fps: float = 30.0
        self.total_frames: int = 0
        self.current_frame_idx: int = 0
        self.playback_speed: float = 1.0

        # Experiment State: "NO_SOURCE", "SETUP", "TRACKING", "PAUSED", "STOPPED"
        self.experiment_state = "NO_SOURCE"

        # Last base frame for exporting images
        self.last_base_frame: Optional[np.ndarray] = None

        # Playback timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_timer_tick)

        # Build user interface
        self.init_ui()
        self.setStyleSheet(DARK_STYLESHEET)

        # Initial state: Standby (clean canvas, waiting for video source)
        self.set_experiment_state("NO_SOURCE")

    def init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # -------------------------------------------------------------
        # 1. LEFT DOCK: Tabs for Arena, Custom Zones, and CV Detection
        # -------------------------------------------------------------
        left_dock = QWidget()
        left_dock.setFixedWidth(330)
        left_layout = QVBoxLayout(left_dock)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        self.tab_left = QTabWidget()

        # === TAB 1: ARENA & GRID SETUP ===
        tab_arena = QWidget()
        t_arena_layout = QVBoxLayout(tab_arena)
        t_arena_layout.setSpacing(8)

        # Apparatus Type
        apparatus_box = QGroupBox("1. Pilihan Aparatus / Arena")
        app_layout = QVBoxLayout(apparatus_box)
        self.combo_arena = QComboBox()
        self.combo_arena.addItems([
            "Arena Kustom (Kotak / Elips)",
            "Morris Water Maze (Preset Kolam)",
            "Elevated Plus Maze (Preset Plus)"
        ])
        self.combo_arena.currentTextChanged.connect(self.on_arena_type_changed)
        app_layout.addWidget(self.combo_arena)
        t_arena_layout.addWidget(apparatus_box)

        # Group 2: Penggambaran Batas Arena Manual
        self.grp_draw_arena = QGroupBox("2. Gambar Batas Arena (Manual)")
        draw_layout = QVBoxLayout(self.grp_draw_arena)

        btn_draw_row = QHBoxLayout()
        self.btn_draw_rect = QPushButton("🔲 Gambar Kotak")
        self.btn_draw_rect.setObjectName("primaryBtn")
        self.btn_draw_rect.clicked.connect(lambda: self.activate_draw_arena_mode("rectangle"))
        btn_draw_row.addWidget(self.btn_draw_rect)

        self.btn_draw_ellipse = QPushButton("⭕ Gambar Elips")
        self.btn_draw_ellipse.setObjectName("primaryBtn")
        self.btn_draw_ellipse.clicked.connect(lambda: self.activate_draw_arena_mode("ellipse"))
        btn_draw_row.addWidget(self.btn_draw_ellipse)
        draw_layout.addLayout(btn_draw_row)

        adj_row = QHBoxLayout()
        self.btn_mode_adjust = QPushButton("📐 Geser / Tarik Sudut")
        self.btn_mode_adjust.clicked.connect(self.activate_adjust_arena_mode)
        adj_row.addWidget(self.btn_mode_adjust)

        self.btn_reset_arena = QPushButton("🔄 Reset Zona")
        self.btn_reset_arena.clicked.connect(self.reset_arena_zone)
        adj_row.addWidget(self.btn_reset_arena)
        draw_layout.addLayout(adj_row)

        self.lbl_draw_hint = QLabel("💡 Klik 'Gambar Kotak' atau 'Gambar Elips', lalu tarik mouse di atas video.")
        self.lbl_draw_hint.setWordWrap(True)
        self.lbl_draw_hint.setStyleSheet("color: #94a3b8; font-size: 11px;")
        draw_layout.addWidget(self.lbl_draw_hint)
        t_arena_layout.addWidget(self.grp_draw_arena)

        # Group 3: Skala Panjang Arena (cm)
        self.grp_scale = QGroupBox("3. Skala Panjang Arena")
        scale_layout = QVBoxLayout(self.grp_scale)

        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Panjang/Diameter Riil:"))
        self.spin_real_cm = QDoubleSpinBox()
        self.spin_real_cm.setRange(1.0, 1000.0)
        self.spin_real_cm.setValue(50.0)
        self.spin_real_cm.setSingleStep(5.0)
        self.spin_real_cm.setSuffix(" cm")
        self.spin_real_cm.valueChanged.connect(self.on_real_length_changed)
        scale_row.addWidget(self.spin_real_cm)
        scale_layout.addLayout(scale_row)

        self.lbl_scale_calc = QLabel("Skala: 5.00 px/cm")
        self.lbl_scale_calc.setStyleSheet("color: #38bdf8; font-weight: bold;")
        scale_layout.addWidget(self.lbl_scale_calc)

        self.btn_calibrate = QPushButton("📏 Kalibrasi Manual (Ruler 2 Titik)")
        self.btn_calibrate.clicked.connect(self.start_scale_calibration)
        scale_layout.addWidget(self.btn_calibrate)
        t_arena_layout.addWidget(self.grp_scale)

        # Group 4: Pembagian Grid Multimode
        self.grp_grid_ctrl = QGroupBox("4. Pembagian Grid Arena")
        grid_ctrl_layout = QVBoxLayout(self.grp_grid_ctrl)

        grid_ctrl_layout.addWidget(QLabel("Pilih Tipe Pembagian Grid:"))
        self.combo_grid_type = QComboBox()
        self.combo_grid_type.addItems([
            "⬛ Square Grid (Kotak N x M)",
            "🎯 Lingkaran Berbasis Diameter (Konsentris)",
            "🧭 Grid Radial Berbasis Sudut (Sektor)",
            "🌐 Polar Grid (Center + Sektor Luar)",
            "⏹️ Tanpa Grid (Satu Arena Penuh)"
        ])
        self.combo_grid_type.currentTextChanged.connect(self.on_grid_type_selection_changed)
        grid_ctrl_layout.addWidget(self.combo_grid_type)

        # Subpanel A: Square Grid Controls
        self.panel_square = QWidget()
        sq_layout = QVBoxLayout(self.panel_square)
        sq_layout.setContentsMargins(0, 4, 0, 0)
        dim_row = QHBoxLayout()
        dim_row.addWidget(QLabel("Baris:"))
        self.spin_rows = QSpinBox()
        self.spin_rows.setRange(2, 10)
        self.spin_rows.setValue(4)
        self.spin_rows.valueChanged.connect(self.on_grid_dim_changed)
        dim_row.addWidget(self.spin_rows)

        dim_row.addWidget(QLabel("Kolom:"))
        self.spin_cols = QSpinBox()
        self.spin_cols.setRange(2, 10)
        self.spin_cols.setValue(4)
        self.spin_cols.valueChanged.connect(self.on_grid_dim_changed)
        dim_row.addWidget(self.spin_cols)
        sq_layout.addLayout(dim_row)

        self.btn_auto_label = QPushButton("⚡ Auto-Label (Corner / Periphery / Center)")
        self.btn_auto_label.clicked.connect(self.on_auto_label_clicked)
        sq_layout.addWidget(self.btn_auto_label)

        # Paint Brush Selector
        sq_layout.addWidget(QLabel("Tandai Petak (Klik Petak di Video):"))
        paint_btn_group = QButtonGroup(self)
        self.rb_corner = QRadioButton("Corner")
        self.rb_periphery = QRadioButton("Periphery")
        self.rb_center = QRadioButton("Center")
        self.rb_custom_paint = QRadioButton("Custom")

        self.rb_corner.setStyleSheet("color: #f43f5e;")
        self.rb_periphery.setStyleSheet("color: #f59e0b;")
        self.rb_center.setStyleSheet("color: #10b981;")
        self.rb_custom_paint.setStyleSheet("color: #d946ef;")

        self.rb_corner.setChecked(True)
        paint_btn_group.addButton(self.rb_corner)
        paint_btn_group.addButton(self.rb_periphery)
        paint_btn_group.addButton(self.rb_center)
        paint_btn_group.addButton(self.rb_custom_paint)

        self.rb_corner.toggled.connect(lambda: self.on_paint_tool_changed("Corner"))
        self.rb_periphery.toggled.connect(lambda: self.on_paint_tool_changed("Periphery"))
        self.rb_center.toggled.connect(lambda: self.on_paint_tool_changed("Center"))
        self.rb_custom_paint.toggled.connect(lambda: self.on_paint_tool_changed("Custom"))

        p_row1 = QHBoxLayout()
        p_row1.addWidget(self.rb_corner)
        p_row1.addWidget(self.rb_periphery)
        p_row2 = QHBoxLayout()
        p_row2.addWidget(self.rb_center)
        p_row2.addWidget(self.rb_custom_paint)
        sq_layout.addLayout(p_row1)
        sq_layout.addLayout(p_row2)

        self.btn_paint_mode = QPushButton("🖌️ Aktifkan Kuas Petak")
        self.btn_paint_mode.clicked.connect(self.activate_paint_cell_mode)
        sq_layout.addWidget(self.btn_paint_mode)

        self.lbl_square_cell_info = QLabel("Ukuran petak: 12.5 x 12.5 cm")
        self.lbl_square_cell_info.setStyleSheet("color: #94a3b8; font-size: 11px;")
        sq_layout.addWidget(self.lbl_square_cell_info)
        grid_ctrl_layout.addWidget(self.panel_square)

        # Subpanel B: Concentric Rings Controls
        self.panel_concentric = QWidget()
        conc_layout = QVBoxLayout(self.panel_concentric)
        conc_layout.setContentsMargins(0, 4, 0, 0)
        conc_row = QHBoxLayout()
        conc_row.addWidget(QLabel("Jumlah Cincin:"))
        self.combo_conc_rings = QComboBox()
        self.combo_conc_rings.addItems(["2 Cincin (Center, Periphery)", "3 Cincin (Center, Middle, Periphery)", "4 Cincin (Center, Inner, Mid, Outer)"])
        self.combo_conc_rings.setCurrentIndex(1)
        self.combo_conc_rings.currentIndexChanged.connect(self.on_concentric_rings_changed)
        conc_row.addWidget(self.combo_conc_rings)
        conc_layout.addLayout(conc_row)

        self.lbl_concentric_info = QLabel("• Center: 0 - 17.5 cm\n• Middle: 17.5 - 35.0 cm\n• Periphery: 35.0 - 50.0 cm")
        self.lbl_concentric_info.setStyleSheet("color: #38bdf8; font-size: 11px;")
        conc_layout.addWidget(self.lbl_concentric_info)
        grid_ctrl_layout.addWidget(self.panel_concentric)
        self.panel_concentric.hide()

        # Subpanel C: Radial Sectors Controls
        self.panel_radial = QWidget()
        rad_layout = QVBoxLayout(self.panel_radial)
        rad_layout.setContentsMargins(0, 4, 0, 0)
        rad_row = QHBoxLayout()
        rad_row.addWidget(QLabel("Pembagian Sudut:"))
        self.combo_radial_sectors = QComboBox()
        self.combo_radial_sectors.addItems(["4 Kuadran (90° per kuadran)", "6 Sektor (60° per sektor)", "8 Sektor / Oktan (45° per sektor)", "12 Sektor (30° per sektor)"])
        self.combo_radial_sectors.currentIndexChanged.connect(self.on_radial_sectors_changed)
        rad_row.addWidget(self.combo_radial_sectors)
        rad_layout.addLayout(rad_row)

        rot_row = QHBoxLayout()
        rot_row.addWidget(QLabel("Rotasi Sudut:"))
        self.spin_radial_rot = QDoubleSpinBox()
        self.spin_radial_rot.setRange(0.0, 360.0)
        self.spin_radial_rot.setValue(0.0)
        self.spin_radial_rot.setSingleStep(15.0)
        self.spin_radial_rot.setSuffix("°")
        self.spin_radial_rot.valueChanged.connect(self.on_radial_rotation_changed)
        rot_row.addWidget(self.spin_radial_rot)
        rad_layout.addLayout(rot_row)
        grid_ctrl_layout.addWidget(self.panel_radial)
        self.panel_radial.hide()

        # Subpanel D: Polar Grid Controls
        self.panel_polar = QWidget()
        pol_layout = QVBoxLayout(self.panel_polar)
        pol_layout.setContentsMargins(0, 4, 0, 0)
        pol_row = QHBoxLayout()
        pol_row.addWidget(QLabel("Sektor Luar:"))
        self.combo_polar_sectors = QComboBox()
        self.combo_polar_sectors.addItems(["4 Sektor Kuadran", "8 Sektor Oktan"])
        self.combo_polar_sectors.currentIndexChanged.connect(self.on_polar_sectors_changed)
        pol_row.addWidget(self.combo_polar_sectors)
        pol_layout.addLayout(pol_row)

        center_frac_row = QHBoxLayout()
        center_frac_row.addWidget(QLabel("Diameter Center:"))
        self.combo_polar_center = QComboBox()
        self.combo_polar_center.addItems(["25% Diameter", "35% Diameter", "50% Diameter"])
        self.combo_polar_center.setCurrentIndex(1)
        self.combo_polar_center.currentIndexChanged.connect(self.on_polar_center_changed)
        center_frac_row.addWidget(self.combo_polar_center)
        pol_layout.addLayout(center_frac_row)
        grid_ctrl_layout.addWidget(self.panel_polar)
        self.panel_polar.hide()

        t_arena_layout.addWidget(self.grp_grid_ctrl)
        t_arena_layout.addStretch()
        self.tab_left.addTab(tab_arena, "Arena & Grid")

        # === TAB 2: CUSTOM FREE ZONES ===
        tab_cz = QWidget()
        t_cz_layout = QVBoxLayout(tab_cz)
        t_cz_layout.setSpacing(8)

        t_cz_layout.addWidget(QLabel("Tambah Zona Bebas (Objek, Shelter, dll):"))
        cz_btn_row = QHBoxLayout()
        self.btn_add_rect = QPushButton("➕ Kotak")
        self.btn_add_rect.clicked.connect(lambda: self.start_draw_custom_zone("rectangle"))
        cz_btn_row.addWidget(self.btn_add_rect)

        self.btn_add_circle = QPushButton("➕ Lingkaran")
        self.btn_add_circle.clicked.connect(lambda: self.start_draw_custom_zone("circle"))
        cz_btn_row.addWidget(self.btn_add_circle)

        self.btn_add_poly = QPushButton("➕ Poligon")
        self.btn_add_poly.clicked.connect(lambda: self.start_draw_custom_zone("polygon"))
        cz_btn_row.addWidget(self.btn_add_poly)
        t_cz_layout.addLayout(cz_btn_row)

        cz_hint = QLabel("• Kotak/Lingkaran: Klik & tarik mouse di atas video.\n• Poligon: Klik titik-titik sudut, klik kanan untuk selesai.")
        cz_hint.setWordWrap(True)
        cz_hint.setStyleSheet("color: #94a3b8; font-size: 11px;")
        t_cz_layout.addWidget(cz_hint)

        t_cz_layout.addWidget(QLabel("Daftar Zona Bebas Aktif:"))
        self.table_cz = QTableWidget()
        self.table_cz.setColumnCount(3)
        self.table_cz.setHorizontalHeaderLabels(["Nama Zona", "Bentuk", "Aksi"])
        self.table_cz.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table_cz.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table_cz.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table_cz.verticalHeader().setVisible(False)
        t_cz_layout.addWidget(self.table_cz)

        self.btn_clear_cz = QPushButton("Hapus Semua Zona Bebas")
        self.btn_clear_cz.clicked.connect(self.clear_all_custom_zones)
        t_cz_layout.addWidget(self.btn_clear_cz)

        self.tab_left.addTab(tab_cz, "Custom Zones")

        # === TAB 3: TRACKING & CV CONFIG ===
        tab_cv = QWidget()
        t_cv_layout = QVBoxLayout(tab_cv)
        t_cv_layout.setSpacing(8)

        # Animal color
        t_cv_layout.addWidget(QLabel("Warna Hewan Uji:"))
        col_group = QButtonGroup(self)
        self.rb_dark_animal = QRadioButton("Dark Animal (Hitam/Gelap)")
        self.rb_light_animal = QRadioButton("Light Animal (Albino/Putih)")
        self.rb_dark_animal.setChecked(True)
        col_group.addButton(self.rb_dark_animal)
        col_group.addButton(self.rb_light_animal)
        self.rb_dark_animal.toggled.connect(self.on_animal_color_changed)
        t_cv_layout.addWidget(self.rb_dark_animal)
        t_cv_layout.addWidget(self.rb_light_animal)

        # Threshold slider
        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Contrast Threshold:"))
        self.lbl_thresh_val = QLabel("100")
        self.lbl_thresh_val.setStyleSheet("color: #38bdf8; font-weight: bold;")
        thresh_row.addWidget(self.lbl_thresh_val)
        t_cv_layout.addLayout(thresh_row)

        self.slider_thresh = QSlider(Qt.Horizontal)
        self.slider_thresh.setRange(0, 255)
        self.slider_thresh.setValue(100)
        self.slider_thresh.valueChanged.connect(self.on_thresh_slider_changed)
        t_cv_layout.addWidget(self.slider_thresh)

        # Min Blob Size
        blob_row = QHBoxLayout()
        blob_row.addWidget(QLabel("Min Blob Area:"))
        self.spin_min_area = QSpinBox()
        self.spin_min_area.setRange(10, 5000)
        self.spin_min_area.setValue(80)
        self.spin_min_area.valueChanged.connect(lambda v: setattr(self.tracker, "min_area", v))
        blob_row.addWidget(self.spin_min_area)
        t_cv_layout.addLayout(blob_row)

        # Freezing Speed Threshold
        freeze_row = QHBoxLayout()
        freeze_row.addWidget(QLabel("Freezing Thresh:"))
        self.spin_freeze_thresh = QDoubleSpinBox()
        self.spin_freeze_thresh.setRange(0.1, 10.0)
        self.spin_freeze_thresh.setSingleStep(0.2)
        self.spin_freeze_thresh.setValue(1.0)
        self.spin_freeze_thresh.setSuffix(" cm/s")
        self.spin_freeze_thresh.valueChanged.connect(lambda v: setattr(self.tracker, "freezing_speed_thresh", v))
        freeze_row.addWidget(self.spin_freeze_thresh)
        t_cv_layout.addLayout(freeze_row)

        # Live Mask Preview Thumbnail
        t_cv_layout.addWidget(QLabel("Live Mask Preview:"))
        self.lbl_mask_preview = QLabel()
        self.lbl_mask_preview.setFixedHeight(85)
        self.lbl_mask_preview.setStyleSheet("background-color: #000000; border: 1px solid #334155; border-radius: 4px;")
        self.lbl_mask_preview.setAlignment(Qt.AlignCenter)
        t_cv_layout.addWidget(self.lbl_mask_preview)

        # Display Toggles
        grp_overlay_view = QGroupBox("Tampilan Visual")
        ov_view_layout = QVBoxLayout(grp_overlay_view)
        self.chk_show_traj = QCheckBox("Lintasan (Trajectory)")
        self.chk_show_traj.setChecked(True)
        ov_view_layout.addWidget(self.chk_show_traj)

        self.chk_show_heat = QCheckBox("Heatmap Kepadatan")
        self.chk_show_heat.setChecked(False)
        ov_view_layout.addWidget(self.chk_show_heat)

        self.chk_show_grid = QCheckBox("Grid & Zona Arena")
        self.chk_show_grid.setChecked(True)
        ov_view_layout.addWidget(self.chk_show_grid)

        t_cv_layout.addWidget(grp_overlay_view)
        t_cv_layout.addStretch()

        self.tab_left.addTab(tab_cv, "Deteksi CV")

        left_layout.addWidget(self.tab_left)

        # -------------------------------------------------------------
        # 2. CENTER: Video Canvas, Top Control Toolbar & Bottom Bar
        # -------------------------------------------------------------
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(8)

        # Top Action Toolbar (Video Source & Status Banner)
        top_bar = QHBoxLayout()

        self.btn_open_file = QPushButton("📂 Buka Video")
        self.btn_open_file.clicked.connect(self.open_video_file)
        top_bar.addWidget(self.btn_open_file)

        self.btn_webcam = QPushButton("📹 Sambungkan Kamera")
        self.btn_webcam.clicked.connect(self.open_camera_dialog)
        top_bar.addWidget(self.btn_webcam)

        self.btn_sim = QPushButton("🧪 Mode Simulasi")
        self.btn_sim.clicked.connect(self.load_simulation_source)
        top_bar.addWidget(self.btn_sim)

        top_bar.addStretch()

        # Status Banner Badge
        self.lbl_status_badge = QLabel("[ 🛠️ MODE SETUP / PERSIAPAN ]")
        self.lbl_status_badge.setStyleSheet(
            "background-color: #1e293b; color: #38bdf8; font-weight: bold; padding: 6px 12px; border-radius: 4px; border: 1px solid #38bdf8;"
        )
        top_bar.addWidget(self.lbl_status_badge)

        center_layout.addLayout(top_bar)

        # Video Canvas
        self.canvas = VideoCanvas(self)
        self.canvas.grid_arena = self.grid_arena
        self.canvas.custom_zones_mgr = self.custom_zones_mgr
        self.canvas.on_cell_clicked = self.handle_cell_click
        self.canvas.on_calib_completed = self.handle_calibration_done
        self.canvas.on_custom_zone_drawn = self.handle_custom_zone_drawn
        self.canvas.on_arena_bbox_changed = self.on_arena_bbox_updated
        center_layout.addWidget(self.canvas, stretch=1)

        # Main Experiment Control Bar (PROMINENT USER CONTROLS)
        exp_ctrl_bar = QHBoxLayout()

        self.btn_start_tracking = QPushButton("▶ Mulai Tracking (Start Trial)")
        self.btn_start_tracking.setObjectName("startBtn")
        self.btn_start_tracking.clicked.connect(self.start_tracking)
        exp_ctrl_bar.addWidget(self.btn_start_tracking)

        self.btn_pause_tracking = QPushButton("⏸ Jeda (Pause)")
        self.btn_pause_tracking.setObjectName("pauseBtn")
        self.btn_pause_tracking.setEnabled(False)
        self.btn_pause_tracking.clicked.connect(self.pause_tracking)
        exp_ctrl_bar.addWidget(self.btn_pause_tracking)

        self.btn_stop_tracking = QPushButton("⏹ Selesai / Hentikan Tracking")
        self.btn_stop_tracking.setObjectName("stopBtn")
        self.btn_stop_tracking.setEnabled(False)
        self.btn_stop_tracking.clicked.connect(self.stop_tracking)
        exp_ctrl_bar.addWidget(self.btn_stop_tracking)

        self.btn_reset_trial = QPushButton("🔄 Reset Percobaan")
        self.btn_reset_trial.clicked.connect(self.reset_trial)
        exp_ctrl_bar.addWidget(self.btn_reset_trial)

        exp_ctrl_bar.addStretch()

        # Step controls & timeline seek
        self.btn_step_back = QPushButton("⏮")
        self.btn_step_back.setFixedWidth(36)
        self.btn_step_back.clicked.connect(self.step_backward)
        exp_ctrl_bar.addWidget(self.btn_step_back)

        self.btn_step_fwd = QPushButton("⏭")
        self.btn_step_fwd.setFixedWidth(36)
        self.btn_step_fwd.clicked.connect(self.step_forward)
        exp_ctrl_bar.addWidget(self.btn_step_fwd)

        self.slider_seek = QSlider(Qt.Horizontal)
        self.slider_seek.setRange(0, 100)
        self.slider_seek.sliderMoved.connect(self.on_seek_moved)
        exp_ctrl_bar.addWidget(self.slider_seek)

        self.lbl_time = QLabel("00:00.0 / 00:00.0 (F: 0)")
        self.lbl_time.setFixedWidth(160)
        self.lbl_time.setAlignment(Qt.AlignCenter)
        self.lbl_time.setStyleSheet("color: #94a3b8; font-family: monospace;")
        exp_ctrl_bar.addWidget(self.lbl_time)

        self.combo_speed = QComboBox()
        self.combo_speed.addItems(["0.5x", "1.0x", "2.0x", "4.0x"])
        self.combo_speed.setCurrentText("1.0x")
        self.combo_speed.currentTextChanged.connect(self.on_speed_changed)
        exp_ctrl_bar.addWidget(self.combo_speed)

        center_layout.addLayout(exp_ctrl_bar)

        # -------------------------------------------------------------
        # 3. RIGHT DOCK: Live Scientific Analytics & Data Exports
        # -------------------------------------------------------------
        right_dock = QWidget()
        right_dock.setFixedWidth(350)
        right_layout = QVBoxLayout(right_dock)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        # Group D: Live HUD Metrics
        grp_hud = QGroupBox("Metrik Perilaku Real-Time")
        hud_layout = QGridLayout(grp_hud)
        hud_layout.setVerticalSpacing(8)

        hud_layout.addWidget(QLabel("Total Jarak:"), 0, 0)
        self.lbl_hud_dist = QLabel("0.0 cm")
        self.lbl_hud_dist.setStyleSheet("color: #38bdf8; font-size: 16px; font-weight: bold;")
        hud_layout.addWidget(self.lbl_hud_dist, 0, 1)

        hud_layout.addWidget(QLabel("Kecepatan:"), 1, 0)
        self.lbl_hud_speed = QLabel("0.0 cm/s")
        self.lbl_hud_speed.setStyleSheet("color: #10b981; font-size: 14px; font-weight: bold;")
        hud_layout.addWidget(self.lbl_hud_speed, 1, 1)

        hud_layout.addWidget(QLabel("Zona / Petak:"), 2, 0)
        self.lbl_hud_zone = QLabel("-")
        self.lbl_hud_zone.setStyleSheet("color: #f59e0b; font-size: 14px; font-weight: bold;")
        hud_layout.addWidget(self.lbl_hud_zone, 2, 1)

        hud_layout.addWidget(QLabel("Custom Zone:"), 3, 0)
        self.lbl_hud_custom = QLabel("-")
        self.lbl_hud_custom.setStyleSheet("color: #ec4899; font-size: 13px; font-weight: bold;")
        hud_layout.addWidget(self.lbl_hud_custom, 3, 1)

        hud_layout.addWidget(QLabel("Status Hewan:"), 4, 0)
        self.lbl_hud_state = QLabel("STANDBY")
        self.lbl_hud_state.setStyleSheet("color: #94a3b8; font-weight: bold;")
        hud_layout.addWidget(self.lbl_hud_state, 4, 1)

        hud_layout.addWidget(QLabel("Waktu Freezing:"), 5, 0)
        self.lbl_hud_freeze = QLabel("0.0 s (0.0%)")
        self.lbl_hud_freeze.setStyleSheet("color: #f43f5e; font-weight: bold;")
        hud_layout.addWidget(self.lbl_hud_freeze, 5, 1)

        hud_layout.addWidget(QLabel("Skala:"), 6, 0)
        self.lbl_hud_scale = QLabel(f"{self.tracker.pixels_per_cm:.1f} px/cm")
        self.lbl_hud_scale.setStyleSheet("color: #94a3b8;")
        hud_layout.addWidget(self.lbl_hud_scale, 6, 1)

        right_layout.addWidget(grp_hud)

        # Group E: Zone Statistics Table
        grp_table = QGroupBox("Tabel Statistik Zona / Petak")
        table_layout = QVBoxLayout(grp_table)

        self.table_zones = QTableWidget()
        self.table_zones.setColumnCount(4)
        self.table_zones.setHorizontalHeaderLabels(["Zona", "Waktu (s)", "% Waktu", "Masuk"])
        self.table_zones.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_zones.verticalHeader().setVisible(False)
        self.table_zones.setAlternatingRowColors(True)
        table_layout.addWidget(self.table_zones)

        right_layout.addWidget(grp_table, stretch=1)

        # Group F: Data Export
        grp_export = QGroupBox("Ekspor Data Penelitian")
        exp_layout = QVBoxLayout(grp_export)

        self.btn_exp_summary = QPushButton("📊 Ekspor Ringkasan (CSV)")
        self.btn_exp_summary.setObjectName("exportBtn")
        self.btn_exp_summary.clicked.connect(self.export_summary_csv)
        exp_layout.addWidget(self.btn_exp_summary)

        self.btn_exp_raw = QPushButton("📝 Ekspor Koordinat Mentah (CSV)")
        self.btn_exp_raw.setObjectName("exportBtn")
        self.btn_exp_raw.clicked.connect(self.export_raw_csv)
        exp_layout.addWidget(self.btn_exp_raw)

        exp_img_row = QHBoxLayout()
        self.btn_exp_traj_img = QPushButton("🗺️ Simpan Lintasan (PNG)")
        self.btn_exp_traj_img.clicked.connect(self.export_trajectory_png)
        exp_img_row.addWidget(self.btn_exp_traj_img)

        self.btn_exp_heat_img = QPushButton("🔥 Simpan Heatmap (PNG)")
        self.btn_exp_heat_img.clicked.connect(self.export_heatmap_png)
        exp_img_row.addWidget(self.btn_exp_heat_img)
        exp_layout.addLayout(exp_img_row)

        right_layout.addWidget(grp_export)

        # Assemble layout
        main_layout.addWidget(left_dock)
        main_layout.addWidget(center_widget, stretch=1)
        main_layout.addWidget(right_dock)

    # -------------------------------------------------------------
    # User-Controlled Tracking Workflow & State Machine
    # -------------------------------------------------------------
    def set_experiment_state(self, new_state: str):
        self.experiment_state = new_state

        if new_state == "NO_SOURCE":
            self.lbl_status_badge.setText("[ 🟡 LANGKAH 1: PILIH SUMBER VIDEO ]")
            self.lbl_status_badge.setStyleSheet(
                "background-color: #1e293b; color: #fbbf24; font-weight: bold; padding: 6px 12px; border-radius: 4px; border: 1px solid #f59e0b;"
            )
            self.btn_start_tracking.setEnabled(False)
            self.btn_pause_tracking.setEnabled(False)
            self.btn_stop_tracking.setEnabled(False)
            self.canvas.interaction_mode = "NONE"

        elif new_state == "SETUP":
            if not getattr(self.grid_arena, "is_defined", False):
                self.lbl_status_badge.setText("[ ✏️ LANGKAH 2: GAMBAR ZONA TRACKING (KOTAK / ELIPS) ]")
                self.lbl_status_badge.setStyleSheet(
                    "background-color: #1e293b; color: #38bdf8; font-weight: bold; padding: 6px 12px; border-radius: 4px; border: 1px solid #38bdf8;"
                )
            else:
                self.lbl_status_badge.setText("[ 🟢 SIAP TRACKING ]")
                self.lbl_status_badge.setStyleSheet(
                    "background-color: #064e3b; color: #34d399; font-weight: bold; padding: 6px 12px; border-radius: 4px; border: 1px solid #10b981;"
                )
            self.btn_start_tracking.setEnabled(True)
            self.btn_start_tracking.setText("▶ Mulai Tracking (Start Trial)")
            self.btn_pause_tracking.setEnabled(False)
            self.btn_stop_tracking.setEnabled(False)
            self.canvas.interaction_mode = "ADJUST_ARENA"

        elif new_state == "TRACKING":
            self.lbl_status_badge.setText("[ 🔴 SEDANG TRACKING AKTIF ]")
            self.lbl_status_badge.setStyleSheet(
                "background-color: #064e3b; color: #34d399; font-weight: bold; padding: 6px 12px; border-radius: 4px; border: 1px solid #10b981;"
            )
            self.btn_start_tracking.setEnabled(False)
            self.btn_pause_tracking.setEnabled(True)
            self.btn_stop_tracking.setEnabled(True)
            self.canvas.interaction_mode = "NONE"

        elif new_state == "PAUSED":
            self.lbl_status_badge.setText("[ ⏸️ TRACKING DIJEDA (PAUSED) ]")
            self.lbl_status_badge.setStyleSheet(
                "background-color: #78350f; color: #fbbf24; font-weight: bold; padding: 6px 12px; border-radius: 4px; border: 1px solid #f59e0b;"
            )
            self.btn_start_tracking.setEnabled(True)
            self.btn_start_tracking.setText("▶ Lanjutkan Tracking")
            self.btn_pause_tracking.setEnabled(False)
            self.btn_stop_tracking.setEnabled(True)

        elif new_state == "STOPPED":
            self.lbl_status_badge.setText("[ 🏁 PERCOBAAN SELESAI / STOPPED ]")
            self.lbl_status_badge.setStyleSheet(
                "background-color: #881337; color: #fda4af; font-weight: bold; padding: 6px 12px; border-radius: 4px; border: 1px solid #f43f5e;"
            )
            self.btn_start_tracking.setEnabled(True)
            self.btn_start_tracking.setText("▶ Mulai Ulang Tracking")
            self.btn_pause_tracking.setEnabled(False)
            self.btn_stop_tracking.setEnabled(False)

    def start_tracking(self):
        if self.source_mode == "NONE":
            QMessageBox.warning(
                self, "Pilih Sumber Video",
                "Silakan pilih sumber video terlebih dahulu!\n"
                "Gunakan tombol di toolbar atas: '📹 Sambungkan Kamera', '📂 Buka Video', atau '🧪 Mode Simulasi'."
            )
            return

        if not getattr(self.grid_arena, "is_defined", False) and self.active_arena_type in ("Tracking Arena", "Arena Kustom (Kotak / Elips)"):
            msg = QMessageBox(self)
            msg.setWindowTitle("Zona Tracking Belum Digambar")
            msg.setText(
                "Batas zona tracking belum ditentukan.\n\n"
                "Rekomendasi Ilmiah: Gambar zona tracking (Kotak atau Elips) terlebih dahulu "
                "agar analisis grid, metrik jarak, dan zona thigmotaksis akurat.\n\n"
                "Pilih tindakan Anda:"
            )
            btn_rect = msg.addButton("🔲 Gambar Kotak", QMessageBox.ActionRole)
            btn_ell = msg.addButton("⭕ Gambar Elips", QMessageBox.ActionRole)
            btn_continue = msg.addButton("Lanjut Seluruh Layar", QMessageBox.DestructiveRole)
            btn_cancel = msg.addButton("Batal", QMessageBox.RejectRole)

            msg.exec_()
            clicked = msg.clickedButton()
            if clicked == btn_rect:
                self.activate_draw_arena_mode("rectangle")
                return
            elif clicked == btn_ell:
                self.activate_draw_arena_mode("ellipse")
                return
            elif clicked == btn_cancel:
                return

        if self.experiment_state == "STOPPED":
            self.tracker.reset()
            self.current_frame_idx = 0

        self.set_experiment_state("TRACKING")
        self.update_timer_interval()
        self.timer.start()

    def pause_tracking(self):
        self.timer.stop()
        self.set_experiment_state("PAUSED")

    def stop_tracking(self):
        self.timer.stop()
        self.set_experiment_state("STOPPED")
        QMessageBox.information(
            self, "Percobaan Selesai",
            "Pelacakan selesai! Data telah dibekukan.\n"
            "Anda dapat mengekspor ringkasan atau lintasan menggunakan tombol di panel kanan."
        )

    def reset_trial(self):
        self.timer.stop()
        self.tracker.reset()
        self.current_frame_idx = 0
        self.update_hud(None)
        self.update_zone_table()
        self.set_experiment_state("SETUP" if self.source_mode != "NONE" else "NO_SOURCE")
        # Render clean frame preview
        self.render_preview_frame()

    # -------------------------------------------------------------
    # Video Sources: Simulation, Camera, File
    # -------------------------------------------------------------
    def load_initial_simulation_preview(self):
        self.stop_current_source()
        self.source_mode = "SIMULATION"
        self.simulation = RodentSimulation(width=640, height=480, fps=30)
        self.video_fps = 30.0
        self.total_frames = 100000
        self.current_frame_idx = 0
        self.set_experiment_state("SETUP")
        self.render_preview_frame()

    def load_simulation_source(self):
        self.load_initial_simulation_preview()
        QMessageBox.information(
            self, "Mode Simulasi Aktif",
            "Simulasi aktif dalam Mode Persiapan!\n\n"
            "Langkah selanjutnya:\n"
            "1. Klik '🔲 Gambar Kotak' atau '⭕ Gambar Elips' di panel kiri.\n"
            "2. Tarik mouse di atas video untuk menentukan batas arena tracking.\n"
            "3. Atur skala panjang dan pembagian grid (Square / Lingkaran Diameter / Radial Sudut).\n"
            "4. Tekan tombol '▶ Mulai Tracking' jika sudah siap."
        )

    def open_camera_dialog(self):
        dlg = CameraSelectDialog(self)
        if dlg.exec_() == QDialog.Accepted and dlg.selected_device is not None:
            self.stop_current_source()
            self.cam_stream = CameraStream()
            ok = self.cam_stream.start(dlg.selected_device, dlg.selected_width, dlg.selected_height)
            if not ok:
                QMessageBox.critical(self, "Error Kamera", f"Gagal membuka stream kamera: {dlg.selected_device.name}")
                return

            self.source_mode = "CAMERA"
            self.video_fps = 30.0
            self.total_frames = 100000
            self.current_frame_idx = 0
            self.set_experiment_state("SETUP")

            # Wait briefly for frame and render preview
            QTimer.singleShot(400, self.render_preview_frame)
            QMessageBox.information(
                self, "Kamera Tersambung",
                f"Kamera '{dlg.selected_device.name}' berhasil tersambung!\n\n"
                "Silakan gambar batas arena tracking (Kotak atau Elips) pada panel kiri, lalu tentukan skala & grid."
            )

    def open_video_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Buka File Video Penelitian", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;All Files (*)"
        )
        if not path:
            return

        self.stop_current_source()
        self.video_cap = cv2.VideoCapture(path)
        if not self.video_cap.isOpened():
            QMessageBox.critical(self, "Error", f"Gagal membuka file video:\n{path}")
            return

        self.source_mode = "FILE"
        self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = float(self.video_cap.get(cv2.CAP_PROP_FPS))
        if self.video_fps <= 0 or np.isnan(self.video_fps):
            self.video_fps = 30.0

        self.current_frame_idx = 0
        self.slider_seek.setRange(0, max(1, self.total_frames - 1))
        self.set_experiment_state("SETUP")
        self.render_preview_frame()
        QMessageBox.information(
            self, "Video Terbuka",
            "Video berhasil dimuat!\n\n"
            "Silakan gambar batas arena tracking (Kotak atau Elips) pada panel kiri, lalu tentukan skala & grid."
        )

    def stop_current_source(self):
        self.timer.stop()
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
        if self.cam_stream is not None:
            self.cam_stream.stop()
            self.cam_stream = None
        self.simulation = None
        self.source_mode = "NONE"

    def render_preview_frame(self):
        """Fetches 1 frame from current source and displays it with overlays in Setup Mode."""
        frame_bgr = None
        if self.source_mode == "SIMULATION" and self.simulation is not None:
            frame_bgr = self.simulation.get_next_frame()
        elif self.source_mode == "CAMERA" and self.cam_stream is not None:
            ret, fr = self.cam_stream.read()
            if ret:
                frame_bgr = fr
        elif self.source_mode == "FILE" and self.video_cap is not None:
            ret, fr = self.video_cap.read()
            if ret:
                frame_bgr = fr

        if frame_bgr is not None:
            self.last_base_frame = frame_bgr.copy()
            h, w = frame_bgr.shape[:2]

            disp = frame_bgr.copy()
            # Draw arena overlay
            if self.chk_show_grid.isChecked():
                if self.active_arena_type in ("Tracking Arena", "Arena Kustom (Kotak / Elips)", "Open Field Grid"):
                    if getattr(self.grid_arena, "is_defined", False):
                        disp = self.grid_arena.draw_overlay(disp, alpha=0.25, show_text=True)
                elif self.active_arena_type == "Morris Water Maze":
                    disp = self.mwm_arena.draw_overlay(disp, alpha=0.25, show_text=True)
                elif self.active_arena_type == "Elevated Plus Maze":
                    disp = self.epm_arena.draw_overlay(disp, alpha=0.25, show_text=True)

            # Draw custom free zones
            disp = self.custom_zones_mgr.draw_all(disp, alpha=0.30, show_labels=True)
            self.canvas.set_frame(disp)
        elif self.source_mode == "NONE":
            self.canvas.set_frame(None)

    # -------------------------------------------------------------
    # Arena & Grid Interactions
    # -------------------------------------------------------------
    def activate_draw_arena_mode(self, shape: str = "rectangle"):
        if self.source_mode == "NONE":
            QMessageBox.warning(
                self, "Pilih Video Dahulu",
                "Silakan pilih sumber video (Kamera, File Video, atau Simulasi) terlebih dahulu sebelum menggambar zona!"
            )
            return
        self.grid_arena.shape_type = shape
        self.canvas.interaction_mode = "DRAW_ARENA_RECT" if shape == "rectangle" else "DRAW_ARENA_ELLIPSE"
        shape_name = "Kotak" if shape == "rectangle" else "Elips"
        self.lbl_status_badge.setText(f"[ 🖱️ KLIK & TARIK MOUSE UNTUK GAMBAR {shape_name.upper()} ARENA ]")
        self.lbl_status_badge.setStyleSheet(
            "background-color: #0369a1; color: #38bdf8; font-weight: bold; padding: 6px 12px; border-radius: 4px; border: 1px solid #38bdf8;"
        )

    def activate_adjust_arena_mode(self):
        self.canvas.interaction_mode = "ADJUST_ARENA"
        self.render_preview_frame()

    def reset_arena_zone(self):
        self.grid_arena.is_defined = False
        self.canvas.interaction_mode = "ADJUST_ARENA"
        self.lbl_status_badge.setText("[ ✏️ LANGKAH 2: GAMBAR ZONA TRACKING ]")
        self.lbl_status_badge.setStyleSheet(
            "background-color: #1e293b; color: #38bdf8; font-weight: bold; padding: 6px 12px; border-radius: 4px; border: 1px solid #38bdf8;"
        )
        self.render_preview_frame()
        self.update_zone_table()

    def on_real_length_changed(self):
        self.grid_arena.real_length_cm = self.spin_real_cm.value()
        self.on_arena_bbox_updated()

    def on_grid_type_selection_changed(self, text: str):
        self.panel_square.setVisible("Square" in text)
        self.panel_concentric.setVisible("Lingkaran" in text)
        self.panel_radial.setVisible("Radial" in text)
        self.panel_polar.setVisible("Polar" in text)

        if "Square" in text:
            self.grid_arena.set_grid_type("square")
        elif "Lingkaran" in text:
            self.grid_arena.set_grid_type("concentric")
        elif "Radial" in text:
            self.grid_arena.set_grid_type("radial")
        elif "Polar" in text:
            self.grid_arena.set_grid_type("polar")
        else:
            self.grid_arena.set_grid_type("none")

        self.update_grid_dimension_info()
        self.render_preview_frame()
        self.update_zone_table()

    def on_concentric_rings_changed(self, idx: int):
        ring_counts = [2, 3, 4]
        if 0 <= idx < len(ring_counts):
            self.grid_arena.set_concentric_ring_count(ring_counts[idx])
            self.update_grid_dimension_info()
            self.render_preview_frame()
            self.update_zone_table()

    def on_radial_sectors_changed(self, idx: int):
        sector_counts = [4, 6, 8, 12]
        if 0 <= idx < len(sector_counts):
            self.grid_arena.set_radial_sector_count(sector_counts[idx])
            self.render_preview_frame()
            self.update_zone_table()

    def on_radial_rotation_changed(self, val: float):
        self.grid_arena.radial_angle_offset_deg = float(val)
        self.render_preview_frame()

    def on_polar_sectors_changed(self, idx: int):
        sectors = [4, 8]
        if 0 <= idx < len(sectors):
            self.grid_arena.polar_sector_count = sectors[idx]
            self.render_preview_frame()
            self.update_zone_table()

    def on_polar_center_changed(self, idx: int):
        fracs = [0.25, 0.35, 0.50]
        if 0 <= idx < len(fracs):
            self.grid_arena.polar_center_fraction = fracs[idx]
            self.render_preview_frame()
            self.update_zone_table()

    def update_grid_dimension_info(self):
        real_len = self.grid_arena.real_length_cm
        # 1. Square grid info
        cell_w_cm = real_len / self.grid_arena.cols
        cell_h_cm = real_len / self.grid_arena.rows
        self.lbl_square_cell_info.setText(f"Ukuran tiap petak: {cell_w_cm:.1f} x {cell_h_cm:.1f} cm")

        # 2. Concentric rings info
        ring_lines = []
        prev_d = 0.0
        for frac, name in sorted(self.grid_arena.concentric_rings, key=lambda x: x[0]):
            curr_d = frac * real_len
            ring_lines.append(f"• {name}: {prev_d:.1f} - {curr_d:.1f} cm")
            prev_d = curr_d
        self.lbl_concentric_info.setText("\n".join(ring_lines))

    def activate_paint_cell_mode(self):
        self.canvas.interaction_mode = "PAINT_CELL"

    def on_arena_bbox_updated(self):
        if self.last_base_frame is not None:
            h, w = self.last_base_frame.shape[:2]
            px_per_cm = self.grid_arena.calculate_pixels_per_cm(w, h)
            self.tracker.pixels_per_cm = px_per_cm
            self.lbl_hud_scale.setText(f"{px_per_cm:.1f} px/cm")
            cm_per_px = 1.0 / max(0.001, px_per_cm)
            self.lbl_scale_calc.setText(f"Skala: {px_per_cm:.2f} px/cm (1 px = {cm_per_px:.2f} cm)")
            self.update_grid_dimension_info()

        if getattr(self.grid_arena, "is_defined", False):
            if self.experiment_state == "SETUP":
                self.lbl_status_badge.setText("[ 🟢 SIAP TRACKING ]")
                self.lbl_status_badge.setStyleSheet(
                    "background-color: #064e3b; color: #34d399; font-weight: bold; padding: 6px 12px; border-radius: 4px; border: 1px solid #10b981;"
                )
        self.render_preview_frame()

    def on_arena_type_changed(self, text: str):
        if "Arena Kustom" in text:
            self.active_arena_type = "Tracking Arena"
            self.grp_draw_arena.setVisible(True)
            self.grp_scale.setVisible(True)
            self.grp_grid_ctrl.setVisible(True)
        elif "Morris Water Maze" in text:
            self.active_arena_type = "Morris Water Maze"
            self.grp_draw_arena.setVisible(False)
            self.grp_scale.setVisible(False)
            self.grp_grid_ctrl.setVisible(False)
        elif "Elevated Plus Maze" in text:
            self.active_arena_type = "Elevated Plus Maze"
            self.grp_draw_arena.setVisible(False)
            self.grp_scale.setVisible(False)
            self.grp_grid_ctrl.setVisible(False)

        self.render_preview_frame()
        self.update_zone_table()

    def on_grid_dim_changed(self):
        rows = self.spin_rows.value()
        cols = self.spin_cols.value()
        self.grid_arena.set_dimensions(rows, cols)
        self.update_grid_dimension_info()
        self.render_preview_frame()

    def on_auto_label_clicked(self):
        self.grid_arena.auto_label_grid()
        self.render_preview_frame()

    def on_paint_tool_changed(self, label: str):
        self.canvas.active_paint_label = label

    def handle_cell_click(self, px: int, py: int, paint_label: str):
        if self.last_base_frame is None or self.grid_arena.grid_type != "square":
            return
        h, w = self.last_base_frame.shape[:2]
        res = self.grid_arena.get_cell_at_pixel(px, py, w, h)
        if res is not None:
            r, c, cell = res
            self.grid_arena.set_cell_label(r, c, paint_label)
            self.render_preview_frame()

    def start_scale_calibration(self):
        self.canvas.interaction_mode = "CALIBRATE"
        self.canvas.calib_p1_px = None
        self.canvas.calib_p2_px = None
        QMessageBox.information(
            self, "Kalibrasi Skala Manual",
            "Klik 2 titik di atas arena video (misalnya panjang sisi kotak arena)\n"
            "untuk menentukan panjang sentimeter nyata."
        )

    def handle_calibration_done(self, p1: Tuple[int, int], p2: Tuple[int, int]):
        real_cm, ok = QInputDialog.getDouble(
            self, "Panjang Riil", "Masukkan jarak nyata antara 2 titik (cm):",
            50.0, 1.0, 500.0, 1
        )
        if ok and real_cm > 0:
            px_per_cm = self.tracker.set_calibration(p1, p2, real_cm)
            self.lbl_hud_scale.setText(f"{px_per_cm:.2f} px/cm")
            self.spin_real_cm.setValue(real_cm)
            self.lbl_scale_calc.setText(f"Skala: {px_per_cm:.2f} px/cm")

    # -------------------------------------------------------------
    # Custom Free Zones Management
    # -------------------------------------------------------------
    def start_draw_custom_zone(self, shape_type: str):
        name, ok = QInputDialog.getText(
            self, "Nama Zona Bebas", f"Masukkan nama untuk zona {shape_type} (misal: Objek A, Shelter):"
        )
        if not ok or not name.strip():
            return

        self.pending_cz_name = name.strip()
        self.pending_cz_shape = shape_type

        # Preset palette colors
        colors = ["#ec4899", "#8b5cf6", "#3b82f6", "#14b8a6", "#f97316", "#eab308"]
        col_idx = len(self.custom_zones_mgr.zones) % len(colors)
        self.pending_cz_color = colors[col_idx]

        if shape_type == "rectangle":
            self.canvas.interaction_mode = "DRAW_CUSTOM_RECT"
        elif shape_type == "circle":
            self.canvas.interaction_mode = "DRAW_CUSTOM_CIRCLE"
        elif shape_type == "polygon":
            self.canvas.interaction_mode = "DRAW_CUSTOM_POLY"
            self.canvas.poly_points_norm.clear()

    def handle_custom_zone_drawn(self, shape_type: str, points_norm: List[Tuple[float, float]]):
        zone = self.custom_zones_mgr.add_zone(
            name=self.pending_cz_name,
            shape_type=shape_type,
            points_norm=points_norm,
            color_hex=self.pending_cz_color
        )
        self.refresh_custom_zones_table()
        self.render_preview_frame()

    def refresh_custom_zones_table(self):
        self.table_cz.setRowCount(len(self.custom_zones_mgr.zones))
        for r, z in enumerate(self.custom_zones_mgr.zones):
            # Name with color badge
            it_name = QTableWidgetItem(z.name)
            it_name.setForeground(QColor(z.color_hex))
            self.table_cz.setItem(r, 0, it_name)

            it_type = QTableWidgetItem(z.shape_type.capitalize())
            self.table_cz.setItem(r, 1, it_type)

            btn_del = QPushButton("Hapus")
            btn_del.setFixedWidth(60)
            btn_del.clicked.connect(lambda _, zid=z.id: self.delete_custom_zone(zid))
            self.table_cz.setCellWidget(r, 2, btn_del)

    def delete_custom_zone(self, zone_id: str):
        self.custom_zones_mgr.remove_zone(zone_id)
        self.refresh_custom_zones_table()
        self.render_preview_frame()

    def clear_all_custom_zones(self):
        self.custom_zones_mgr.clear()
        self.refresh_custom_zones_table()
        self.render_preview_frame()

    # -------------------------------------------------------------
    # Main Timer Tick & Real-time Tracking Loop
    # -------------------------------------------------------------
    def on_timer_tick(self):
        if self.experiment_state != "TRACKING":
            return

        frame_bgr = None
        if self.source_mode == "SIMULATION" and self.simulation is not None:
            frame_bgr = self.simulation.get_next_frame()
        elif self.source_mode == "CAMERA" and self.cam_stream is not None:
            ret, fr = self.cam_stream.read()
            if ret:
                frame_bgr = fr
        elif self.source_mode == "FILE" and self.video_cap is not None:
            ret, fr = self.video_cap.read()
            if ret:
                frame_bgr = fr
            else:
                self.stop_tracking()
                return

        if frame_bgr is None:
            return

        self.last_base_frame = frame_bgr.copy()
        self.current_frame_idx += 1
        h, w = frame_bgr.shape[:2]

        arena_obj = None
        if self.active_arena_type in ("Tracking Arena", "Arena Kustom (Kotak / Elips)", "Open Field Grid"):
            arena_obj = self.grid_arena
        elif self.active_arena_type == "Morris Water Maze":
            arena_obj = self.mwm_arena
        elif self.active_arena_type == "Elevated Plus Maze":
            arena_obj = self.epm_arena

        # Computer Vision Tracking
        vis_frame, det_data, mask_preview = self.tracker.process_frame(
            frame_bgr,
            frame_idx=self.current_frame_idx,
            fps=self.video_fps,
            arena_obj=arena_obj,
            active_arena_type=self.active_arena_type,
            custom_zones_mgr=self.custom_zones_mgr
        )

        display_frame = vis_frame.copy()

        # Draw Arena overlay
        if self.chk_show_grid.isChecked() and arena_obj is not None:
            if hasattr(arena_obj, "is_defined") and not arena_obj.is_defined:
                pass
            else:
                display_frame = arena_obj.draw_overlay(display_frame, alpha=0.25, show_text=True)

        # Draw Custom Free Zones
        display_frame = self.custom_zones_mgr.draw_all(display_frame, alpha=0.30, show_labels=True)

        # Draw Heatmap
        if self.chk_show_heat.isChecked():
            heat_overlay = self.tracker.generate_heatmap_overlay(w, h, alpha=0.55)
            mask = cv2.cvtColor(heat_overlay, cv2.COLOR_BGR2GRAY) > 10
            display_frame[mask] = cv2.addWeighted(heat_overlay, 0.55, display_frame, 0.45, 0)[mask]

        # Draw Trajectory
        if self.chk_show_traj.isChecked():
            display_frame = self.tracker.draw_trajectory(display_frame, trail_length=400, show_full=False)

        self.canvas.set_frame(display_frame)

        # Mask thumbnail
        small_mask = cv2.resize(mask_preview, (140, 80))
        m_h, m_w, m_ch = small_mask.shape
        q_mask = QImage(small_mask.data, m_w, m_h, m_ch * m_w, QImage.Format_BGR888)
        self.lbl_mask_preview.setPixmap(QPixmap.fromImage(q_mask))

        # HUD & Tables
        self.update_hud(det_data)
        if self.current_frame_idx % 3 == 0:
            self.update_zone_table()

    def update_hud(self, det_data: Optional[Dict]):
        dist = self.tracker.total_distance_cm
        if dist > 100.0:
            self.lbl_hud_dist.setText(f"{dist:.1f} cm ({dist/100:.2f} m)")
        else:
            self.lbl_hud_dist.setText(f"{dist:.1f} cm")

        cur_time_s = self.current_frame_idx / self.video_fps
        total_time_s = self.total_frames / self.video_fps if self.total_frames < 10000 else cur_time_s
        self.lbl_time.setText(f"{cur_time_s:.1f}s / {total_time_s:.1f}s (F: {self.current_frame_idx})")
        if self.source_mode == "FILE":
            self.slider_seek.setValue(self.current_frame_idx)

        if det_data is not None:
            spd = det_data["speed_cm_s"]
            self.lbl_hud_speed.setText(f"{spd:.1f} cm/s")

            zone_str = det_data['zone']
            if det_data.get('cell_id') not in ("None", "Outside"):
                zone_str = f"{det_data['zone']} ({det_data['cell_id']})"
            self.lbl_hud_zone.setText(zone_str)

            cz_list = det_data.get("custom_zones", [])
            self.lbl_hud_custom.setText(", ".join(cz_list) if cz_list else "-")

            if det_data["is_freezing"]:
                self.lbl_hud_state.setText("FREEZING (DIAM)")
                self.lbl_hud_state.setStyleSheet("color: #f43f5e; font-weight: bold; font-size: 13px;")
            else:
                self.lbl_hud_state.setText("MOVING (BERGERAK)")
                self.lbl_hud_state.setStyleSheet("color: #10b981; font-weight: bold; font-size: 13px;")
        else:
            self.lbl_hud_speed.setText("0.0 cm/s")
            self.lbl_hud_zone.setText("-")
            self.lbl_hud_custom.setText("-")
            self.lbl_hud_state.setText("STANDBY")
            self.lbl_hud_state.setStyleSheet("color: #94a3b8; font-weight: bold;")

        fz_time = self.tracker.freezing_frames_total / self.video_fps
        fz_pct = (fz_time / cur_time_s * 100.0) if cur_time_s > 0 else 0.0
        self.lbl_hud_freeze.setText(f"{fz_time:.1f} s ({fz_pct:.1f}%)")

    def update_zone_table(self):
        summary = self.tracker.get_summary_stats(self.video_fps)
        zones = summary.get("zones", {})
        custom_zones = summary.get("custom_zones", {})

        total_rows = len(zones) + len(custom_zones)
        self.table_zones.setRowCount(total_rows)

        row_idx = 0
        for z_name, z_data in zones.items():
            it_name = QTableWidgetItem(z_name)
            hex_col = LABEL_COLORS_HEX.get(z_name, "#38bdf8")
            it_name.setForeground(QColor(hex_col))
            self._set_table_row(row_idx, it_name, z_data)
            row_idx += 1

        for cz_name, cz_data in custom_zones.items():
            it_name = QTableWidgetItem(f"★ {cz_name}")
            it_name.setForeground(QColor("#ec4899"))
            self._set_table_row(row_idx, it_name, cz_data)
            row_idx += 1

    def _set_table_row(self, row_idx: int, item_name: QTableWidgetItem, data: Dict):
        it_time = QTableWidgetItem(f"{data['time_s']:.1f} s")
        it_pct = QTableWidgetItem(f"{data['pct_time']:.1f} %")
        it_ent = QTableWidgetItem(str(data['entries']))
        for itm in (item_name, it_time, it_pct, it_ent):
            itm.setTextAlignment(Qt.AlignCenter)
        self.table_zones.setItem(row_idx, 0, item_name)
        self.table_zones.setItem(row_idx, 1, it_time)
        self.table_zones.setItem(row_idx, 2, it_pct)
        self.table_zones.setItem(row_idx, 3, it_ent)

    # -------------------------------------------------------------
    # CV Param Handlers & Playback
    # -------------------------------------------------------------
    def on_thresh_slider_changed(self, val: int):
        self.tracker.threshold_val = val
        self.lbl_thresh_val.setText(str(val))

    def on_animal_color_changed(self):
        self.tracker.is_dark_animal = self.rb_dark_animal.isChecked()

    def on_speed_changed(self, text: str):
        speed_val = float(text.replace("x", ""))
        self.playback_speed = speed_val
        self.update_timer_interval()

    def update_timer_interval(self):
        effective_fps = max(1.0, self.video_fps * self.playback_speed)
        interval_ms = int(1000.0 / effective_fps)
        self.timer.setInterval(max(5, interval_ms))

    def step_forward(self):
        self.on_timer_tick()

    def step_backward(self):
        if self.source_mode == "FILE" and self.video_cap is not None:
            target_f = max(0, self.current_frame_idx - 2)
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
            self.current_frame_idx = target_f
            self.on_timer_tick()

    def on_seek_moved(self, pos: int):
        if self.source_mode == "FILE" and self.video_cap is not None:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            self.current_frame_idx = pos
            self.render_preview_frame()

    # -------------------------------------------------------------
    # Data Export Handlers
    # -------------------------------------------------------------
    def export_summary_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Simpan Ringkasan CSV", "summary_report.csv", "CSV Files (*.csv)")
        if not path:
            return
        stats = self.tracker.get_summary_stats(self.video_fps)
        ok = ExportManager.export_summary_csv(path, stats, self.active_arena_type)
        if ok:
            QMessageBox.information(self, "Ekspor Berhasil", f"Data ringkasan berhasil disimpan ke:\n{path}")
        else:
            QMessageBox.critical(self, "Error", "Gagal menyimpan file ringkasan CSV.")

    def export_raw_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Simpan Koordinat Mentah CSV", "raw_trajectory.csv", "CSV Files (*.csv)")
        if not path:
            return
        ok = ExportManager.export_raw_trajectory_csv(path, self.tracker.history)
        if ok:
            QMessageBox.information(self, "Ekspor Berhasil", f"Data koordinat ({len(self.tracker.history)} baris) berhasil disimpan ke:\n{path}")
        else:
            QMessageBox.critical(self, "Error", "Gagal menyimpan file raw trajectory CSV.")

    def export_trajectory_png(self):
        if self.last_base_frame is None:
            QMessageBox.warning(self, "Peringatan", "Belum ada frame video untuk diekspor.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Simpan Gambar Lintasan PNG", "trajectory_plot.png", "PNG Images (*.png)")
        if not path:
            return
        arena_obj = self.grid_arena if self.active_arena_type == "Open Field Grid" else None
        ok = ExportManager.export_trajectory_image(
            path, self.last_base_frame, self.tracker.history,
            arena_obj=arena_obj, custom_zones_mgr=self.custom_zones_mgr
        )
        if ok:
            QMessageBox.information(self, "Ekspor Berhasil", f"Gambar lintasan berhasil disimpan ke:\n{path}")

    def export_heatmap_png(self):
        if self.last_base_frame is None:
            QMessageBox.warning(self, "Peringatan", "Belum ada frame video untuk diekspor.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Simpan Gambar Heatmap PNG", "heatmap_plot.png", "PNG Images (*.png)")
        if not path:
            return
        h, w = self.last_base_frame.shape[:2]
        heat_overlay = self.tracker.generate_heatmap_overlay(w, h, alpha=0.6)
        ok = ExportManager.export_heatmap_image(path, self.last_base_frame, heat_overlay)
        if ok:
            QMessageBox.information(self, "Ekspor Berhasil", f"Gambar heatmap berhasil disimpan ke:\n{path}")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
