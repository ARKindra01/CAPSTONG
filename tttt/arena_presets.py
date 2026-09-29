"""
arena_presets.py - Additional Behavioral Apparatus Presets:
1. Morris Water Maze (MWM): Circular pool with 4 quadrants and Target Platform.
2. Elevated Plus Maze (EPM): Cross-shaped maze with Open Arms, Closed Arms, and Center.
"""

from typing import Dict, List, Optional, Tuple
import math
import cv2
import numpy as np


class MorrisWaterMazeArena:
    def __init__(self):
        # Normalized circle center (cx, cy) and radius r (relative to frame dimensions)
        self.norm_center = [0.50, 0.50]
        self.norm_radius = 0.38
        # Normalized target platform (default in NW quadrant)
        self.norm_platform = [0.38, 0.38, 0.08]  # [cx, cy, radius]

    def get_pixels(self, frame_w: int, frame_h: int):
        scale = min(frame_w, frame_h)
        cx = int(self.norm_center[0] * frame_w)
        cy = int(self.norm_center[1] * frame_h)
        r = int(self.norm_radius * scale)

        pcx = int(self.norm_platform[0] * frame_w)
        pcy = int(self.norm_platform[1] * frame_h)
        pr = int(self.norm_platform[2] * scale)
        return (cx, cy, r), (pcx, pcy, pr)

    def set_arena_pixels(self, cx: int, cy: int, r: int, frame_w: int, frame_h: int):
        scale = min(frame_w, frame_h)
        self.norm_center = [cx / frame_w, cy / frame_h]
        self.norm_radius = r / scale

    def set_platform_pixels(self, pcx: int, pcy: int, pr: int, frame_w: int, frame_h: int):
        scale = min(frame_w, frame_h)
        self.norm_platform = [pcx / frame_w, pcy / frame_h, pr / scale]

    def get_zone_at_pixel(self, px: float, py: float, frame_w: int, frame_h: int) -> Optional[Tuple[str, str]]:
        """Returns (zone_label, detailed_id) or None if outside pool."""
        (cx, cy, r), (pcx, pcy, pr) = self.get_pixels(frame_w, frame_h)

        # 1. Check distance to pool center
        dist_to_center = math.hypot(px - cx, py - cy)
        if dist_to_center > r:
            return None

        # 2. Check if inside Target Platform
        dist_to_plat = math.hypot(px - pcx, py - pcy)
        if dist_to_plat <= pr:
            return "Target Platform", "Platform"

        # 3. Determine Quadrant (NE, NW, SE, SW)
        dx = px - cx
        dy = py - cy
        if dx >= 0 and dy < 0:
            quad = "NE Quadrant"
        elif dx < 0 and dy < 0:
            quad = "NW Quadrant"
        elif dx < 0 and dy >= 0:
            quad = "SW Quadrant"
        else:
            quad = "SE Quadrant"

        return quad, quad

    def get_all_categories(self) -> List[str]:
        return ["Target Platform", "NW Quadrant", "NE Quadrant", "SW Quadrant", "SE Quadrant"]

    def draw_overlay(self, frame_bgr: np.ndarray, alpha: float = 0.25, show_text: bool = True) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        (cx, cy, r), (pcx, pcy, pr) = self.get_pixels(w, h)

        overlay = frame_bgr.copy()

        # Quadrant colors (BGR)
        quad_colors = {
            "NW": (230, 180, 50),   # Cyan
            "NE": (60, 200, 100),   # Green
            "SW": (50, 100, 240),   # Orange
            "SE": (200, 70, 200)    # Violet
        }

        # Draw quadrants using filled circle masks / angles
        for start_ang, end_ang, qname in [(180, 270, "NW"), (270, 360, "NE"), (90, 180, "SW"), (0, 90, "SE")]:
            cv2.ellipse(overlay, (cx, cy), (r, r), 0, start_ang, end_ang, quad_colors[qname], -1)

        # Draw platform
        cv2.circle(overlay, (pcx, pcy), pr, (0, 0, 255), -1)

        blended = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)

        # Clean outlines
        cv2.circle(blended, (cx, cy), r, (255, 255, 255), 2)
        cv2.line(blended, (cx - r, cy), (cx + r, cy), (255, 255, 255), 1)
        cv2.line(blended, (cx, cy - r), (cx, cy + r), (255, 255, 255), 1)
        cv2.circle(blended, (pcx, pcy), pr, (50, 50, 255), 2)

        if show_text:
            font_scale = 0.5
            cv2.putText(blended, "NW", (cx - r // 2, cy - r // 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(blended, "NE", (cx + r // 2 - 20, cy - r // 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(blended, "SW", (cx - r // 2, cy + r // 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(blended, "SE", (cx + r // 2 - 20, cy + r // 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(blended, "Platform", (pcx - 24, pcy - pr - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        return blended


class ElevatedPlusMazeArena:
    def __init__(self):
        # Cross geometry defined relative to center
        self.norm_center = [0.50, 0.50]
        self.norm_arm_length = 0.35
        self.norm_arm_width = 0.10

    def get_pixels(self, frame_w: int, frame_h: int):
        scale = min(frame_w, frame_h)
        cx = int(self.norm_center[0] * frame_w)
        cy = int(self.norm_center[1] * frame_h)
        L = int(self.norm_arm_length * scale)
        W = int(self.norm_arm_width * scale) // 2
        return cx, cy, L, W

    def get_zone_at_pixel(self, px: float, py: float, frame_w: int, frame_h: int) -> Optional[Tuple[str, str]]:
        cx, cy, L, W = self.get_pixels(frame_w, frame_h)

        # Center square: [cx - W .. cx + W], [cy - W .. cy + W]
        if abs(px - cx) <= W and abs(py - cy) <= W:
            return "Center Area", "Center"

        # Open arms (Horizontal: East & West)
        if abs(py - cy) <= W:
            if cx + W < px <= cx + L:
                return "Open Arms", "East Open Arm"
            elif cx - L <= px < cx - W:
                return "Open Arms", "West Open Arm"

        # Closed arms (Vertical: North & South)
        if abs(px - cx) <= W:
            if cy - L <= py < cy - W:
                return "Closed Arms", "North Closed Arm"
            elif cy + W < py <= cy + L:
                return "Closed Arms", "South Closed Arm"

        return None

    def get_all_categories(self) -> List[str]:
        return ["Open Arms", "Closed Arms", "Center Area"]

    def draw_overlay(self, frame_bgr: np.ndarray, alpha: float = 0.25, show_text: bool = True) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        cx, cy, L, W = self.get_pixels(w, h)

        overlay = frame_bgr.copy()

        # Open arms (Bright Yellow/Green)
        cv2.rectangle(overlay, (cx - L, cy - W), (cx - W, cy + W), (50, 220, 220), -1)
        cv2.rectangle(overlay, (cx + W, cy - W), (cx + L, cy + W), (50, 220, 220), -1)

        # Closed arms (Darker Blue/Purple)
        cv2.rectangle(overlay, (cx - W, cy - L), (cx + W, cy - W), (220, 100, 60), -1)
        cv2.rectangle(overlay, (cx - W, cy + W), (cx + W, cy + L), (220, 100, 60), -1)

        # Center (White/Cyan)
        cv2.rectangle(overlay, (cx - W, cy - W), (cx + W, cy + W), (200, 200, 200), -1)

        blended = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)

        # Outlines
        pts_poly = np.array([
            [cx - W, cy - L], [cx + W, cy - L], [cx + W, cy - W],
            [cx + L, cy - W], [cx + L, cy + W], [cx + W, cy + W],
            [cx + W, cy + L], [cx - W, cy + L], [cx - W, cy + W],
            [cx - L, cy + W], [cx - L, cy - W], [cx - W, cy - W]
        ], np.int32)
        cv2.polylines(blended, [pts_poly], True, (255, 255, 255), 2)

        if show_text:
            cv2.putText(blended, "Open", (cx + L // 2, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(blended, "Open", (cx - L // 2 - 20, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(blended, "Closed", (cx - 20, cy - L // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(blended, "Closed", (cx - 20, cy + L // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return blended
