"""
custom_zones.py - Custom Freeform & Geometric Zone Management
Supports user-defined arbitrary Polygons, Rectangles, and Circles for
Novel Object Recognition, Social Interaction, Food/Water zones, Shelters, etc.
"""

from typing import Dict, List, Optional, Tuple
import math
import uuid
import cv2
import numpy as np


class CustomZone:
    def __init__(
        self,
        name: str,
        shape_type: str,  # "polygon", "rectangle", "circle"
        points_norm: List[Tuple[float, float]],  # Normalized (0.0 - 1.0) coordinates
        color_hex: str = "#ec4899",
        zone_id: Optional[str] = None
    ):
        self.id = zone_id if zone_id else str(uuid.uuid4())[:8]
        self.name = name
        self.shape_type = shape_type
        self.points_norm = points_norm  # [(x, y), ...]
        self.color_hex = color_hex
        self.color_bgr = self._hex_to_bgr(color_hex)
        self.is_visible = True

    @staticmethod
    def _hex_to_bgr(hex_str: str) -> Tuple[int, int, int]:
        h = hex_str.lstrip("#")
        if len(h) == 6:
            r = int(h[0:2], 16)
            g = int(h[2:4], 16)
            b = int(h[4:6], 16)
            return (b, g, r)
        return (255, 100, 200)

    def is_point_inside(self, px: float, py: float, frame_w: int, frame_h: int) -> bool:
        """Checks whether pixel coordinate (px, py) is inside this custom zone."""
        if not self.points_norm:
            return False

        if self.shape_type == "rectangle":
            if len(self.points_norm) < 2:
                return False
            p1 = (self.points_norm[0][0] * frame_w, self.points_norm[0][1] * frame_h)
            p2 = (self.points_norm[1][0] * frame_w, self.points_norm[1][1] * frame_h)
            min_x, max_x = min(p1[0], p2[0]), max(p1[0], p2[0])
            min_y, max_y = min(p1[1], p2[1]), max(p1[1], p2[1])
            return min_x <= px <= max_x and min_y <= py <= max_y

        elif self.shape_type == "circle":
            if len(self.points_norm) < 2:
                return False
            cx = self.points_norm[0][0] * frame_w
            cy = self.points_norm[0][1] * frame_h
            ex = self.points_norm[1][0] * frame_w
            ey = self.points_norm[1][1] * frame_h
            radius = math.hypot(ex - cx, ey - cy)
            return math.hypot(px - cx, py - cy) <= radius

        elif self.shape_type == "polygon":
            if len(self.points_norm) < 3:
                return False
            pts_px = np.array(
                [[int(x * frame_w), int(y * frame_h)] for x, y in self.points_norm],
                dtype=np.int32
            )
            # cv2.pointPolygonTest returns >= 0 if inside or on boundary
            return cv2.pointPolygonTest(pts_px, (float(px), float(py)), False) >= 0

        return False

    def draw(self, frame_bgr: np.ndarray, alpha: float = 0.30, show_label: bool = True) -> np.ndarray:
        """Renders filled translucent shape with border and text tag."""
        if not self.is_visible or not self.points_norm:
            return frame_bgr

        h, w = frame_bgr.shape[:2]
        overlay = frame_bgr.copy()
        color = self.color_bgr

        label_pos = (50, 50)

        if self.shape_type == "rectangle" and len(self.points_norm) >= 2:
            x1 = int(self.points_norm[0][0] * w)
            y1 = int(self.points_norm[0][1] * h)
            x2 = int(self.points_norm[1][0] * w)
            y2 = int(self.points_norm[1][1] * h)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            label_pos = (min(x1, x2) + 6, min(y1, y2) + 16)

        elif self.shape_type == "circle" and len(self.points_norm) >= 2:
            cx = int(self.points_norm[0][0] * w)
            cy = int(self.points_norm[0][1] * h)
            ex = int(self.points_norm[1][0] * w)
            ey = int(self.points_norm[1][1] * h)
            rad = int(math.hypot(ex - cx, ey - cy))
            cv2.circle(overlay, (cx, cy), rad, color, -1)
            cv2.circle(frame_bgr, (cx, cy), rad, color, 2)
            label_pos = (cx - 20, cy - rad - 6)

        elif self.shape_type == "polygon" and len(self.points_norm) >= 3:
            pts_px = np.array(
                [[int(x * w), int(y * h)] for x, y in self.points_norm],
                dtype=np.int32
            )
            cv2.fillPoly(overlay, [pts_px], color)
            cv2.polylines(frame_bgr, [pts_px], True, color, 2)
            label_pos = (pts_px[0][0] + 6, pts_px[0][1] + 16)

        # Blend
        blended = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)

        # Label tag
        if show_label:
            (tw, th), _ = cv2.getTextSize(self.name, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            lx, ly = label_pos
            lx = max(4, min(lx, w - tw - 10))
            ly = max(th + 4, min(ly, h - 6))

            # Dark background pill for readability
            cv2.rectangle(blended, (lx - 3, ly - th - 3), (lx + tw + 3, ly + 3), (15, 20, 30), -1)
            cv2.putText(blended, self.name, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return blended


class CustomZoneManager:
    def __init__(self):
        self.zones: List[CustomZone] = []

    def add_zone(self, name: str, shape_type: str, points_norm: List[Tuple[float, float]], color_hex: str = "#ec4899") -> CustomZone:
        zone = CustomZone(name, shape_type, points_norm, color_hex)
        self.zones.append(zone)
        return zone

    def remove_zone(self, zone_id: str) -> bool:
        for i, z in enumerate(self.zones):
            if z.id == zone_id:
                self.zones.pop(i)
                return True
        return False

    def clear(self):
        self.zones.clear()

    def get_zones_containing_point(self, px: float, py: float, frame_w: int, frame_h: int) -> List[CustomZone]:
        return [z for z in self.zones if z.is_point_inside(px, py, frame_w, frame_h)]

    def draw_all(self, frame_bgr: np.ndarray, alpha: float = 0.30, show_labels: bool = True) -> np.ndarray:
        res = frame_bgr
        for z in self.zones:
            res = z.draw(res, alpha=alpha, show_label=show_labels)
        return res
