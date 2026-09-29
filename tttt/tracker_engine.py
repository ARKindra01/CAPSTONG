"""
tracker_engine.py - Core Computer Vision & Animal Behavior Tracking Engine
Performs thresholding, contour filtering, centroid tracking, trajectory smoothing,
freezing detection, zone statistics accumulation, and 2D occupancy heatmap generation.
"""

from typing import Dict, List, Optional, Tuple
import math
import cv2
import numpy as np


class AnimalTracker:
    def __init__(self):
        # Tracking hyperparameters
        self.threshold_val: int = 100
        self.is_dark_animal: bool = True  # True: dark animal on light background
        self.min_area: int = 80           # Minimum blob size (pixels)
        self.max_area: int = 15000        # Maximum blob size (pixels)
        self.pixels_per_cm: float = 5.0   # Scale calibration (pixels per cm)

        # Freezing detection parameters
        self.freezing_speed_thresh: float = 1.0  # cm/s below which is considered freezing
        self.freezing_min_frames: int = 10       # Consecutive frames to confirm freezing

        # Heatmap resolution
        self.heat_h: int = 120
        self.heat_w: int = 160
        self.heatmap_acc = np.zeros((self.heat_h, self.heat_w), dtype=np.float32)

        # Tracking state
        self.history: List[Dict] = []
        self.total_distance_cm: float = 0.0
        self.max_speed_cm_s: float = 0.0
        self.freezing_frames_total: int = 0
        self.current_freezing_run: int = 0

        # Zone stats: {zone_name: {"time_s": 0.0, "entries": 0, "latency_s": None, "distance_cm": 0.0}}
        self.zone_stats: Dict[str, Dict] = {}
        # Cell stats (specific to Grid arena): {cell_id: {"time_s": 0.0, "entries": 0, "label": ""}}
        self.cell_stats: Dict[str, Dict] = {}
        # Custom freeform zones stats: {cz_name: {"time_s": 0.0, "entries": 0, "latency_s": None, "distance_cm": 0.0}}
        self.custom_zone_stats: Dict[str, Dict] = {}

        self.last_pos_px: Optional[Tuple[float, float]] = None
        self.last_zone: Optional[str] = None
        self.last_cell: Optional[str] = None
        self.last_custom_zones: set = set()

        # Speed smoothing queue
        self.recent_speeds: List[float] = []

    def reset(self):
        """Resets all tracking history and metrics."""
        self.history.clear()
        self.heatmap_acc.fill(0.0)
        self.total_distance_cm = 0.0
        self.max_speed_cm_s = 0.0
        self.freezing_frames_total = 0
        self.current_freezing_run = 0
        self.zone_stats.clear()
        self.cell_stats.clear()
        self.custom_zone_stats.clear()
        self.last_pos_px = None
        self.last_zone = None
        self.last_cell = None
        self.last_custom_zones.clear()
        self.recent_speeds.clear()

    def set_calibration(self, p1: Tuple[int, int], p2: Tuple[int, int], real_cm: float) -> float:
        """Calculates pixels_per_cm based on 2 clicked points and known real cm distance."""
        dist_px = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
        if real_cm > 0 and dist_px > 0:
            self.pixels_per_cm = dist_px / real_cm
            return self.pixels_per_cm
        return self.pixels_per_cm

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        fps: float,
        arena_obj=None,
        active_arena_type: str = "Open Field Grid",
        custom_zones_mgr=None
    ) -> Tuple[np.ndarray, Optional[Dict], np.ndarray]:
        """
        Processes one video frame:
        1. Segments animal blob.
        2. Detects centroid and bounding box.
        3. Computes speed, distance, freezing state, and zone.
        4. Accumulates heatmap and updates trajectory.
        Returns:
            (visualization_frame, detection_dict_or_none, threshold_mask_preview)
        """
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        dt = 1.0 / fps if fps > 0 else 1.0 / 30.0
        current_time_s = frame_idx * dt

        # 1. Thresholding
        if self.is_dark_animal:
            # Dark animal on light background: pixels darker than threshold become white
            _, thresh = cv2.threshold(gray, self.threshold_val, 255, cv2.THRESH_BINARY_INV)
        else:
            # Light animal on dark background: pixels brighter than threshold become white
            _, thresh = cv2.threshold(gray, self.threshold_val, 255, cv2.THRESH_BINARY)

        # Morphological clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # Apply arena mask if available to reject noise/objects outside the arena
        if arena_obj is not None:
            if hasattr(arena_obj, "get_mask") and getattr(arena_obj, "is_defined", True):
                mask_arena = arena_obj.get_mask(w, h)
                thresh_clean = cv2.bitwise_and(thresh_clean, thresh_clean, mask=mask_arena)
            elif active_arena_type == "Open Field Grid" and hasattr(arena_obj, "get_bbox_pixels"):
                mask_arena = np.zeros((h, w), dtype=np.uint8)
                bx, by, bw, bh = arena_obj.get_bbox_pixels(w, h)
                cv2.rectangle(mask_arena, (bx, by), (bx + bw, by + bh), 255, -1)
                thresh_clean = cv2.bitwise_and(thresh_clean, thresh_clean, mask=mask_arena)
            elif active_arena_type == "Morris Water Maze" and hasattr(arena_obj, "get_pixels"):
                mask_arena = np.zeros((h, w), dtype=np.uint8)
                (cx, cy, r), _ = arena_obj.get_pixels(w, h)
                cv2.circle(mask_arena, (cx, cy), r, 255, -1)
                thresh_clean = cv2.bitwise_and(thresh_clean, thresh_clean, mask=mask_arena)

        # 2. Find contours (using RETR_LIST to catch contours inside arena)
        contours, _ = cv2.findContours(thresh_clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        best_contour = None
        best_area = 0.0
        best_centroid = None
        best_bbox = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area <= area <= self.max_area:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = float(M["m10"] / M["m00"])
                    cy = float(M["m01"] / M["m00"])

                    # If multiple valid blobs, pick largest or closest to last position
                    if self.last_pos_px is not None:
                        dist_prev = math.hypot(cx - self.last_pos_px[0], cy - self.last_pos_px[1])
                        score = area / (1.0 + 0.05 * dist_prev)
                    else:
                        score = area

                    if score > best_area:
                        best_area = score
                        best_contour = cnt
                        best_centroid = (cx, cy)
                        best_bbox = cv2.boundingRect(cnt)

        # 3. Analyze movement if detected
        detection_data = None
        speed_cm_s = 0.0
        is_freezing = False
        current_zone = "Outside"
        current_cell_id = "None"

        if best_centroid is not None:
            cx, cy = best_centroid

            # Spatial distance & speed
            if self.last_pos_px is not None:
                d_px = math.hypot(cx - self.last_pos_px[0], cy - self.last_pos_px[1])
                d_cm = d_px / max(0.001, self.pixels_per_cm)
                inst_speed = d_cm / max(0.001, dt)
                self.total_distance_cm += d_cm
            else:
                d_cm = 0.0
                inst_speed = 0.0

            # Smooth speed over last 5 frames
            self.recent_speeds.append(inst_speed)
            if len(self.recent_speeds) > 5:
                self.recent_speeds.pop(0)
            speed_cm_s = sum(self.recent_speeds) / len(self.recent_speeds)
            self.max_speed_cm_s = max(self.max_speed_cm_s, speed_cm_s)

            # Freezing detection
            if speed_cm_s < self.freezing_speed_thresh:
                self.current_freezing_run += 1
                if self.current_freezing_run >= self.freezing_min_frames:
                    is_freezing = True
                    self.freezing_frames_total += 1
            else:
                self.current_freezing_run = 0

            # Zone determination
            if arena_obj is not None:
                if hasattr(arena_obj, "is_defined") and not arena_obj.is_defined:
                    current_zone = "Full Frame"
                    current_cell_id = "Frame"
                elif hasattr(arena_obj, "get_zone_at_pixel"):
                    res = arena_obj.get_zone_at_pixel(cx, cy, w, h)
                    if res is not None:
                        current_zone, current_cell_id = res
                    else:
                        current_zone = "Outside"
                        current_cell_id = "Outside"
                elif active_arena_type == "Open Field Grid" and hasattr(arena_obj, "get_cell_at_pixel"):
                    res = arena_obj.get_cell_at_pixel(cx, cy, w, h)
                    if res is not None:
                        r, c, cell = res
                        current_zone = cell.label
                        current_cell_id = cell.cell_id
                    else:
                        current_zone = "Outside"
                        current_cell_id = "Outside"
                elif active_arena_type in ("Morris Water Maze", "Elevated Plus Maze"):
                    res = arena_obj.get_zone_at_pixel(cx, cy, w, h)
                    if res is not None:
                        current_zone, current_cell_id = res
                    else:
                        current_zone = "Outside"
                        current_cell_id = "Outside"
            else:
                current_zone = "Full Frame"
                current_cell_id = "Frame"

            # Update zone statistics
            if current_zone != "Outside":
                if current_zone not in self.zone_stats:
                    self.zone_stats[current_zone] = {
                        "time_s": 0.0,
                        "entries": 0,
                        "latency_s": current_time_s,
                        "distance_cm": 0.0,
                    }

                self.zone_stats[current_zone]["time_s"] += dt
                self.zone_stats[current_zone]["distance_cm"] += d_cm

                # Check zone entry transition
                if current_zone != self.last_zone:
                    self.zone_stats[current_zone]["entries"] += 1
                    if self.zone_stats[current_zone]["latency_s"] is None:
                        self.zone_stats[current_zone]["latency_s"] = current_time_s

            # Update cell statistics (for grid)
            if current_cell_id not in ("None", "Outside"):
                if current_cell_id not in self.cell_stats:
                    self.cell_stats[current_cell_id] = {
                        "time_s": 0.0,
                        "entries": 0,
                        "label": current_zone
                    }
                self.cell_stats[current_cell_id]["time_s"] += dt
                if current_cell_id != self.last_cell:
                    self.cell_stats[current_cell_id]["entries"] += 1

            # Update Custom Free Zones statistics
            active_custom_names = []
            if custom_zones_mgr is not None and hasattr(custom_zones_mgr, "get_zones_containing_point"):
                matching_zones = custom_zones_mgr.get_zones_containing_point(cx, cy, w, h)
                for cz in matching_zones:
                    active_custom_names.append(cz.name)
                    if cz.name not in self.custom_zone_stats:
                        self.custom_zone_stats[cz.name] = {
                            "time_s": 0.0,
                            "entries": 0,
                            "latency_s": current_time_s,
                            "distance_cm": 0.0
                        }
                    self.custom_zone_stats[cz.name]["time_s"] += dt
                    self.custom_zone_stats[cz.name]["distance_cm"] += d_cm

                    # Check entry transition
                    if cz.name not in self.last_custom_zones:
                        self.custom_zone_stats[cz.name]["entries"] += 1
                        if self.custom_zone_stats[cz.name]["latency_s"] is None:
                            self.custom_zone_stats[cz.name]["latency_s"] = current_time_s

            self.last_custom_zones = set(active_custom_names)

            # Accumulate 2D heatmap
            hx = int((cx / w) * self.heat_w)
            hy = int((cy / h) * self.heat_h)
            if 0 <= hx < self.heat_w and 0 <= hy < self.heat_h:
                # Add gaussian splash
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        ny, nx = hy + dy, hx + dx
                        if 0 <= nx < self.heat_w and 0 <= ny < self.heat_h:
                            w_dist = math.exp(-(dx*dx + dy*dy) / 2.0)
                            self.heatmap_acc[ny, nx] += w_dist

            self.last_pos_px = (cx, cy)
            self.last_zone = current_zone
            self.last_cell = current_cell_id

            # Save history record
            x_cm = cx / self.pixels_per_cm
            y_cm = cy / self.pixels_per_cm

            detection_data = {
                "frame": frame_idx,
                "time_s": current_time_s,
                "x_px": cx,
                "y_px": cy,
                "x_cm": x_cm,
                "y_cm": y_cm,
                "speed_cm_s": speed_cm_s,
                "zone": current_zone,
                "cell_id": current_cell_id,
                "custom_zones": active_custom_names,
                "is_freezing": is_freezing,
                "bbox": best_bbox
            }
            self.history.append(detection_data)

        # 4. Prepare Threshold preview (convert to 3-channel for UI display)
        thresh_preview = cv2.cvtColor(thresh_clean, cv2.COLOR_GRAY2BGR)

        return frame_bgr, detection_data, thresh_preview

    def draw_trajectory(
        self,
        frame_bgr: np.ndarray,
        trail_length: int = 300,
        show_full: bool = False
    ) -> np.ndarray:
        """Draws trajectory lines with color gradient based on speed or recency."""
        if len(self.history) < 2:
            return frame_bgr

        points_to_draw = self.history if show_full else self.history[-trail_length:]
        n = len(points_to_draw)

        for i in range(1, n):
            p1 = (int(points_to_draw[i - 1]["x_px"]), int(points_to_draw[i - 1]["y_px"]))
            p2 = (int(points_to_draw[i]["x_px"]), int(points_to_draw[i]["y_px"]))

            # Gradient color: Blue (slow) -> Green (medium) -> Red (fast)
            speed = points_to_draw[i]["speed_cm_s"]
            speed_ratio = min(1.0, speed / 25.0)

            # Color interpolation
            b = int(255 * (1.0 - speed_ratio))
            g = int(255 * (1.0 - abs(speed_ratio - 0.5) * 2))
            r = int(255 * speed_ratio)

            # Alpha fade for trailing
            thickness = 2
            cv2.line(frame_bgr, p1, p2, (b, g, r), thickness, cv2.LINE_AA)

        # Draw current position crosshair
        if self.history:
            last = self.history[-1]
            cx, cy = int(last["x_px"]), int(last["y_px"])
            color = (0, 0, 255) if last["is_freezing"] else (0, 255, 0)
            cv2.drawMarker(frame_bgr, (cx, cy), color, cv2.MARKER_CROSS, 16, 2)
            cv2.circle(frame_bgr, (cx, cy), 6, color, 2)

            # Bounding box
            if last["bbox"]:
                bx, by, bw, bh = last["bbox"]
                cv2.rectangle(frame_bgr, (bx, by), (bx + bw, by + bh), color, 1)

        return frame_bgr

    def generate_heatmap_overlay(self, target_w: int, target_h: int, alpha: float = 0.5) -> np.ndarray:
        """
        Generates colored occupancy heatmap resized to target dimensions.
        Returns BGR image ready for blending.
        """
        if np.max(self.heatmap_acc) <= 0:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)

        # Normalize 0 - 255
        norm_heat = (self.heatmap_acc / np.max(self.heatmap_acc) * 255.0).astype(np.uint8)
        norm_heat = cv2.GaussianBlur(norm_heat, (9, 9), 0)

        # Resize to video dimensions
        heat_resized = cv2.resize(norm_heat, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        # Apply scientific colormap (JET or INFERNO)
        heat_colored = cv2.applyColorMap(heat_resized, cv2.COLORMAP_JET)

        # Mask out very low values so background remains visible
        mask = heat_resized > 15
        overlay = np.zeros_like(heat_colored)
        overlay[mask] = heat_colored[mask]

        return overlay

    def get_summary_stats(self, fps: float) -> Dict:
        """Computes comprehensive scientific summary dictionary."""
        total_time_s = len(self.history) / fps if fps > 0 else 0.0
        freezing_time_s = self.freezing_frames_total / fps if fps > 0 else 0.0
        freezing_pct = (freezing_time_s / total_time_s * 100.0) if total_time_s > 0 else 0.0
        mean_speed = (self.total_distance_cm / total_time_s) if total_time_s > 0 else 0.0

        # Zone breakdown with percentages
        zone_summary = {}
        for z_name, z_data in self.zone_stats.items():
            z_time = z_data["time_s"]
            z_pct = (z_time / total_time_s * 100.0) if total_time_s > 0 else 0.0
            z_mean_speed = (z_data["distance_cm"] / z_time) if z_time > 0 else 0.0
            zone_summary[z_name] = {
                "time_s": round(z_time, 2),
                "pct_time": round(z_pct, 1),
                "entries": z_data["entries"],
                "latency_s": round(z_data["latency_s"], 2) if z_data["latency_s"] is not None else 0.0,
                "distance_cm": round(z_data["distance_cm"], 2),
                "mean_speed_cm_s": round(z_mean_speed, 2)
            }

        # Custom free zones breakdown
        cz_summary = {}
        for cz_name, cz_data in self.custom_zone_stats.items():
            cz_time = cz_data["time_s"]
            cz_pct = (cz_time / total_time_s * 100.0) if total_time_s > 0 else 0.0
            cz_mean_speed = (cz_data["distance_cm"] / cz_time) if cz_time > 0 else 0.0
            cz_summary[cz_name] = {
                "time_s": round(cz_time, 2),
                "pct_time": round(cz_pct, 1),
                "entries": cz_data["entries"],
                "latency_s": round(cz_data["latency_s"], 2) if cz_data["latency_s"] is not None else 0.0,
                "distance_cm": round(cz_data["distance_cm"], 2),
                "mean_speed_cm_s": round(cz_mean_speed, 2)
            }

        return {
            "total_time_s": round(total_time_s, 2),
            "total_distance_cm": round(self.total_distance_cm, 2),
            "total_distance_m": round(self.total_distance_cm / 100.0, 3),
            "mean_speed_cm_s": round(mean_speed, 2),
            "max_speed_cm_s": round(self.max_speed_cm_s, 2),
            "freezing_time_s": round(freezing_time_s, 2),
            "freezing_pct": round(freezing_pct, 1),
            "pixels_per_cm": round(self.pixels_per_cm, 2),
            "zones": zone_summary,
            "custom_zones": cz_summary,
            "cells": self.cell_stats
        }
