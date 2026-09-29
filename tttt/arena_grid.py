"""
arena_grid.py - Advanced Tracking Arena & Multi-Mode Grid System
Supports:
1. Interactive Manual Geometry: Rectangle (Kotak) & Ellipse (Elips).
2. Scale Calibration: Real-world dimension (cm) to calculate pixels_per_cm.
3. Multi-Mode Grid Division:
   - Square Grid (N x M) with Corner/Periphery/Center labeling & cell painting.
   - Concentric Circular Grid based on diameter (e.g. Center, Middle Ring, Periphery).
   - Radial Grid based on angle / sectors (e.g. 4 Quadrants, 6 Sectors, 8 Octants).
   - Polar Grid (Center circular zone + outer radial angular sectors).
   - Single Arena (Full Arena without grid division).
4. Point-in-zone queries, CV masking, and overlay visualization.
"""

from typing import Dict, List, Optional, Tuple
import math
import cv2
import numpy as np


class GridCell:
    def __init__(self, row: int, col: int, label: str = "Center"):
        self.row = row
        self.col = col
        self.cell_id = f"R{row + 1}C{col + 1}"
        self.label = label  # "Corner", "Periphery", "Center", "Custom"

    def to_dict(self):
        return {
            "row": self.row,
            "col": self.col,
            "cell_id": self.cell_id,
            "label": self.label,
        }


# Color mapping for zone labels (BGR for OpenCV, Hex for Qt)
LABEL_COLORS_BGR = {
    "Corner": (60, 60, 240),         # Bright Red
    "Periphery": (30, 180, 255),     # Orange / Gold
    "Center": (60, 220, 80),         # Bright Green
    "Middle Ring": (240, 160, 40),   # Amber
    "Outer Ring": (50, 100, 240),    # Red-Orange
    "Quadrant 1 (SE)": (60, 200, 100),  # Green
    "Quadrant 2 (SW)": (230, 180, 50),  # Cyan
    "Quadrant 3 (NW)": (50, 100, 240),  # Orange
    "Quadrant 4 (NE)": (200, 70, 200),  # Violet
    "Custom": (230, 80, 200),        # Magenta / Purple
    "Arena": (70, 180, 250),         # Sky Blue
    "Unassigned": (120, 120, 120)    # Grey
}

LABEL_COLORS_HEX = {
    "Corner": "#f43f5e",
    "Periphery": "#f59e0b",
    "Center": "#10b981",
    "Middle Ring": "#fbbf24",
    "Outer Ring": "#f97316",
    "Quadrant 1 (SE)": "#22c55e",
    "Quadrant 2 (SW)": "#06b6d4",
    "Quadrant 3 (NW)": "#f97316",
    "Quadrant 4 (NE)": "#a855f7",
    "Custom": "#d946ef",
    "Arena": "#38bdf8",
    "Unassigned": "#64748b"
}

# Distinct palette for dynamic sectors
SECTOR_PALETTE_BGR = [
    (60, 200, 100),   # Green
    (230, 180, 50),   # Cyan
    (50, 100, 240),   # Orange
    (200, 70, 200),   # Violet
    (50, 220, 240),   # Yellow
    (220, 100, 60),   # Blue
    (180, 50, 220),   # Magenta
    (70, 190, 150),   # Mint
    (240, 120, 90),   # Soft Blue
    (120, 210, 70),   # Lime
    (210, 80, 140),   # Purple-Blue
    (60, 150, 240)    # Amber-Orange
]


class TrackingArena:
    """
    Unified Tracking Arena supporting Rectangle and Ellipse shapes,
    scale calibration (panjang riil arena in cm), and multi-mode grid divisions:
    - 'square': N x M rows and columns
    - 'concentric': diameter-based concentric circular rings
    - 'radial': angular sectors (quadrants, octants, custom sectors)
    - 'polar': center circle + outer radial sectors
    - 'none': whole arena as single zone
    """
    def __init__(
        self,
        shape_type: str = "rectangle",  # 'rectangle' or 'ellipse'
        grid_type: str = "square",      # 'square', 'concentric', 'radial', 'polar', 'none'
        real_length_cm: float = 50.0,
        rows: int = 4,
        cols: int = 4
    ):
        self.shape_type = shape_type
        self.grid_type = grid_type
        self.real_length_cm = max(1.0, float(real_length_cm))

        # False initially: user must draw the arena first!
        self.is_defined = False

        # Normalized bounding box inside the frame [x_norm, y_norm, w_norm, h_norm] (0.0 - 1.0)
        self.norm_bbox = [0.15, 0.10, 0.70, 0.80]

        # 1. Square Grid state
        self.rows = max(2, rows)
        self.cols = max(2, cols)
        self.cells: List[List[GridCell]] = []
        self.init_cells()
        self.auto_label_grid()

        # 2. Concentric Rings state (fractions of radius/diameter: 0.0 to 1.0)
        # Default: 3 rings: Center (inner 35% diam), Middle Ring (70% diam), Periphery (outer 100%)
        self.concentric_rings: List[Tuple[float, str]] = [
            (0.35, "Center"),
            (0.70, "Middle Ring"),
            (1.00, "Periphery")
        ]

        # 3. Radial Sectors state
        self.radial_sector_count: int = 4  # 4 Quadrants, 6, 8, etc.
        self.radial_angle_offset_deg: float = 0.0  # Rotation offset in degrees

        # 4. Polar Grid state
        self.polar_center_fraction: float = 0.35
        self.polar_sector_count: int = 4

    # -------------------------------------------------------------
    # Geometry & Bounding Box Setup
    # -------------------------------------------------------------
    def set_shape_type(self, shape_type: str):
        if shape_type in ("rectangle", "ellipse"):
            self.shape_type = shape_type

    def set_grid_type(self, grid_type: str):
        if grid_type in ("square", "concentric", "radial", "polar", "none"):
            self.grid_type = grid_type

    def set_bbox_pixels(self, x: int, y: int, w: int, h: int, frame_w: int, frame_h: int):
        """Sets bounding box from pixel coordinates and marks arena as defined."""
        if frame_w <= 0 or frame_h <= 0:
            return
        x = max(0, min(x, frame_w - 10))
        y = max(0, min(y, frame_h - 10))
        w = max(10, min(w, frame_w - x))
        h = max(10, min(h, frame_h - y))
        self.norm_bbox = [x / frame_w, y / frame_h, w / frame_w, h / frame_h]
        self.is_defined = True

    def get_bbox_pixels(self, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
        """Returns (x, y, w, h) in pixel coordinates."""
        nx, ny, nw, nh = self.norm_bbox
        return int(nx * frame_w), int(ny * frame_h), int(nw * frame_w), int(nh * frame_h)

    def get_geometry_params(self, frame_w: int, frame_h: int):
        """
        Returns:
            bx, by, bw, bh: Bounding box in pixels
            cx, cy: Center point in pixels
            rx, ry: Semi-axes (radii) in pixels (bw / 2, bh / 2)
        """
        bx, by, bw, bh = self.get_bbox_pixels(frame_w, frame_h)
        cx = bx + bw / 2.0
        cy = by + bh / 2.0
        rx = max(1.0, bw / 2.0)
        ry = max(1.0, bh / 2.0)
        return bx, by, bw, bh, cx, cy, rx, ry

    def calculate_pixels_per_cm(self, frame_w: int, frame_h: int) -> float:
        """
        Calculates pixels_per_cm based on arena pixel width/diameter vs real_length_cm.
        """
        _, _, bw, _ = self.get_bbox_pixels(frame_w, frame_h)
        if self.real_length_cm > 0:
            return max(0.01, bw / self.real_length_cm)
        return 5.0

    def is_point_inside(self, px: float, py: float, frame_w: int, frame_h: int) -> bool:
        """Checks whether pixel (px, py) is strictly inside the arena boundary."""
        if not self.is_defined:
            return True  # If no arena defined, treat entire frame as active
        bx, by, bw, bh, cx, cy, rx, ry = self.get_geometry_params(frame_w, frame_h)

        if self.shape_type == "rectangle":
            return (bx <= px <= bx + bw) and (by <= py <= by + bh)
        elif self.shape_type == "ellipse":
            dx = (px - cx) / rx
            dy = (py - cy) / ry
            return (dx * dx + dy * dy) <= 1.0
        return False

    def get_mask(self, frame_w: int, frame_h: int) -> np.ndarray:
        """Returns a binary uint8 mask (255 inside arena, 0 outside) for CV filtering."""
        mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        if not self.is_defined:
            mask.fill(255)
            return mask

        bx, by, bw, bh, cx, cy, rx, ry = self.get_geometry_params(frame_w, frame_h)
        if self.shape_type == "rectangle":
            cv2.rectangle(mask, (bx, by), (bx + bw, by + bh), 255, -1)
        elif self.shape_type == "ellipse":
            cv2.ellipse(mask, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, 255, -1)
        return mask

    # -------------------------------------------------------------
    # Square Grid Methods
    # -------------------------------------------------------------
    def init_cells(self):
        """Initializes empty square grid cells matrix."""
        self.cells = []
        for r in range(self.rows):
            row_cells = []
            for c in range(self.cols):
                row_cells.append(GridCell(r, c, "Center"))
            self.cells.append(row_cells)

    def set_dimensions(self, rows: int, cols: int):
        """Changes square grid resolution (e.g. 3x3, 4x4, 5x5) and auto-labels."""
        self.rows = max(2, int(rows))
        self.cols = max(2, int(cols))
        self.init_cells()
        self.auto_label_grid()

    def auto_label_grid(self):
        """Standard Open Field auto-labeling (Corner, Periphery, Center)."""
        for r in range(self.rows):
            for c in range(self.cols):
                is_top = (r == 0)
                is_bottom = (r == self.rows - 1)
                is_left = (c == 0)
                is_right = (c == self.cols - 1)

                if (is_top and is_left) or (is_top and is_right) or \
                   (is_bottom and is_left) or (is_bottom and is_right):
                    self.cells[r][c].label = "Corner"
                elif is_top or is_bottom or is_left or is_right:
                    self.cells[r][c].label = "Periphery"
                else:
                    self.cells[r][c].label = "Center"

    def set_cell_label(self, row: int, col: int, label: str):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.cells[row][col].label = label

    def toggle_cell_label(self, row: int, col: int) -> str:
        labels = ["Center", "Periphery", "Corner", "Custom"]
        if 0 <= row < self.rows and 0 <= col < self.cols:
            curr = self.cells[row][col].label
            next_idx = (labels.index(curr) + 1) % len(labels) if curr in labels else 0
            self.cells[row][col].label = labels[next_idx]
            return self.cells[row][col].label
        return "Unknown"

    def get_cell_at_pixel(self, px: float, py: float, frame_w: int, frame_h: int) -> Optional[Tuple[int, int, GridCell]]:
        """Compatible with legacy open field: returns (row, col, GridCell) or None."""
        if not self.is_point_inside(px, py, frame_w, frame_h):
            return None

        bx, by, bw, bh = self.get_bbox_pixels(frame_w, frame_h)
        cell_w = bw / self.cols
        cell_h = bh / self.rows

        col = int((px - bx) // cell_w)
        row = int((py - by) // cell_h)

        col = max(0, min(col, self.cols - 1))
        row = max(0, min(row, self.rows - 1))

        return row, col, self.cells[row][col]

    # -------------------------------------------------------------
    # Concentric Rings Setup
    # -------------------------------------------------------------
    def set_concentric_ring_count(self, count: int):
        """Sets number of concentric rings (2, 3, or 4)."""
        count = max(2, min(5, int(count)))
        if count == 2:
            self.concentric_rings = [
                (0.50, "Center"),
                (1.00, "Periphery")
            ]
        elif count == 3:
            self.concentric_rings = [
                (0.35, "Center"),
                (0.70, "Middle Ring"),
                (1.00, "Periphery")
            ]
        elif count == 4:
            self.concentric_rings = [
                (0.25, "Center"),
                (0.50, "Inner Ring"),
                (0.75, "Middle Ring"),
                (1.00, "Periphery")
            ]

    # -------------------------------------------------------------
    # Radial Sectors Setup
    # -------------------------------------------------------------
    def set_radial_sector_count(self, count: int):
        """Sets number of radial sectors (e.g. 4 quadrants, 6 sectors, 8 octants)."""
        self.radial_sector_count = max(2, min(16, int(count)))

    # -------------------------------------------------------------
    # Universal Zone Query: (zone_label, cell_or_sector_id)
    # -------------------------------------------------------------
    def get_zone_at_pixel(self, px: float, py: float, frame_w: int, frame_h: int) -> Optional[Tuple[str, str]]:
        """
        Determines which zone / cell / sector pixel (px, py) belongs to.
        Returns:
            (zone_label, cell_id) or None if outside arena.
        """
        if not self.is_point_inside(px, py, frame_w, frame_h):
            return None

        bx, by, bw, bh, cx, cy, rx, ry = self.get_geometry_params(frame_w, frame_h)

        # 1. NONE: Single Arena
        if self.grid_type == "none":
            return "Arena", "Arena"

        # 2. SQUARE GRID
        elif self.grid_type == "square":
            cell_w = bw / self.cols
            cell_h = bh / self.rows
            col = max(0, min(int((px - bx) // cell_w), self.cols - 1))
            row = max(0, min(int((py - by) // cell_h), self.rows - 1))
            cell = self.cells[row][col]
            return cell.label, cell.cell_id

        # 3. CONCENTRIC RINGS (Lingkaran Berbasis Diameter)
        elif self.grid_type == "concentric":
            # Normalized radial distance [0.0 - 1.0]
            dx = (px - cx) / rx
            dy = (py - cy) / ry
            r_norm = math.hypot(dx, dy)
            for frac, name in sorted(self.concentric_rings, key=lambda x: x[0]):
                if r_norm <= frac:
                    return name, name
            # Fallback to outermost
            return self.concentric_rings[-1][1], self.concentric_rings[-1][1]

        # 4. RADIAL SECTORS (Grid Radial Berbasis Sudut)
        elif self.grid_type == "radial":
            # Angle in degrees [0 - 360) clockwise starting from 3 o'clock
            angle = math.degrees(math.atan2(py - cy, px - cx))
            angle = (angle - self.radial_angle_offset_deg) % 360.0

            sector_size = 360.0 / self.radial_sector_count
            sec_idx = int(angle // sector_size) % self.radial_sector_count

            if self.radial_sector_count == 4:
                # Standard quadrants names: Q1 (0-90° SE), Q2 (90-180° SW), Q3 (180-270° NW), Q4 (270-360° NE)
                quad_names = ["Quadrant 1 (SE)", "Quadrant 2 (SW)", "Quadrant 3 (NW)", "Quadrant 4 (NE)"]
                name = quad_names[sec_idx]
                return name, f"Q{sec_idx + 1}"
            else:
                name = f"Sektor {sec_idx + 1} ({int(sec_idx * sector_size)}°-{int((sec_idx + 1) * sector_size)}°)"
                return name, f"S{sec_idx + 1}"

        # 5. POLAR GRID (Center Circle + Outer Radial Sectors)
        elif self.grid_type == "polar":
            dx = (px - cx) / rx
            dy = (py - cy) / ry
            r_norm = math.hypot(dx, dy)
            if r_norm <= self.polar_center_fraction:
                return "Center", "Center"

            angle = math.degrees(math.atan2(py - cy, px - cx)) % 360.0
            sector_size = 360.0 / self.polar_sector_count
            sec_idx = int(angle // sector_size) % self.polar_sector_count
            return f"Outer Sektor {sec_idx + 1}", f"P{sec_idx + 1}"

        return "Arena", "Arena"

    # -------------------------------------------------------------
    # Get All Active Categories / Labels
    # -------------------------------------------------------------
    def get_all_categories(self) -> List[str]:
        """Returns ordered list of all active categories in current arena mode."""
        if self.grid_type == "none":
            return ["Arena"]

        elif self.grid_type == "square":
            cats = set()
            for r in range(self.rows):
                for c in range(self.cols):
                    cats.add(self.cells[r][c].label)
            order = ["Corner", "Periphery", "Center", "Custom"]
            return [cat for cat in order if cat in cats] + [cat for cat in cats if cat not in order]

        elif self.grid_type == "concentric":
            return [name for _, name in sorted(self.concentric_rings, key=lambda x: x[0])]

        elif self.grid_type == "radial":
            if self.radial_sector_count == 4:
                return ["Quadrant 1 (SE)", "Quadrant 2 (SW)", "Quadrant 3 (NW)", "Quadrant 4 (NE)"]
            sector_size = 360.0 / self.radial_sector_count
            return [f"Sektor {i + 1} ({int(i * sector_size)}°-{int((i + 1) * sector_size)}°)" for i in range(self.radial_sector_count)]

        elif self.grid_type == "polar":
            res = ["Center"]
            res.extend([f"Outer Sektor {i + 1}" for i in range(self.polar_sector_count)])
            return res

        return ["Arena"]

    # -------------------------------------------------------------
    # Overlay Visualization (OpenCV)
    # -------------------------------------------------------------
    def draw_overlay(self, frame_bgr: np.ndarray, alpha: float = 0.25, show_text: bool = True) -> np.ndarray:
        """
        Renders rich translucent grid cells/rings/sectors, crisp outer borders,
        and clean labels onto the frame.
        """
        if not self.is_defined:
            return frame_bgr

        h, w = frame_bgr.shape[:2]
        bx, by, bw, bh, cx, cy, rx, ry = self.get_geometry_params(w, h)
        icx, icy, irx, iry = int(cx), int(cy), int(rx), int(ry)

        overlay = frame_bgr.copy()

        # ---------------------------------------------------------
        # A. SQUARE GRID
        # ---------------------------------------------------------
        if self.grid_type == "square":
            cell_w = bw / self.cols
            cell_h = bh / self.rows

            # Fill colored cells
            for r in range(self.rows):
                for c in range(self.cols):
                    cell = self.cells[r][c]
                    color = LABEL_COLORS_BGR.get(cell.label, (120, 120, 120))
                    cx1 = int(bx + c * cell_w)
                    cy1 = int(by + r * cell_h)
                    cx2 = int(bx + (c + 1) * cell_w)
                    cy2 = int(by + (r + 1) * cell_h)
                    cv2.rectangle(overlay, (cx1, cy1), (cx2, cy2), color, -1)

            # If ellipse shape with square grid, mask the overlay to the ellipse
            if self.shape_type == "ellipse":
                mask_ell = np.zeros((h, w), dtype=np.uint8)
                cv2.ellipse(mask_ell, (icx, icy), (irx, iry), 0, 0, 360, 255, -1)
                overlay = cv2.bitwise_and(overlay, overlay, mask=mask_ell)
                blended = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)
                # Outer Ellipse
                cv2.ellipse(blended, (icx, icy), (irx, iry), 0, 0, 360, (255, 255, 255), 2)
            else:
                blended = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)
                # Outer Rectangle
                cv2.rectangle(blended, (bx, by), (bx + bw, by + bh), (255, 255, 255), 2)

            # Draw interior grid lines
            for r in range(1, self.rows):
                ly = int(by + r * cell_h)
                cv2.line(blended, (bx, ly), (bx + bw, ly), (200, 200, 200), 1)
            for c in range(1, self.cols):
                lx = int(bx + c * cell_w)
                cv2.line(blended, (lx, by), (lx, by + bh), (200, 200, 200), 1)

            # Text labels
            if show_text:
                font_scale = max(0.32, min(0.60, min(cell_w, cell_h) / 110.0))
                for r in range(self.rows):
                    for c in range(self.cols):
                        cell = self.cells[r][c]
                        cx1 = int(bx + c * cell_w)
                        cy1 = int(by + r * cell_h)
                        cx2 = int(bx + (c + 1) * cell_w)
                        cy2 = int(by + (r + 1) * cell_h)
                        label_text = cell.label if min(cell_w, cell_h) >= 60 else cell.label[:3].upper()

                        mid_x = (cx1 + cx2) // 2
                        mid_y = (cy1 + cy2) // 2

                        # Cell ID
                        cv2.putText(blended, cell.cell_id, (cx1 + 4, cy1 + 13),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.75, (255, 255, 255), 1, cv2.LINE_AA)
                        # Label centered
                        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                        cv2.putText(blended, label_text, (mid_x - tw // 2, mid_y + th // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
            return blended

        # ---------------------------------------------------------
        # B. CONCENTRIC RINGS (Diameter-based Circles)
        # ---------------------------------------------------------
        elif self.grid_type == "concentric":
            sorted_rings = sorted(self.concentric_rings, key=lambda x: x[0], reverse=True)
            for frac, name in sorted_rings:
                c_rx = max(2, int(rx * frac))
                c_ry = max(2, int(ry * frac))
                color = LABEL_COLORS_BGR.get(name, (100, 180, 240))
                if self.shape_type == "ellipse":
                    cv2.ellipse(overlay, (icx, icy), (c_rx, c_ry), 0, 0, 360, color, -1)
                else:
                    # In rectangle arena, draw concentric centered boxes
                    cw = int(bw * frac)
                    ch = int(bh * frac)
                    cv2.rectangle(overlay, (icx - cw // 2, icy - ch // 2), (icx + cw // 2, icy + ch // 2), color, -1)

            blended = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)

            # Draw crisp outlines for each ring
            for frac, name in self.concentric_rings:
                c_rx = max(2, int(rx * frac))
                c_ry = max(2, int(ry * frac))
                if self.shape_type == "ellipse":
                    cv2.ellipse(blended, (icx, icy), (c_rx, c_ry), 0, 0, 360, (255, 255, 255), 2 if frac == 1.0 else 1)
                else:
                    cw = int(bw * frac)
                    ch = int(bh * frac)
                    cv2.rectangle(blended, (icx - cw // 2, icy - ch // 2), (icx + cw // 2, icy + ch // 2),
                                  (255, 255, 255), 2 if frac == 1.0 else 1)

            if show_text:
                for frac, name in sorted(self.concentric_rings, key=lambda x: x[0]):
                    pos_y = icy - int(ry * frac * 0.7)
                    (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.putText(blended, name, (icx - tw // 2, pos_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            return blended

        # ---------------------------------------------------------
        # C. RADIAL SECTORS (Angular division based on degrees)
        # ---------------------------------------------------------
        elif self.grid_type == "radial":
            N = self.radial_sector_count
            sector_size = 360.0 / N
            offset = self.radial_angle_offset_deg

            for i in range(N):
                ang_start = offset + i * sector_size
                ang_end = offset + (i + 1) * sector_size
                color = SECTOR_PALETTE_BGR[i % len(SECTOR_PALETTE_BGR)]
                cv2.ellipse(overlay, (icx, icy), (irx, iry), 0, ang_start, ang_end, color, -1)

            # If rectangle shape with radial sectors, clip to bounding box
            if self.shape_type == "rectangle":
                mask_rect = np.zeros((h, w), dtype=np.uint8)
                cv2.rectangle(mask_rect, (bx, by), (bx + bw, by + bh), 255, -1)
                overlay = cv2.bitwise_and(overlay, overlay, mask=mask_rect)

            blended = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)

            # Draw outer boundary
            if self.shape_type == "ellipse":
                cv2.ellipse(blended, (icx, icy), (irx, iry), 0, 0, 360, (255, 255, 255), 2)
            else:
                cv2.rectangle(blended, (bx, by), (bx + bw, by + bh), (255, 255, 255), 2)

            # Draw spoke lines from center
            for i in range(N):
                rad = math.radians(offset + i * sector_size)
                ex = int(icx + irx * math.cos(rad))
                ey = int(icy + iry * math.sin(rad))
                cv2.line(blended, (icx, icy), (ex, ey), (255, 255, 255), 1)

            # Text labels in middle of each sector
            if show_text:
                for i in range(N):
                    mid_ang = math.radians(offset + (i + 0.5) * sector_size)
                    tx = int(icx + 0.65 * irx * math.cos(mid_ang))
                    ty = int(icy + 0.65 * iry * math.sin(mid_ang))

                    if N == 4:
                        lbl = f"Q{i + 1}"
                    else:
                        lbl = f"S{i + 1}"

                    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.putText(blended, lbl, (tx - tw // 2, ty + th // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            return blended

        # ---------------------------------------------------------
        # D. POLAR GRID (Center Circle + Outer Radial Sectors)
        # ---------------------------------------------------------
        elif self.grid_type == "polar":
            N = self.polar_sector_count
            sector_size = 360.0 / N

            # 1. Outer radial sectors
            for i in range(N):
                color = SECTOR_PALETTE_BGR[i % len(SECTOR_PALETTE_BGR)]
                cv2.ellipse(overlay, (icx, icy), (irx, iry), 0, i * sector_size, (i + 1) * sector_size, color, -1)

            # 2. Inner center circle
            c_rx = int(irx * self.polar_center_fraction)
            c_ry = int(iry * self.polar_center_fraction)
            cv2.ellipse(overlay, (icx, icy), (c_rx, c_ry), 0, 0, 360, LABEL_COLORS_BGR["Center"], -1)

            blended = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)

            # Outer border & inner circle
            cv2.ellipse(blended, (icx, icy), (irx, iry), 0, 0, 360, (255, 255, 255), 2)
            cv2.ellipse(blended, (icx, icy), (c_rx, c_ry), 0, 0, 360, (255, 255, 255), 1)

            # Spokes outside inner circle
            for i in range(N):
                rad = math.radians(i * sector_size)
                sx = int(icx + c_rx * math.cos(rad))
                sy = int(icy + c_ry * math.sin(rad))
                ex = int(icx + irx * math.cos(rad))
                ey = int(icy + iry * math.sin(rad))
                cv2.line(blended, (sx, sy), (ex, ey), (255, 255, 255), 1)

            if show_text:
                cv2.putText(blended, "Center", (icx - 18, icy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
                for i in range(N):
                    mid_ang = math.radians((i + 0.5) * sector_size)
                    mid_r = (c_rx + irx) / 2.0
                    tx = int(icx + mid_r * math.cos(mid_ang))
                    ty = int(icy + mid_r * math.sin(mid_ang))
                    cv2.putText(blended, f"P{i + 1}", (tx - 8, ty + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
            return blended

        # ---------------------------------------------------------
        # E. NONE (Single Arena Zone)
        # ---------------------------------------------------------
        else:
            color = LABEL_COLORS_BGR.get("Arena", (70, 180, 250))
            if self.shape_type == "ellipse":
                cv2.ellipse(overlay, (icx, icy), (irx, iry), 0, 0, 360, color, -1)
                blended = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)
                cv2.ellipse(blended, (icx, icy), (irx, iry), 0, 0, 360, (255, 255, 255), 2)
            else:
                cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), color, -1)
                blended = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)
                cv2.rectangle(blended, (bx, by), (bx + bw, by + bh), (255, 255, 255), 2)
            return blended


class OpenFieldGridArena(TrackingArena):
    """
    Backward-compatible subclass for legacy Open Field Test code.
    Maintains exact constructor signature and pre-defined bounding box.
    """
    def __init__(self, rows: int = 4, cols: int = 4):
        super().__init__(shape_type="rectangle", grid_type="square", rows=rows, cols=cols)
        self.is_defined = True
        self.norm_bbox = [0.15, 0.10, 0.70, 0.80]
        self.init_cells()
        self.auto_label_grid()
