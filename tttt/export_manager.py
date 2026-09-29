"""
export_manager.py - Data Export & Reporting Module
Exports research-grade data into standard formats:
1. Summary CSV (ready for SPSS, GraphPad Prism, R, Excel).
2. Raw Trajectory CSV (frame-by-frame coordinates, speeds, zones, freezing states).
3. High-resolution Trajectory Plot & Heatmap PNG images.
"""

from typing import Dict, List, Optional
import csv
import os
import cv2
import numpy as np


class ExportManager:
    @staticmethod
    def export_summary_csv(filepath: str, summary_data: Dict, arena_name: str) -> bool:
        """Exports overall experiment metrics and zone breakdown into clean CSV."""
        try:
            with open(filepath, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                # Header Metadata
                writer.writerow(["MazeTrack Scientific Summary Report"])
                writer.writerow(["Apparatus / Arena", arena_name])
                writer.writerow(["Scale (pixels/cm)", summary_data.get("pixels_per_cm", 0.0)])
                writer.writerow([])

                # Overall Metrics
                writer.writerow(["OVERALL METRICS", "VALUE", "UNIT"])
                writer.writerow(["Total Duration", summary_data.get("total_time_s", 0.0), "seconds"])
                writer.writerow(["Total Distance", summary_data.get("total_distance_cm", 0.0), "cm"])
                writer.writerow(["Total Distance", summary_data.get("total_distance_m", 0.0), "meters"])
                writer.writerow(["Mean Speed", summary_data.get("mean_speed_cm_s", 0.0), "cm/s"])
                writer.writerow(["Max Speed", summary_data.get("max_speed_cm_s", 0.0), "cm/s"])
                writer.writerow(["Freezing / Immobility Duration", summary_data.get("freezing_time_s", 0.0), "seconds"])
                writer.writerow(["Freezing / Immobility Percentage", summary_data.get("freezing_pct", 0.0), "%"])
                writer.writerow([])

                # Zone Breakdown Table
                writer.writerow(["ZONE METRICS"])
                writer.writerow([
                    "Zone / Category",
                    "Time (s)",
                    "Time (%)",
                    "Entries (count)",
                    "Latency to First Entry (s)",
                    "Distance (cm)",
                    "Mean Speed (cm/s)"
                ])

                zones = summary_data.get("zones", {})
                for z_name, z_vals in zones.items():
                    writer.writerow([
                        z_name,
                        z_vals.get("time_s", 0.0),
                        z_vals.get("pct_time", 0.0),
                        z_vals.get("entries", 0),
                        z_vals.get("latency_s", 0.0),
                        z_vals.get("distance_cm", 0.0),
                        z_vals.get("mean_speed_cm_s", 0.0)
                    ])

                # Custom Free Zones Breakdown (if available)
                custom_zones = summary_data.get("custom_zones", {})
                if custom_zones:
                    writer.writerow([])
                    writer.writerow(["CUSTOM FREE ZONES METRICS"])
                    writer.writerow([
                        "Zone Name",
                        "Time (s)",
                        "Time (%)",
                        "Entries (count)",
                        "Latency to First Entry (s)",
                        "Distance (cm)",
                        "Mean Speed (cm/s)"
                    ])
                    for cz_name, cz_vals in custom_zones.items():
                        writer.writerow([
                            cz_name,
                            cz_vals.get("time_s", 0.0),
                            cz_vals.get("pct_time", 0.0),
                            cz_vals.get("entries", 0),
                            cz_vals.get("latency_s", 0.0),
                            cz_vals.get("distance_cm", 0.0),
                            cz_vals.get("mean_speed_cm_s", 0.0)
                        ])

                # Grid Cell Specific Breakdown (if available)
                cells = summary_data.get("cells", {})
                if cells:
                    writer.writerow([])
                    writer.writerow(["GRID CELL BREAKDOWN"])
                    writer.writerow(["Cell ID", "Label", "Time (s)", "Entries"])
                    for cell_id, c_data in sorted(cells.items()):
                        writer.writerow([
                            cell_id,
                            c_data.get("label", ""),
                            round(c_data.get("time_s", 0.0), 2),
                            c_data.get("entries", 0)
                        ])

            return True
        except Exception as e:
            print(f"Error exporting summary CSV: {e}")
            return False

    @staticmethod
    def export_raw_trajectory_csv(filepath: str, history: List[Dict]) -> bool:
        """Exports frame-by-frame raw tracking data for detailed scientific modeling."""
        try:
            with open(filepath, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Frame",
                    "Time_s",
                    "X_px",
                    "Y_px",
                    "X_cm",
                    "Y_cm",
                    "Speed_cm_s",
                    "Zone",
                    "Cell_ID",
                    "Custom_Zones",
                    "Is_Freezing"
                ])

                for row in history:
                    cz_str = ";".join(row.get("custom_zones", []))
                    writer.writerow([
                        row.get("frame", 0),
                        round(row.get("time_s", 0.0), 3),
                        round(row.get("x_px", 0.0), 1),
                        round(row.get("y_px", 0.0), 1),
                        round(row.get("x_cm", 0.0), 2),
                        round(row.get("y_cm", 0.0), 2),
                        round(row.get("speed_cm_s", 0.0), 2),
                        row.get("zone", ""),
                        row.get("cell_id", ""),
                        cz_str,
                        1 if row.get("is_freezing", False) else 0
                    ])

            return True
        except Exception as e:
            print(f"Error exporting raw CSV: {e}")
            return False

    @staticmethod
    def export_trajectory_image(
        filepath: str,
        base_frame: np.ndarray,
        history: List[Dict],
        arena_obj=None,
        custom_zones_mgr=None
    ) -> bool:
        """Exports high-resolution trajectory plot saved to PNG."""
        try:
            img = base_frame.copy()

            # Draw arena overlay if available
            if arena_obj is not None and hasattr(arena_obj, "draw_overlay"):
                img = arena_obj.draw_overlay(img, alpha=0.18, show_text=True)

            # Draw custom zones if available
            if custom_zones_mgr is not None and hasattr(custom_zones_mgr, "draw_all"):
                img = custom_zones_mgr.draw_all(img, alpha=0.25, show_labels=True)

            # Draw full trajectory
            for i in range(1, len(history)):
                p1 = (int(history[i - 1]["x_px"]), int(history[i - 1]["y_px"]))
                p2 = (int(history[i]["x_px"]), int(history[i]["y_px"]))

                speed = history[i]["speed_cm_s"]
                ratio = min(1.0, speed / 25.0)
                b = int(255 * (1.0 - ratio))
                g = int(255 * (1.0 - abs(ratio - 0.5) * 2))
                r = int(255 * ratio)
                cv2.line(img, p1, p2, (b, g, r), 2, cv2.LINE_AA)

            # Mark start and end points
            if len(history) > 0:
                # Start: Green circle
                sp = (int(history[0]["x_px"]), int(history[0]["y_px"]))
                cv2.circle(img, sp, 8, (0, 255, 0), -1)
                cv2.circle(img, sp, 10, (0, 0, 0), 2)
                cv2.putText(img, "START", (sp[0] + 10, sp[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # End: Red circle
                ep = (int(history[-1]["x_px"]), int(history[-1]["y_px"]))
                cv2.circle(img, ep, 8, (0, 0, 255), -1)
                cv2.circle(img, ep, 10, (0, 0, 0), 2)
                cv2.putText(img, "END", (ep[0] + 10, ep[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.imwrite(filepath, img)
            return True
        except Exception as e:
            print(f"Error exporting trajectory image: {e}")
            return False

    @staticmethod
    def export_heatmap_image(
        filepath: str,
        base_frame: np.ndarray,
        heatmap_overlay: np.ndarray,
        alpha: float = 0.55
    ) -> bool:
        """Exports blended 2D occupancy heatmap saved to PNG."""
        try:
            h, w = base_frame.shape[:2]
            heat_resized = cv2.resize(heatmap_overlay, (w, h))

            # Blend where heatmap has values
            mask = cv2.cvtColor(heat_resized, cv2.COLOR_BGR2GRAY) > 10
            blended = base_frame.copy()
            blended[mask] = cv2.addWeighted(heat_resized, alpha, base_frame, 1 - alpha, 0)[mask]

            cv2.imwrite(filepath, blended)
            return True
        except Exception as e:
            print(f"Error exporting heatmap image: {e}")
            return False
