"""
test_core.py - Automated Verification Suite for MazeTrack Desktop
Tests Grid logic, Apparatus Presets, Simulation, Tracking Engine, and Export Manager.
"""

import os
import shutil
import tempfile
import numpy as np

from arena_grid import OpenFieldGridArena
from arena_presets import MorrisWaterMazeArena, ElevatedPlusMazeArena
from simulation import RodentSimulation
from tracker_engine import AnimalTracker
from export_manager import ExportManager


def test_grid_arena():
    print("Testing OpenFieldGridArena...")
    arena = OpenFieldGridArena(rows=4, cols=4)
    arena.set_bbox_pixels(100, 100, 400, 400, 600, 600)

    # Check 4 corners
    assert arena.cells[0][0].label == "Corner", f"Expected Corner, got {arena.cells[0][0].label}"
    assert arena.cells[0][3].label == "Corner", f"Expected Corner, got {arena.cells[0][3].label}"
    assert arena.cells[3][0].label == "Corner", f"Expected Corner, got {arena.cells[3][0].label}"
    assert arena.cells[3][3].label == "Corner", f"Expected Corner, got {arena.cells[3][3].label}"

    # Check periphery and center
    assert arena.cells[0][1].label == "Periphery", f"Expected Periphery, got {arena.cells[0][1].label}"
    assert arena.cells[1][1].label == "Center", f"Expected Center, got {arena.cells[1][1].label}"

    # Check point in cell
    # Cell (0,0) is [100..200, 100..200]
    res = arena.get_cell_at_pixel(150, 150, 600, 600)
    assert res is not None
    r, c, cell = res
    assert r == 0 and c == 0 and cell.label == "Corner"

    # Test overlay drawing
    dummy_frame = np.full((600, 600, 3), 200, dtype=np.uint8)
    overlay = arena.draw_overlay(dummy_frame)
    assert overlay.shape == dummy_frame.shape
    print("[PASS] OpenFieldGridArena passed!")


def test_presets():
    print("Testing MorrisWaterMaze & EPM...")
    mwm = MorrisWaterMazeArena()
    # Center is 300, 300 in 600x600
    res_center = mwm.get_zone_at_pixel(300, 300, 600, 600)
    assert res_center is not None, "Center should be inside pool"

    epm = ElevatedPlusMazeArena()
    res_epm = epm.get_zone_at_pixel(300, 300, 600, 600)
    assert res_epm is not None and res_epm[0] == "Center Area"
    print("[PASS] Apparatus presets passed!")


def test_simulation_and_tracking():
    print("Testing Simulation & Tracker Engine...")
    sim = RodentSimulation(width=640, height=480, fps=30)
    tracker = AnimalTracker()
    tracker.threshold_val = 100
    tracker.is_dark_animal = True

    arena = OpenFieldGridArena(rows=4, cols=4)
    arena.set_bbox_pixels(sim.margin_x, sim.margin_y, sim.arena_w, sim.arena_h, 640, 480)

    # Run 30 frames of simulation
    detected_count = 0
    for f in range(30):
        frame = sim.get_next_frame()
        assert frame.shape == (480, 640, 3)
        vis, det, mask = tracker.process_frame(frame, f + 1, fps=30.0, arena_obj=arena, active_arena_type="Open Field Grid")
        if det is not None:
            detected_count += 1

    assert detected_count >= 25, f"Expected rodent detected in >=25/30 frames, got {detected_count}"
    assert tracker.total_distance_cm > 0.0, "Total distance should be > 0"

    summary = tracker.get_summary_stats(fps=30.0)
    assert "zones" in summary
    print(f"[PASS] Simulation & Tracker passed! Detected in {detected_count}/30 frames, distance: {tracker.total_distance_cm:.1f} cm")


def test_exports():
    print("Testing ExportManager...")
    sim = RodentSimulation(width=640, height=480, fps=30)
    tracker = AnimalTracker()
    arena = OpenFieldGridArena(rows=4, cols=4)
    arena.set_bbox_pixels(sim.margin_x, sim.margin_y, sim.arena_w, sim.arena_h, 640, 480)

    last_frame = None
    for f in range(20):
        last_frame = sim.get_next_frame()
        tracker.process_frame(last_frame, f + 1, fps=30.0, arena_obj=arena, active_arena_type="Open Field Grid")

    temp_dir = tempfile.mkdtemp()
    try:
        summary_csv = os.path.join(temp_dir, "summary.csv")
        raw_csv = os.path.join(temp_dir, "raw.csv")
        traj_png = os.path.join(temp_dir, "traj.png")
        heat_png = os.path.join(temp_dir, "heat.png")

        stats = tracker.get_summary_stats(fps=30.0)
        assert ExportManager.export_summary_csv(summary_csv, stats, "Open Field Grid")
        assert os.path.exists(summary_csv) and os.path.getsize(summary_csv) > 100

        assert ExportManager.export_raw_trajectory_csv(raw_csv, tracker.history)
        assert os.path.exists(raw_csv) and os.path.getsize(raw_csv) > 100

        assert ExportManager.export_trajectory_image(traj_png, last_frame, tracker.history, arena)
        assert os.path.exists(traj_png) and os.path.getsize(traj_png) > 1000

        heat_overlay = tracker.generate_heatmap_overlay(640, 480)
        assert ExportManager.export_heatmap_image(heat_png, last_frame, heat_overlay)
        assert os.path.exists(heat_png) and os.path.getsize(heat_png) > 1000
        print("[PASS] All exports (CSV, PNG) generated successfully!")
    finally:
        shutil.rmtree(temp_dir)

def test_custom_zones():
    print("Testing CustomZoneManager...")
    from custom_zones import CustomZoneManager
    czm = CustomZoneManager()
    z1 = czm.add_zone("Novel Object A", "rectangle", [(0.2, 0.2), (0.4, 0.4)], "#ec4899")
    z2 = czm.add_zone("Shelter", "circle", [(0.7, 0.7), (0.8, 0.7)], "#8b5cf6")
    z3 = czm.add_zone("Custom Poly", "polygon", [(0.5, 0.5), (0.6, 0.5), (0.55, 0.65)], "#14b8a6")

    assert len(czm.zones) == 3
    # Test point inside rectangle z1 (0.3, 0.3) in 600x600 -> (180, 180)
    matching = czm.get_zones_containing_point(180, 180, 600, 600)
    assert len(matching) == 1 and matching[0].name == "Novel Object A"

    # Test point outside
    matching_out = czm.get_zones_containing_point(10, 10, 600, 600)
    assert len(matching_out) == 0

    dummy_frame = np.full((480, 640, 3), 220, dtype=np.uint8)
    drawn = czm.draw_all(dummy_frame)
    assert drawn.shape == dummy_frame.shape
    print("[PASS] CustomZoneManager passed!")


def test_camera_manager():
    print("Testing CameraManager...")
    from camera_manager import CameraManager
    cams = CameraManager.list_cameras()
    print(f"[INFO] Discovered cameras: {cams}")
    assert isinstance(cams, list)
    print("[PASS] CameraManager passed!")


def test_tracking_arena_shapes_and_grids():
    print("Testing TrackingArena (Kotak, Elips, Concentric, Radial, Polar)...")
    from arena_grid import TrackingArena

    # 1. Rectangle Arena with Square Grid
    rect_arena = TrackingArena(shape_type="rectangle", grid_type="square", real_length_cm=50.0, rows=4, cols=4)
    assert not rect_arena.is_defined, "Initially arena should not be defined"
    rect_arena.set_bbox_pixels(100, 100, 400, 400, 600, 600)
    assert rect_arena.is_defined, "Arena should be defined after setting bbox"

    # Scale calculation
    px_per_cm = rect_arena.calculate_pixels_per_cm(600, 600)
    assert abs(px_per_cm - 8.0) < 1e-4, f"Expected 8.0 px/cm, got {px_per_cm}"

    # Inside and outside queries
    assert rect_arena.is_point_inside(200, 200, 600, 600)
    assert not rect_arena.is_point_inside(50, 50, 600, 600)

    # 2. Ellipse Arena with Concentric Diameter Rings
    ell_arena = TrackingArena(shape_type="ellipse", grid_type="concentric", real_length_cm=100.0)
    ell_arena.set_bbox_pixels(100, 100, 400, 400, 600, 600)
    # Center is at (300, 300) with rx = 200, ry = 200
    assert ell_arena.is_point_inside(300, 300, 600, 600), "Center should be inside ellipse"
    # Corner of bounding box (105, 105) is outside the circle (distance ~ 275 > 200)
    assert not ell_arena.is_point_inside(105, 105, 600, 600), "Corner of bbox should be outside ellipse"

    # Concentric rings point test
    # Center (at 300, 300): distance = 0 <= 0.35 -> "Center"
    z_center = ell_arena.get_zone_at_pixel(300, 300, 600, 600)
    assert z_center is not None and z_center[0] == "Center", f"Expected Center, got {z_center}"

    # Middle ring (at 300, 400): distance = 100 / 200 = 0.50 -> "Middle Ring"
    z_mid = ell_arena.get_zone_at_pixel(300, 400, 600, 600)
    assert z_mid is not None and z_mid[0] == "Middle Ring", f"Expected Middle Ring, got {z_mid}"

    # Periphery (at 300, 480): distance = 180 / 200 = 0.90 -> "Periphery"
    z_peri = ell_arena.get_zone_at_pixel(300, 480, 600, 600)
    assert z_peri is not None and z_peri[0] == "Periphery", f"Expected Periphery, got {z_peri}"

    # 3. Radial Grid (Angular Sectors)
    rad_arena = TrackingArena(shape_type="ellipse", grid_type="radial", real_length_cm=60.0)
    rad_arena.set_bbox_pixels(100, 100, 400, 400, 600, 600)
    rad_arena.set_radial_sector_count(4)

    # Q1: dx > 0, dy > 0 (SE) -> angle between 0 and 90
    z_q1 = rad_arena.get_zone_at_pixel(350, 350, 600, 600)
    assert z_q1 is not None and "Quadrant 1" in z_q1[0], f"Expected Q1, got {z_q1}"

    # Q2: dx < 0, dy > 0 (SW) -> angle between 90 and 180
    z_q2 = rad_arena.get_zone_at_pixel(250, 350, 600, 600)
    assert z_q2 is not None and "Quadrant 2" in z_q2[0], f"Expected Q2, got {z_q2}"

    # 4. Polar Grid (Center + Outer Sectors)
    pol_arena = TrackingArena(shape_type="ellipse", grid_type="polar", real_length_cm=60.0)
    pol_arena.set_bbox_pixels(100, 100, 400, 400, 600, 600)
    z_pol_center = pol_arena.get_zone_at_pixel(300, 300, 600, 600)
    assert z_pol_center is not None and z_pol_center[0] == "Center"

    # 5. Tracking Simulation with TrackingArena
    sim = RodentSimulation(width=640, height=480, fps=30)
    tracker = AnimalTracker()
    test_arena = TrackingArena(shape_type="rectangle", grid_type="concentric", real_length_cm=50.0)
    test_arena.set_bbox_pixels(sim.margin_x, sim.margin_y, sim.arena_w, sim.arena_h, 640, 480)

    for f in range(25):
        frame = sim.get_next_frame()
        vis, det, mask = tracker.process_frame(frame, f + 1, fps=30.0, arena_obj=test_arena, active_arena_type="Tracking Arena")

    summary = tracker.get_summary_stats(fps=30.0)
    assert len(summary.get("zones", {})) > 0, "Zones should be recorded in summary stats"
    print("[PASS] TrackingArena shapes & multi-mode grids passed successfully!")


if __name__ == "__main__":
    test_grid_arena()
    test_tracking_arena_shapes_and_grids()
    test_presets()
    test_custom_zones()
    test_camera_manager()
    test_simulation_and_tracking()
    test_exports()
    print("\n>>> ALL TESTS PASSED SUCCESSFULLY! <<<")
