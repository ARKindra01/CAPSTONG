"""
simulation.py - Realistic Synthetic Laboratory Rodent Movement Generator
Generates realistic video frames of a mouse exploring an arena (exhibiting thigmotaxis,
wall-hugging, corner visits, center crossings, and periodic freezing/grooming pauses).
Enables instant 1-click verification of tracking, zones, and metrics without external files.
"""

from typing import Tuple
import math
import random
import cv2
import numpy as np


class RodentSimulation:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps

        # Arena bounds inside simulation (light grey box)
        self.margin_x = 70
        self.margin_y = 50
        self.arena_w = width - 2 * self.margin_x
        self.arena_h = height - 2 * self.margin_y

        # Rodent physical state
        self.x = float(self.margin_x + 50)
        self.y = float(self.margin_y + 50)
        self.heading = 0.0  # radians
        self.speed = 4.0    # pixels/frame
        self.target_speed = 4.0

        # State machine: "EXPLORING", "WALL_HUGGING", "DARTING", "FREEZING"
        self.state = "WALL_HUGGING"
        self.state_timer = 0
        self.tail_history = []

        # Generate static background once
        self.bg_frame = self._generate_arena_background()

    def _generate_arena_background(self) -> np.ndarray:
        """Draws realistic high-contrast lab arena box with walls and floor."""
        frame = np.full((self.height, self.width, 3), 235, dtype=np.uint8)  # Light room

        # Outer room floor texture
        noise = np.random.randint(-4, 4, (self.height, self.width, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Arena floor (clean white-grey acrylic plate)
        cv2.rectangle(
            frame,
            (self.margin_x, self.margin_y),
            (self.margin_x + self.arena_w, self.margin_y + self.arena_h),
            (250, 250, 252),
            -1
        )

        # Arena wall border (light grey border)
        cv2.rectangle(
            frame,
            (self.margin_x, self.margin_y),
            (self.margin_x + self.arena_w, self.margin_y + self.arena_h),
            (170, 175, 180),
            3
        )
        return frame

    def update_physics(self):
        """Updates rodent state machine, heading, speed, and position."""
        self.state_timer -= 1

        min_x = self.margin_x + 22
        max_x = self.margin_x + self.arena_w - 22
        min_y = self.margin_y + 22
        max_y = self.margin_y + self.arena_h - 22

        # State transitions
        if self.state_timer <= 0:
            rand = random.random()
            if rand < 0.15:
                # Freeze / Grooming (stands still)
                self.state = "FREEZING"
                self.state_timer = random.randint(30, 90)  # 1 - 3 seconds
                self.target_speed = 0.0
            elif rand < 0.55:
                # Thigmotaxis (wall following)
                self.state = "WALL_HUGGING"
                self.state_timer = random.randint(60, 180)
                self.target_speed = random.uniform(3.0, 5.0)
            elif rand < 0.85:
                # Open exploration
                self.state = "EXPLORING"
                self.state_timer = random.randint(45, 120)
                self.target_speed = random.uniform(3.5, 6.0)
            else:
                # Sudden dart across arena
                self.state = "DARTING"
                self.state_timer = random.randint(30, 60)
                self.target_speed = random.uniform(7.0, 10.0)

        # Smooth acceleration / deceleration
        self.speed += (self.target_speed - self.speed) * 0.15

        # Behavioral steering
        if self.state == "FREEZING":
            # Very small head twitch
            self.heading += random.uniform(-0.04, 0.04)
        elif self.state == "WALL_HUGGING":
            # Favor moving parallel to closest wall
            dist_left = abs(self.x - min_x)
            dist_right = abs(self.x - max_x)
            dist_top = abs(self.y - min_y)
            dist_bottom = abs(self.y - max_y)
            min_dist = min(dist_left, dist_right, dist_top, dist_bottom)

            if min_dist > 35:
                # Steer towards nearest wall
                if min_dist == dist_left:
                    target_ang = math.pi
                elif min_dist == dist_right:
                    target_ang = 0.0
                elif min_dist == dist_top:
                    target_ang = -math.pi / 2
                else:
                    target_ang = math.pi / 2
                self.heading += (target_ang - self.heading) * 0.1
            else:
                # Follow perimeter
                self.heading += random.uniform(-0.15, 0.15)
        else:
            # Random exploration walk
            self.heading += random.uniform(-0.35, 0.35)

        # Move forward
        if self.speed > 0.05:
            dx = math.cos(self.heading) * self.speed
            dy = math.sin(self.heading) * self.speed
            self.x += dx
            self.y += dy

        # Wall collisions and bounce
        if self.x < min_x:
            self.x = min_x
            self.heading = math.pi - self.heading + random.uniform(-0.3, 0.3)
        elif self.x > max_x:
            self.x = max_x
            self.heading = math.pi - self.heading + random.uniform(-0.3, 0.3)

        if self.y < min_y:
            self.y = min_y
            self.heading = -self.heading + random.uniform(-0.3, 0.3)
        elif self.y > max_y:
            self.y = max_y
            self.heading = -self.heading + random.uniform(-0.3, 0.3)

        # Record tail track
        self.tail_history.append((self.x, self.y))
        if len(self.tail_history) > 12:
            self.tail_history.pop(0)

    def get_next_frame(self) -> np.ndarray:
        """Renders the rodent onto the arena background and returns BGR frame."""
        self.update_physics()
        frame = self.bg_frame.copy()

        cx, cy = int(self.x), int(self.y)
        ang_deg = math.degrees(self.heading)

        # 1. Draw Tail
        if len(self.tail_history) >= 4:
            pts = []
            for i, (tx, ty) in enumerate(self.tail_history):
                pts.append([int(tx), int(ty)])
            cv2.polylines(frame, [np.array(pts, np.int32)], False, (180, 160, 160), 2, cv2.LINE_AA)

        # 2. Draw Soft Shadow under animal
        cv2.ellipse(frame, (cx + 2, cy + 3), (17, 10), ang_deg, 0, 360, (200, 205, 210), -1)

        # 3. Draw Rodent Body (Dark C57BL/6 Mouse: charcoal/black coat)
        body_color = (40, 38, 38)
        cv2.ellipse(frame, (cx, cy), (15, 9), ang_deg, 0, 360, body_color, -1)

        # 4. Draw Head
        hx = int(cx + math.cos(self.heading) * 11)
        hy = int(cy + math.sin(self.heading) * 11)
        cv2.ellipse(frame, (hx, hy), (8, 6), ang_deg, 0, 360, body_color, -1)

        # 5. Draw Pinkish Ears
        ear_dist = 6
        ear_ang1 = self.heading + math.pi / 2
        ear_ang2 = self.heading - math.pi / 2
        e1x = int(hx + math.cos(ear_ang1) * ear_dist)
        e1y = int(hy + math.sin(ear_ang1) * ear_dist)
        e2x = int(hx + math.cos(ear_ang2) * ear_dist)
        e2y = int(hy + math.sin(ear_ang2) * ear_dist)
        cv2.circle(frame, (e1x, e1y), 3, (160, 140, 180), -1)
        cv2.circle(frame, (e2x, e2y), 3, (160, 140, 180), -1)

        # 6. Snout tip
        sx = int(hx + math.cos(self.heading) * 6)
        sy = int(hy + math.sin(self.heading) * 6)
        cv2.circle(frame, (sx, sy), 2, (150, 130, 170), -1)

        return frame
