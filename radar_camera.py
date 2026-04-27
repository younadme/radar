"""
Camera Radar - Motion Detection Radar using OpenCV
Requirements: pip install opencv-python numpy
Run: python radar_camera.py
"""

import cv2
import numpy as np
import math
import time

# ── Config ──────────────────────────────────────────────────────────────────
RADAR_SIZE   = 600          # radar window size (square)
CAM_W, CAM_H = 320, 240     # camera capture resolution
SWEEP_SPEED  = 2.5          # degrees per frame
BLIP_LIFE    = 60           # frames a blip stays visible
MOTION_THRESH = 25          # pixel diff threshold (0-255)
GRID         = 14           # motion detection grid cells per axis

BG_COLOR     = (5,  18,  10)
GREEN_DIM    = (20, 100,  45)
GREEN_MID    = (30, 160,  80)
GREEN_BRIGHT = (80, 220, 130)
GREEN_SWEEP  = (50, 200, 110)


# ── Helpers ──────────────────────────────────────────────────────────────────
def polar_to_xy(cx, cy, r, angle_deg):
    rad = math.radians(angle_deg)
    return (int(cx + r * math.cos(rad)), int(cy + r * math.sin(rad)))


def draw_radar_background(img, cx, cy, radius):
    img[:] = BG_COLOR

    # range rings
    for i in range(1, 5):
        r = int(radius * i / 4)
        cv2.circle(img, (cx, cy), r, GREEN_DIM, 1, cv2.LINE_AA)

    # crosshairs / spoke lines
    for angle in range(0, 360, 45):
        x, y = polar_to_xy(cx, cy, radius, angle)
        cv2.line(img, (cx, cy), (x, y), GREEN_DIM, 1, cv2.LINE_AA)

    # outer border ring
    cv2.circle(img, (cx, cy), radius, GREEN_MID, 1, cv2.LINE_AA)

    # compass labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.38, 1
    for label, angle in [("N", -90), ("E", 0), ("S", 90), ("W", 180)]:
        lx, ly = polar_to_xy(cx, cy, radius - 14, angle)
        cv2.putText(img, label, (lx - 5, ly + 5), font, fs, GREEN_BRIGHT, th, cv2.LINE_AA)


def draw_sweep(img, cx, cy, radius, angle_deg, trail_steps=40):
    overlay = img.copy()
    for i in range(trail_steps, 0, -1):
        a = angle_deg - i * (SWEEP_SPEED * 0.7)
        alpha = (trail_steps - i) / trail_steps * 0.35
        x, y = polar_to_xy(cx, cy, radius, a)
        # filled wedge slice as a thin triangle
        pts = np.array([[cx, cy], [x, y],
                        [*polar_to_xy(cx, cy, radius, a + SWEEP_SPEED * 0.7)]], np.int32)
        cv2.fillPoly(overlay, [pts], GREEN_SWEEP)

    # blend trail
    cv2.addWeighted(overlay, 0.28, img, 0.72, 0, img)

    # bright sweep line
    x, y = polar_to_xy(cx, cy, radius, angle_deg)
    cv2.line(img, (cx, cy), (x, y), GREEN_BRIGHT, 2, cv2.LINE_AA)


def draw_blips(img, blips):
    for bx, by, age, max_age in blips:
        life = 1.0 - age / max_age
        radius_blip = max(2, int(5 * life))
        alpha_blip  = life

        # inner dot
        color = tuple(int(c * alpha_blip) for c in GREEN_BRIGHT)
        cv2.circle(img, (bx, by), radius_blip, color, -1, cv2.LINE_AA)

        # expanding ring
        ring_r = int(14 * (1 - life) + 4)
        ring_a = life * 0.6
        ring_c = tuple(int(c * ring_a) for c in GREEN_MID)
        cv2.circle(img, (bx, by), ring_r, ring_c, 1, cv2.LINE_AA)


def detect_motion(frame, prev_gray, cx, cy, radius):
    """
    Compare current frame to previous. Return list of (nx, ny) in [-1,1]
    for cells with significant motion inside the radar circle.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if prev_gray is None:
        return gray, []

    diff = cv2.absdiff(gray, prev_gray)
    _, thresh = cv2.threshold(diff, MOTION_THRESH, 255, cv2.THRESH_BINARY)

    h, w = gray.shape
    targets = []
    cell_w = w // GRID
    cell_h = h // GRID

    for row in range(GRID):
        for col in range(GRID):
            cell = thresh[row * cell_h:(row + 1) * cell_h,
                          col * cell_w:(col + 1) * cell_w]
            if cell.mean() > 8:
                # normalised position -1..1
                nx = (col / GRID - 0.5) * 2
                ny = (row / GRID - 0.5) * 2
                if nx * nx + ny * ny < 1.0:   # inside radar circle
                    targets.append((nx, ny))

    return gray, targets


def map_to_radar(nx, ny, cx, cy, radius):
    """Map normalised camera coord to radar pixel coord."""
    bx = int(cx + nx * radius * 0.99)
    by = int(cy + ny * radius * 0.99)
    return bx, by


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    cx = cy = RADAR_SIZE // 2
    radius = RADAR_SIZE // 2 - 20

    radar_img = np.zeros((RADAR_SIZE, RADAR_SIZE, 3), dtype=np.uint8)
    angle     = 0.0
    blips     = []          # list of [bx, by, age, max_age]
    prev_gray = None
    contact_count = 0

    font = cv2.FONT_HERSHEY_SIMPLEX

    print("Camera Radar started. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # motion detection
        prev_gray, targets = detect_motion(frame, prev_gray, cx, cy, radius)

        # spawn blips for detected targets near the current sweep angle
        for nx, ny in targets:
            target_angle = math.degrees(math.atan2(ny, nx)) % 360
            sweep_angle  = angle % 360
            diff = abs(target_angle - sweep_angle)
            if diff > 180:
                diff = 360 - diff
            if diff < 20:          # only reveal blip when sweep passes over it
                bx, by = map_to_radar(nx, ny, cx, cy, radius)
                blips.append([bx, by, 0, BLIP_LIFE])

        contact_count = len(targets)

        # age out old blips
        blips = [[bx, by, age + 1, mx] for bx, by, age, mx in blips if age < mx]

        # draw
        draw_radar_background(radar_img, cx, cy, radius)
        draw_sweep(radar_img, cx, cy, radius, angle)
        draw_blips(radar_img, blips)

        # HUD text
        ts = time.strftime("%H:%M:%S")
        status = f"CONTACTS: {contact_count}   SWEEP: {int(angle % 360):03d} deg   {ts}"
        cv2.putText(radar_img, status, (12, RADAR_SIZE - 14),
                    font, 0.38, GREEN_MID, 1, cv2.LINE_AA)
        cv2.putText(radar_img, "RADAR  [Q] quit", (12, 22),
                    font, 0.4, GREEN_MID, 1, cv2.LINE_AA)

        # mask outside circle
        mask = np.zeros_like(radar_img)
        cv2.circle(mask, (cx, cy), radius, (255, 255, 255), -1)
        radar_img = cv2.bitwise_and(radar_img, mask)
        # redraw border (masked away)
        cv2.circle(radar_img, (cx, cy), radius, GREEN_MID, 1, cv2.LINE_AA)
        cv2.putText(radar_img, status, (12, RADAR_SIZE - 14),
                    font, 0.38, GREEN_MID, 1, cv2.LINE_AA)
        cv2.putText(radar_img, "RADAR  [Q] quit", (12, 22),
                    font, 0.4, GREEN_MID, 1, cv2.LINE_AA)

        angle = (angle + SWEEP_SPEED) % 360

        cv2.imshow("Camera Radar", radar_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Radar stopped.")


if __name__ == "__main__":
    main()
