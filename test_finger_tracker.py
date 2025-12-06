import cv2
import numpy as np
import time
from joblib import load
from hog_utils import create_hog, compute_hog, sliding_windows

HOG_WIN_SIZE = (64, 64)
BLOCK_SIZE = (16, 16)
BLOCK_STRIDE = (8, 8)
CELL_SIZE = (8, 8)
NBINS = 9

SVM_MODEL_PATH = "models/hog_finger_svm.joblib"

SLIDE_STEP = 32  # pixels between windows on each scaled frame

# Base scale for detection; we'll build a small pyramid around this
DETECT_SCALE = 0.5

# Multi-scale pyramid factors (relative to full-res)
PYRAMID_SCALES = [1.0, 0.75, DETECT_SCALE]  # e.g. full, 3/4, and 1/2 size

# SVM decision-function threshold.
SCORE_THRESHOLD = 0.3

DETECT_EVERY_N = 3  # do full HOG detection every N frames

# Exclusion zone (middle vertical band in the FULL-RES frame)
EXCLUDE_X1_FRAC = 0.30  # left edge of middle band (30% of width)
EXCLUDE_X2_FRAC = 0.70  # right edge of middle band (70% of width)
EXCLUDE_Y1_FRAC = 0.0
EXCLUDE_Y2_FRAC = 1.0

# Paddle config
PADDLE_WIDTH = 20
PADDLE_HEIGHT = 120
PADDLE_X_OFFSET = 40  # distance from each side wall
PADDLE_ALPHA = 0.4  # smoothing factor for player paddle movement

# AI paddle config
AI_PADDLE_ALPHA = 0.2  # smoothing factor for AI paddle following ball

# Ball config
BALL_RADIUS = 12
BALL_SPEED_X = 350.0  # px/sec (horizontal)
BALL_SPEED_Y = 250.0  # px/sec (vertical)

# Fingertip motion sanity check
# Max physically plausible fingertip speed (pixels per second).
# At 30 FPS and 1500 px/s, max per-frame jump ~ 50 px.
MAX_FINGER_SPEED = 1500.0


# ---------------------------------------------------------
# FINGER DETECTION (HOG + SVM, MULTI-SCALE)
# ---------------------------------------------------------

def detect_finger_hog(frame, hog, clf):
    """
    Run sliding-window HOG + SVM on a multi-scale pyramid of the frame.
    - Uses only the RIGHT HALF of each scaled frame.
    - Skips windows whose center lies in the central vertical exclusion band
      (EXCLUDE_*_FRAC), defined in *scaled* coordinates.
    Returns (x, y, w, h), best_score in original-frame coordinates.
    If no good match (score < SCORE_THRESHOLD), returns (None, best_score).
    """
    H, W = frame.shape[:2]
    win_w, win_h = hog.winSize

    best_score = -np.inf
    best_box_full = None

    for scale in PYRAMID_SCALES:
        if scale <= 0:
            continue

        small = cv2.resize(frame, None, fx=scale, fy=scale)
        Hs, Ws = small.shape[:2]

        if Hs < win_h or Ws < win_w:
            continue

        # Right half in small-frame coordinates
        right_half_x_min = Ws // 2

        # Exclusion band in this scale's coordinates
        ex_x1 = int(EXCLUDE_X1_FRAC * Ws)
        ex_x2 = int(EXCLUDE_X2_FRAC * Ws)
        ex_y1 = int(EXCLUDE_Y1_FRAC * Hs)
        ex_y2 = int(EXCLUDE_Y2_FRAC * Hs)

        for (x, y, patch) in sliding_windows(small, hog=hog, step=SLIDE_STEP):
            cx = x + win_w // 2
            cy = y + win_h // 2

            # 1) Only use right half
            if cx < right_half_x_min:
                continue

            # 2) Skip if center in middle exclusion band
            if ex_x1 <= cx <= ex_x2 and ex_y1 <= cy <= ex_y2:
                continue

            feat = compute_hog(patch, hog)
            score = clf.decision_function([feat])[0]

            if score > best_score:
                best_score = score
                xs, ys = x, y
                ws, hs = win_w, win_h
                x_full = int(xs / scale)
                y_full = int(ys / scale)
                w_full = int(ws / scale)
                h_full = int(hs / scale)
                best_box_full = (x_full, y_full, w_full, h_full)

    if best_box_full is None or best_score < SCORE_THRESHOLD:
        return None, best_score

    return best_box_full, best_score


# ---------------------------------------------------------
# FINGERTIP ESTIMATION USING CONTOURS
# ---------------------------------------------------------

def find_fingertip_in_box(frame, hand_box):
    x, y, w, h = hand_box
    roi = frame[y:y + h, x:x + w]

    if roi.size == 0:
        return None, None

    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)

    lower = np.array([0, 135, 85], dtype=np.uint8)
    upper = np.array([255, 180, 135], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)

    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    cnt = max(contours, key=cv2.contourArea)

    if cv2.contourArea(cnt) < 0.02 * (w * h):
        return None, None

    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None, None

    cx_local = int(M["m10"] / M["m00"])
    cy_local = int(M["m01"] / M["m00"])

    max_dist = -1
    fx_local, fy_local = None, None

    # Prefer points above centroid
    for p in cnt[:, 0, :]:
        px, py = int(p[0]), int(p[1])
        if py > cy_local:
            continue
        dx = px - cx_local
        dy = py - cy_local
        d2 = dx * dx + dy * dy
        if d2 > max_dist:
            max_dist = d2
            fx_local, fy_local = px, py

    # Fallback: farthest overall
    if fx_local is None:
        max_dist = -1
        for p in cnt[:, 0, :]:
            px, py = int(p[0]), int(p[1])
            dx = px - cx_local
            dy = py - cy_local
            d2 = dx * dx + dy * dy
            if d2 > max_dist:
                max_dist = d2
                fx_local, fy_local = px, py

    if fx_local is None:
        return None, None

    fx = fx_local + x
    fy = fy_local + y
    cx = cx_local + x
    cy = cy_local + y

    return (fx, fy), (cx, cy)


# ---------------------------------------------------------
# MAIN TEST LOOP (Pong-style)
# ---------------------------------------------------------

def main():
    cap = cv2.VideoCapture(0)  # change index if needed

    hog = create_hog(
        win_size=HOG_WIN_SIZE,
        block_size=BLOCK_SIZE,
        block_stride=BLOCK_STRIDE,
        cell_size=CELL_SIZE,
        nbins=NBINS
    )
    clf = load(SVM_MODEL_PATH)

    prev_tip = None
    prev_time = None
    ema_tip = None
    alpha = 0.3  # fingertip smoothing

    frame_idx = 0
    hand_box = None
    best_score = -np.inf

    t_prev = time.time()

    # Game objects - initialized after first frame (once we know H, W)
    game_initialized = False
    player_paddle_cy = None  # right paddle (finger controlled)
    ai_paddle_cy = None  # left paddle (computer controlled)

    ball_x = ball_y = None
    ball_vx = ball_vy = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        display = frame.copy()
        frame_idx += 1

        H, W = frame.shape[:2]

        # Initialize game positions once we know frame size
        if not game_initialized:
            player_paddle_cy = H // 2
            ai_paddle_cy = H // 2

            ball_x = W // 2
            ball_y = H // 2
            ball_vx = BALL_SPEED_X  # start moving to the right
            ball_vy = BALL_SPEED_Y  # downward

            game_initialized = True

        # FPS estimation
        t_now = time.time()
        dt_frame = t_now - t_prev
        fps = 1.0 / dt_frame if dt_frame > 0 else 0.0
        t_prev = t_now

        # -------------------------
        # Visualize exclusion band (in full-res coords)
        # -------------------------
        ex_x1 = int(EXCLUDE_X1_FRAC * W)
        ex_x2 = int(EXCLUDE_X2_FRAC * W)
        ex_y1 = int(EXCLUDE_Y1_FRAC * H)
        ex_y2 = int(EXCLUDE_Y2_FRAC * H)

        cv2.rectangle(display, (ex_x1, ex_y1), (ex_x2, ex_y2),
                      (0, 255, 255), 1)
        cv2.putText(display, "No-detect middle zone",
                    (max(ex_x1, 10), max(ex_y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)

        # -------------------------
        # HOG finger detection (right half only, excluding middle band)
        # -------------------------
        if frame_idx % DETECT_EVERY_N == 0 or hand_box is None:
            hand_box, best_score = detect_finger_hog(display, hog, clf)

        fingertip, centroid = None, None

        # Only try fingertip if we actually have a good HOG box
        if hand_box is not None:
            fingertip, centroid = find_fingertip_in_box(display, hand_box)

            # If fingertip failed, treat this as "no detection"
            if fingertip is None:
                hand_box = None

        # Draw box *only* if fingertip also exists (so we don't "lock" on junk)
        if hand_box is not None and fingertip is not None:
            x, y, w, h = hand_box
            cv2.rectangle(display, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)

        vx, vy = 0.0, 0.0

        if fingertip is not None:
            fx, fy = fingertip

            # -------------------------------------------------
            # SANITY CHECK: reject physically impossible jumps
            # -------------------------------------------------
            valid_tip = True
            if prev_tip is not None and prev_time is not None:
                dt = t_now - prev_time
                if dt > 0:
                    dx = fx - prev_tip[0]
                    dy = fy - prev_tip[1]
                    dist = np.hypot(dx, dy)
                    max_dist = MAX_FINGER_SPEED * dt

                    if dist > max_dist:
                        # This is an implausible jump: ignore this fingertip
                        valid_tip = False

            if valid_tip:
                # Smooth fingertip
                if ema_tip is None:
                    ema_tip = np.array([fx, fy], dtype=np.float32)
                else:
                    ema_tip = alpha * np.array([fx, fy], dtype=np.float32) + \
                              (1 - alpha) * ema_tip

                # Draw fingertip + centroid
                cv2.circle(display, (int(ema_tip[0]), int(ema_tip[1])),
                           6, (0, 0, 255), -1)
                if centroid is not None:
                    cv2.circle(display, centroid, 4, (255, 0, 0), -1)

                # Velocity (pixels/sec) using unsmoothed tip
                if prev_tip is not None and prev_time is not None:
                    dt = t_now - prev_time
                    if dt > 0:
                        vx = (fx - prev_tip[0]) / dt
                        vy = (fy - prev_tip[1]) / dt

                prev_tip = (fx, fy)
                prev_time = t_now

                # -------------------------
                # Right paddle (player) vertical control
                # -------------------------
                tip_y = ema_tip[1]
                if player_paddle_cy is None:
                    player_paddle_cy = tip_y
                else:
                    player_paddle_cy = PADDLE_ALPHA * tip_y + \
                                       (1 - PADDLE_ALPHA) * player_paddle_cy
            else:
                # Invalid jump: ignore this frame's tip, keep previous EMA and paddle
                # (Optionally, you could visualize this with a different color)
                pass

        else:
            # No fingertip: clear smoothing state and don't update paddle
            ema_tip = None
            prev_tip = None
            prev_time = None

        # Clamp paddles to frame bounds
        half_h = PADDLE_HEIGHT // 2

        if player_paddle_cy is None:
            player_paddle_cy = H // 2
        player_paddle_cy = max(half_h, min(H - half_h, player_paddle_cy))

        if ai_paddle_cy is None:
            ai_paddle_cy = H // 2
        ai_paddle_cy = max(half_h, min(H - half_h, ai_paddle_cy))

        # -------------------------
        # Ball movement
        # -------------------------
        if dt_frame > 0:
            ball_x += ball_vx * dt_frame
            ball_y += ball_vy * dt_frame

        # Top/bottom wall collision
        if ball_y - BALL_RADIUS < 0:
            ball_y = BALL_RADIUS
            ball_vy *= -1
        elif ball_y + BALL_RADIUS > H:
            ball_y = H - BALL_RADIUS
            ball_vy *= -1

        # -------------------------
        # Paddle positions (rectangles)
        # -------------------------
        # Left (AI) paddle
        lp_x1 = PADDLE_X_OFFSET
        lp_x2 = lp_x1 + PADDLE_WIDTH
        lp_y1 = int(ai_paddle_cy - half_h)
        lp_y2 = int(ai_paddle_cy + half_h)

        # Right (player) paddle
        rp_x2 = W - PADDLE_X_OFFSET
        rp_x1 = rp_x2 - PADDLE_WIDTH
        rp_y1 = int(player_paddle_cy - half_h)
        rp_y2 = int(player_paddle_cy + half_h)

        # -------------------------
        # AI paddle movement (follow ball)
        # -------------------------
        ai_target_cy = ball_y
        ai_paddle_cy = (1 - AI_PADDLE_ALPHA) * ai_paddle_cy + \
                       AI_PADDLE_ALPHA * ai_target_cy
        ai_paddle_cy = max(half_h, min(H - half_h, ai_paddle_cy))
        lp_y1 = int(ai_paddle_cy - half_h)
        lp_y2 = int(ai_paddle_cy + half_h)

        # -------------------------
        # Ball-paddle collisions
        # -------------------------

        # Collision with right (player) paddle
        if (ball_x + BALL_RADIUS >= rp_x1 and
                ball_x - BALL_RADIUS <= rp_x2 and
                rp_y1 <= ball_y <= rp_y2 and
                ball_vx > 0):
            ball_x = rp_x1 - BALL_RADIUS
            ball_vx *= -1

        # Collision with left (AI) paddle
        if (ball_x - BALL_RADIUS <= lp_x2 and
                ball_x + BALL_RADIUS >= lp_x1 and
                lp_y1 <= ball_y <= lp_y2 and
                ball_vx < 0):
            ball_x = lp_x2 + BALL_RADIUS
            ball_vx *= -1

        # If ball goes off-screen horizontally, reset to center and reverse direction
        if ball_x < -BALL_RADIUS or ball_x > W + BALL_RADIUS:
            ball_x = W // 2
            ball_y = H // 2
            ball_vx *= -1  # send it back the other way

        # -------------------------
        # Draw paddles & ball
        # -------------------------
        # Left AI paddle
        cv2.rectangle(display, (lp_x1, lp_y1), (lp_x2, lp_y2),
                      (255, 0, 0), -1)
        cv2.putText(display, "AI",
                    (lp_x1, max(lp_y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 0, 0), 2)

        # Right player paddle
        cv2.rectangle(display, (rp_x1, rp_y1), (rp_x2, rp_y2),
                      (0, 255, 0), -1)
        cv2.putText(display, "YOU",
                    (rp_x1 - 10, max(rp_y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)

        # Ball
        cv2.circle(display, (int(ball_x), int(ball_y)),
                   BALL_RADIUS, (0, 255, 255), -1)

        # HUD
        cv2.putText(display, "Right side = finger control | Left side = AI",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)
        cv2.putText(display, f"SVM score: {best_score:.2f}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)

        cv2.imshow("Finger Pong (right side player)", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
