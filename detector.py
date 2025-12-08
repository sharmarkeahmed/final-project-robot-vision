"""
detector.py

Set of functions for detecting the finger. Uses HOG feature extraction from OpenCV & SVM model
produced from train_finger_svm.py
"""

import cv2
import numpy as np
from joblib import load

from hog_utils import create_hog, compute_hog, sliding_windows
from config import (
    HOG_WIN_SIZE, BLOCK_SIZE, BLOCK_STRIDE, CELL_SIZE, NBINS,
    SVM_MODEL_PATH, SLIDE_STEP, PYRAMID_SCALES, SCORE_THRESHOLD,
    EXCLUDE_X1_FRAC, EXCLUDE_X2_FRAC, EXCLUDE_Y1_FRAC, EXCLUDE_Y2_FRAC,
    MAX_FINGER_SPEED,
)


class FingerDetector:
    """
    Handles HOG + SVM hand-box detection and fingertip tracking, including:

    - Multi-scale sliding-window detection on the right half of the frame.
    - Skin-color segmentation inside the detected box to locate the fingertip.
    - Temporal smoothing of the fingertip.
    - Sanity check on fingertip motion to reject implausible jumps.
    """

    def __init__(self):
        self.hog = create_hog(
            win_size=HOG_WIN_SIZE,
            block_size=BLOCK_SIZE,
            block_stride=BLOCK_STRIDE,
            cell_size=CELL_SIZE,
            nbins=NBINS,
        )
        self.clf = load(SVM_MODEL_PATH)

        # detection state
        self.hand_box = None
        self.best_score = -np.inf

        # fingertip tracking state
        self.prev_tip = None
        self.prev_time = None
        self.ema_tip = None
        self.alpha = 0.3  # smoothing factor

    def _detect_finger_hog(self, frame):
        """
        Run sliding-window HOG + SVM on a multi-scale pyramid of the frame.
        - Uses only the RIGHT HALF of each scaled frame.
        - Skips windows whose center lies in the central vertical exclusion band
          (EXCLUDE_*_FRAC), defined in scaled coordinates.
        Returns:
            (box, best_score) in original frame coordinates, where box is (x,y,w,h)
            or (None, best_score) if no box passes the SCORE_THRESHOLD.
        """
        H, W = frame.shape[:2]
        win_w, win_h = self.hog.winSize

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

            for (x, y, patch) in sliding_windows(small, hog=self.hog, step=SLIDE_STEP):
                cx = x + win_w // 2
                cy = y + win_h // 2

                # 1) Only use right half
                if cx < right_half_x_min:
                    continue

                # 2) Skip if center in middle exclusion band
                if ex_x1 <= cx <= ex_x2 and ex_y1 <= cy <= ex_y2:
                    continue

                feat = compute_hog(patch, self.hog)
                score = self.clf.decision_function([feat])[0]

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

    @staticmethod
    def _find_fingertip_in_box(frame, hand_box):
        """
        Given a hand bounding box, segment likely skin pixels in YCrCb and
        locate a fingertip as the farthest contour point above the contour
        centroid (or farthest overall as fallback).

        Returns:
            fingertip (x, y), centroid (x, y) in full-frame coordinates,
            or (None, None) if detection fails.
        """
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

    def update(self, frame, t_now, run_detection=True):
        """
        Update detection + fingertip tracking on this frame.

        Args:
            frame: BGR image (full resolution).
            t_now: current time (seconds, from time.time()).
            run_detection: if False, reuse the last hand_box (if any) and only
                           update fingertip tracking; if True, run full HOG.

        Returns:
            fingertip_ema (tuple | None): smoothed fingertip (x, y) or None
            centroid (tuple | None): (x, y) or None
            hand_box (tuple | None): (x, y, w, h) or None
            best_score (float): latest SVM score for the current box
        """
        if run_detection or self.hand_box is None:
            self.hand_box, self.best_score = self._detect_finger_hog(frame)

        fingertip, centroid = None, None

        if self.hand_box is not None:
            fingertip, centroid = self._find_fingertip_in_box(frame, self.hand_box)
            if fingertip is None:
                # no fingertip inside the box; treat as failure
                self.hand_box = None

        fingertip_ema = None

        if fingertip is not None:
            fx, fy = fingertip

            # Physically plausible motion check
            valid_tip = True
            if self.prev_tip is not None and self.prev_time is not None:
                dt = t_now - self.prev_time
                if dt > 0:
                    dx = fx - self.prev_tip[0]
                    dy = fy - self.prev_tip[1]
                    dist = np.hypot(dx, dy)
                    max_dist = MAX_FINGER_SPEED * dt
                    if dist > max_dist:
                        valid_tip = False

            if valid_tip:
                # EMA smoothing
                if self.ema_tip is None:
                    self.ema_tip = np.array([fx, fy], dtype=np.float32)
                else:
                    self.ema_tip = self.alpha * np.array([fx, fy], np.float32) + \
                                   (1 - self.alpha) * self.ema_tip

                fingertip_ema = (float(self.ema_tip[0]), float(self.ema_tip[1]))

                self.prev_tip = (fx, fy)
                self.prev_time = t_now
            else:
                # Implausible jump: ignore this fingertip
                fingertip_ema = None
        else:
            # No fingertip this frame; reset temporal state
            self.prev_tip = None
            self.prev_time = None
            self.ema_tip = None

        return fingertip_ema, centroid, self.hand_box, self.best_score
