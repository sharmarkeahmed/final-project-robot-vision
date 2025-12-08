"""
main.py

Run this file to play the ping pong game
"""

import time
import cv2

from detector import FingerDetector
from game import PongGame
from config import (
    DEBUG_MODE,
    EXCLUDE_X1_FRAC, EXCLUDE_X2_FRAC, EXCLUDE_Y1_FRAC, EXCLUDE_Y2_FRAC,
    DETECT_EVERY_N,
)


def main():
    cap = cv2.VideoCapture(0)

    detector = FingerDetector()
    game = PongGame()

    frame_idx = 0
    t_prev = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        display = frame.copy()
        frame_idx += 1

        H, W = frame.shape[:2]

        # FPS estimation
        t_now = time.time()
        dt_frame = t_now - t_prev
        t_prev = t_now
        fps = 1.0 / dt_frame if dt_frame > 0 else 0.0

        # Debug-only: visualize exclusion band in full-res coords
        if DEBUG_MODE:
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

        # Decide whether to run full HOG detection this frame
        run_detection = (frame_idx % DETECT_EVERY_N == 0) or (detector.hand_box is None)

        fingertip_ema, centroid, hand_box, best_score = detector.update(
            display, t_now, run_detection=run_detection
        )

        # Debug-only: draw detection box and fingertip/centroid overlays
        if DEBUG_MODE and hand_box is not None and fingertip_ema is not None:
            x, y, w, h = hand_box
            cv2.rectangle(display, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)  # green box

            cv2.circle(display, (int(fingertip_ema[0]), int(fingertip_ema[1])),
                       6, (0, 0, 255), -1)  # red fingertip
            if centroid is not None:
                cv2.circle(display, centroid, 4, (255, 0, 0), -1)  # blue centroid

        # Update game using fingertip y (if any)
        fingertip_y = fingertip_ema[1] if fingertip_ema is not None else None
        lp_rect, rp_rect = game.update(H, W, dt_frame, fingertip_y)
        game.draw(display, lp_rect, rp_rect)

        # -------------------------
        # HUD
        # -------------------------
        # FPS
        cv2.putText(display, f"FPS: {fps:.1f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)

        # Basic instructions
        cv2.putText(display, "Right side = finger control | Left side = AI",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)

        # Score (always visible)
        score_text = f"SCORE  YOU {game.player_score} : {game.ai_score} AI"
        cv2.putText(display, score_text,
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        # SVM score only in debug mode
        if DEBUG_MODE:
            cv2.putText(display, f"SVM score: {best_score:.2f}",
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)

        cv2.imshow("Finger Pong (right side player)", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # for runtime toggle:
        # elif key == ord('d'):
        #     DEBUG_MODE = not DEBUG_MODE

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
