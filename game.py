"""
game.py

Contains a PongGame class for running the ping pong game
"""

import cv2
import numpy as np  # for simple randomness in AI behavior

from config import (
    PADDLE_WIDTH, PADDLE_HEIGHT, PADDLE_X_OFFSET,
    PADDLE_ALPHA, AI_PADDLE_ALPHA,
    BALL_RADIUS, BALL_SPEED_X, BALL_SPEED_Y,
)


class PongGame:
    """
    Encapsulates Pong game state and logic:
    - Player (right) and AI (left) paddles
    - Ball position and velocity
    - Scoring (player vs AI)
    - Collisions and AI behavior

    The AI is intended to be "human-like":
    - Limited vertical speed (cannot instantly follow the ball)
    - Reacts more strongly when the ball is close and moving toward it
    - Is particularly vulnerable to quick vertical direction changes (jukes)
    - Occasional small mistakes so it is still beatable
    """

    def __init__(self):
        self.initialized = False

        self.player_paddle_cy = None
        self.ai_paddle_cy = None

        self.ball_x = None
        self.ball_y = None
        self.ball_vx = BALL_SPEED_X
        self.ball_vy = BALL_SPEED_Y

        self.player_score = 0
        self.ai_score = 0

        # AI behavior parameters (tune these to make AI easier/harder)
        self.ai_max_speed = 320.0      # px/sec vertical max speed
        self.ai_react_x_frac = 0.55    # AI fully reacts when ball_x < W * this
        self.ai_error = 0.0            # vertical offset error (pixels)
        self.ai_error_alpha = 0.95     # smoothing factor for AI error (EMA)

        # "Dumbness": chance that when ball is threatening, AI reacts poorly
        self.ai_dumb_chance = 0.2      # 0.0 = perfect robot, 1.0 = clown

        # Juke / reaction delay logic
        self.time = 0.0
        self.prev_ball_y = None
        self.prev_vy_sign = 0          # sign of previous vertical velocity
        self.last_dir_change_time = 0.0
        self.reaction_delay = 0.18     # seconds after vy flip where AI is slow to adapt

    def _init_if_needed(self, H, W):
        if self.initialized:
            return
        self.player_paddle_cy = H // 2
        self.ai_paddle_cy = H // 2
        self.ball_x = W // 2
        self.ball_y = H // 2
        self.initialized = True

    def update(self, H, W, dt, fingertip_y):
        """
        Update paddles and ball for one frame.

        Args:
            H, W: frame height and width.
            dt: time step (seconds).
            fingertip_y: smoothed fingertip y position in frame coords, or None.

        Returns:
            (lp_rect, rp_rect):
                lp_rect = (x1, y1, x2, y2) for left (AI) paddle
                rp_rect = (x1, y1, x2, y2) for right (player) paddle
        """
        self._init_if_needed(H, W)
        half_h = PADDLE_HEIGHT // 2

        # Advance internal time
        if dt > 0:
            self.time += dt

        # -------------------------
        # Player paddle follows fingertip (if present)
        # -------------------------
        if fingertip_y is not None:
            if self.player_paddle_cy is None:
                self.player_paddle_cy = fingertip_y
            else:
                self.player_paddle_cy = (
                    PADDLE_ALPHA * fingertip_y
                    + (1 - PADDLE_ALPHA) * self.player_paddle_cy
                )

        if self.player_paddle_cy is None:
            self.player_paddle_cy = H // 2

        self.player_paddle_cy = max(half_h, min(H - half_h, self.player_paddle_cy))

        # -------------------------
        # Move ball
        # -------------------------
        if dt > 0:
            self.ball_x += self.ball_vx * dt
            self.ball_y += self.ball_vy * dt

        # Top/bottom walls
        if self.ball_y - BALL_RADIUS < 0:
            self.ball_y = BALL_RADIUS
            self.ball_vy *= -1
        elif self.ball_y + BALL_RADIUS > H:
            self.ball_y = H - BALL_RADIUS
            self.ball_vy *= -1

        # -------------------------
        # Track ball vertical direction & juke detection
        # -------------------------
        recent_flip = False
        if self.prev_ball_y is not None and dt > 0:
            vy_est = (self.ball_y - self.prev_ball_y) / dt
            # Ignore tiny motion
            if abs(vy_est) < 20:
                vy_sign = 0
            else:
                vy_sign = int(np.sign(vy_est))

            if vy_sign != 0 and vy_sign != self.prev_vy_sign:
                # Vertical direction just flipped → possible juke
                self.prev_vy_sign = vy_sign
                self.last_dir_change_time = self.time

        # Check if we're still within the reaction delay window
        if (self.time - self.last_dir_change_time) < self.reaction_delay:
            recent_flip = True

        self.prev_ball_y = self.ball_y

        # -------------------------
        # AI target logic (juke-able)
        # -------------------------
        # Base target is the ball y
        ai_target_cy = self.ball_y

        # Small random vertical bias so AI doesn't line up perfectly
        jitter = np.random.uniform(-6.0, 6.0)
        self.ai_error = (
            self.ai_error_alpha * self.ai_error
            + (1.0 - self.ai_error_alpha) * jitter * 3.0
        )

        ball_moving_toward_ai = (self.ball_vx < 0)

        if ball_moving_toward_ai:
            # Ball moving left (toward AI)
            if self.ball_x < W * self.ai_react_x_frac:
                # Ball is close enough to be dangerous

                if recent_flip:
                    # Recently juked: AI is slow / confused → under-react.
                    # Move only partway toward the ball, mixing with current position.
                    ai_target_cy = (
                        0.6 * self.ai_paddle_cy
                        + 0.4 * (self.ball_y + self.ai_error)
                    )
                else:
                    # No recent juke → normal tracking with small chance of bad aim
                    if np.random.rand() < self.ai_dumb_chance:
                        # Slightly bad reaction: overshoot/undershoot a bit
                        miss = np.random.uniform(-200, 200)
                        ai_target_cy = self.ball_y + self.ai_error + miss
                    else:
                        ai_target_cy = self.ball_y + self.ai_error
            else:
                # Ball still far: be lazier, drift between center and ball
                ai_target_cy = 0.5 * (self.ball_y + H / 2) + 0.5 * self.ai_error
        else:
            # Ball moving away from AI: relax toward center
            ai_target_cy = H / 2 + 0.7 * self.ai_error

        # -------------------------
        # AI paddle movement with speed limit
        # -------------------------
        if self.ai_paddle_cy is None:
            self.ai_paddle_cy = H // 2

        dy = ai_target_cy - self.ai_paddle_cy

        if dt > 0:
            max_step = self.ai_max_speed * dt
            if abs(dy) > max_step:
                dy = np.sign(dy) * max_step

        self.ai_paddle_cy += dy
        self.ai_paddle_cy = max(half_h, min(H - half_h, self.ai_paddle_cy))

        # -------------------------
        # Paddle rectangles
        # -------------------------
        lp_x1 = PADDLE_X_OFFSET
        lp_x2 = lp_x1 + PADDLE_WIDTH
        lp_y1 = int(self.ai_paddle_cy - half_h)
        lp_y2 = int(self.ai_paddle_cy + half_h)

        rp_x2 = W - PADDLE_X_OFFSET
        rp_x1 = rp_x2 - PADDLE_WIDTH
        rp_y1 = int(self.player_paddle_cy - half_h)
        rp_y2 = int(self.player_paddle_cy + half_h)

        # -------------------------
        # Ball-paddle collisions
        # -------------------------
        # Collision with right (player) paddle
        if (self.ball_x + BALL_RADIUS >= rp_x1 and
            self.ball_x - BALL_RADIUS <= rp_x2 and
            rp_y1 <= self.ball_y <= rp_y2 and
            self.ball_vx > 0):
            self.ball_x = rp_x1 - BALL_RADIUS
            self.ball_vx *= -1

        # Collision with left (AI) paddle
        if (self.ball_x - BALL_RADIUS <= lp_x2 and
            self.ball_x + BALL_RADIUS >= lp_x1 and
            lp_y1 <= self.ball_y <= lp_y2 and
            self.ball_vx < 0):
            self.ball_x = lp_x2 + BALL_RADIUS
            self.ball_vx *= -1

        # -------------------------
        # Off-screen (scoring)
        # -------------------------
        if self.ball_x < -BALL_RADIUS:
            # Ball exited left side → YOU score
            self.player_score += 1
            self.ball_x = W // 2
            self.ball_y = H // 2
            self.ball_vx = abs(self.ball_vx)  # send it right
        elif self.ball_x > W + BALL_RADIUS:
            # Ball exited right side → AI scores
            self.ai_score += 1
            self.ball_x = W // 2
            self.ball_y = H // 2
            self.ball_vx = -abs(self.ball_vx)  # send it left

        lp_rect = (lp_x1, lp_y1, lp_x2, lp_y2)
        rp_rect = (rp_x1, rp_y1, rp_x2, rp_y2)
        return lp_rect, rp_rect

    def draw(self, display, lp_rect, rp_rect):
        """
        Draw paddles and ball onto the display frame.
        Score text and HUD are handled in main.py.
        """
        (lp_x1, lp_y1, lp_x2, lp_y2) = lp_rect
        (rp_x1, rp_y1, rp_x2, rp_y2) = rp_rect

        # AI paddle (left)
        cv2.rectangle(display, (lp_x1, lp_y1), (lp_x2, lp_y2),
                      (255, 0, 0), -1)
        cv2.putText(display, "AI",
                    (lp_x1, max(lp_y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 0, 0), 2)

        # Player paddle (right)
        cv2.rectangle(display, (rp_x1, rp_y1), (rp_x2, rp_y2),
                      (0, 255, 0), -1)
        cv2.putText(display, "YOU",
                    (rp_x1 - 10, max(rp_y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)

        # Ball
        cv2.circle(display, (int(self.ball_x), int(self.ball_y)),
                   BALL_RADIUS, (0, 255, 255), -1)
