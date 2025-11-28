import cv2
import numpy as np
import os
from joblib import dump, load

from hog_utils import create_hog, compute_hog  # your existing HOG utils

# -----------------------------
# CONFIG
# -----------------------------
HOG_WIN_SIZE = (64, 64)      # must match your detection HOG
BLOCK_SIZE   = (16, 16)
BLOCK_STRIDE = (8, 8)
CELL_SIZE    = (8, 8)
NBINS        = 9

OUTPUT_DATASET_PATH = "data/hog_finger_click_dataset.joblib"

# -----------------------------
# GLOBALS FOR MOUSE CALLBACK
# -----------------------------
last_frame = None   # current frame
X = []              # list of feature vectors
y = []              # list of labels (1=positive, 0=negative)

hog = create_hog(
    win_size=HOG_WIN_SIZE,
    block_size=BLOCK_SIZE,
    block_stride=BLOCK_STRIDE,
    cell_size=CELL_SIZE,
    nbins=NBINS
)


def add_sample_at(x, y_img, label):
    """
    Extract a patch centered at (x, y_img) from last_frame,
    compute HOG, add to dataset with given label.
    """
    global last_frame, X, y

    if last_frame is None:
        return

    H, W = last_frame.shape[:2]
    win_w, win_h = HOG_WIN_SIZE

    # Top-left of patch
    x1 = int(x - win_w // 2)
    y1 = int(y_img - win_h // 2)
    x2 = x1 + win_w
    y2 = y1 + win_h

    # Ensure patch is within frame
    if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
        print("[WARN] Click too close to edge, patch would go out of frame. Ignoring.")
        return

    patch = last_frame[y1:y2, x1:x2]

    # Compute HOG (no image saved)
    feat = compute_hog(patch, hog)
    X.append(feat)
    y.append(label)

    label_str = "POSITIVE" if label == 1 else "NEGATIVE"
    y_arr = np.array(y)
    n_pos = int((y_arr == 1).sum())
    n_neg = int((y_arr == 0).sum())
    print(f"[INFO] Saved {label_str} sample at ({x}, {y_img}). "
          f"Total: pos={n_pos}, neg={n_neg}, total={len(y)}")


def mouse_callback(event, x, y_img, flags, param):
    """
    Left-click  -> positive sample (finger present at/near click)
    Right-click -> negative sample
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        # POSITIVE
        add_sample_at(x, y_img, label=1)

    elif event == cv2.EVENT_RBUTTONDOWN:
        # NEGATIVE
        add_sample_at(x, y_img, label=0)


def maybe_load_existing_dataset():
    """
    If OUTPUT_DATASET_PATH exists, ask user whether to load + append.
    If yes, load existing X, y and convert to lists so we can extend them.
    """
    global X, y

    if not os.path.exists(OUTPUT_DATASET_PATH):
        print("[INFO] No existing dataset found. Starting fresh.")
        return

    ans = input(f"[PROMPT] Existing dataset found at '{OUTPUT_DATASET_PATH}'. "
                "Load and append new samples? (y/n): ").strip().lower()

    if ans == "y":
        X_arr, y_arr = load(OUTPUT_DATASET_PATH)
        print(f"[INFO] Loaded existing dataset: X={X_arr.shape}, y={y_arr.shape}")
        # Convert to lists to keep appending
        X = [row for row in X_arr]
        y = list(y_arr)
        print(f"[INFO] Continuing from existing dataset. Current total samples: {len(y)}")
    else:
        print("[INFO] Starting fresh dataset (existing file will be overwritten on save).")


def main():
    global last_frame, X, y

    # Optional: create data folder if not present
    data_dir = os.path.dirname(OUTPUT_DATASET_PATH) or "."
    os.makedirs(data_dir, exist_ok=True)

    # Ask whether to load existing data
    maybe_load_existing_dataset()

    cap = cv2.VideoCapture(0)  # adjust index if needed

    cv2.namedWindow("Collect finger data (click)")
    cv2.setMouseCallback("Collect finger data (click)", mouse_callback)

    print("[INFO] Left-click = POSITIVE (finger present)")
    print("[INFO] Right-click = NEGATIVE (no finger / background)")
    print("[INFO] Press 's' to save dataset, 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Optional: flip horizontally
        frame = cv2.flip(frame, 1)

        last_frame = frame.copy()

        # Draw a crosshair so you have a sense of the center
        H, W = frame.shape[:2]
        cv2.line(frame, (W//2, 0), (W//2, H), (0, 255, 0), 1)
        cv2.line(frame, (0, H//2), (W, H//2), (0, 255, 0), 1)

        cv2.putText(frame, "L-click: POS | R-click: NEG | s: save | q: quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)

        cv2.imshow("Collect finger data (click)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if len(X) == 0:
                print("[WARN] No samples collected; skipping save.")
            else:
                X_arr = np.vstack(X)   # shape (N, D)
                y_arr = np.array(y)
                dump((X_arr, y_arr), OUTPUT_DATASET_PATH)
                print(f"[SAVED] Dataset with shape X={X_arr.shape}, y={y_arr.shape} "
                      f"saved to {OUTPUT_DATASET_PATH}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
