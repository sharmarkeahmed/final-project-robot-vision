"""
hog_finger_data.py

Used to collect manual finger/hand samples for HOG based-detection via a user's camera

This script lets the user label:
    - positive samples: where a finger/hand is present
    - negative samples: background/regions where no finger is present

For each sample taken, a HOG descriptor is computed on a fixed-size patches (64x64 pixels) and stored as a feature
vector. After all samples are taken, the features and labels are saved to 'data/hog_finger_click_dataset.joblib' as a
tuple (features, labels), where:
    - features is a NumPy array of shape (N, D) containing HOG features
    - labels is a NumPy array of shape (N,) containing labels (1 for positive, 0 for negative)

Note that if an existing 'data/hog_finger_click_dataset.joblib' file already exists, the user will be asked if they want
to load and append new samples to it, or override it. In either case, a backup copy of the old file is saved in
data/hog_finger_click_dataset_old.joblib'

To load this dataset in another Python file, perform the following:
    from joblib import load
    features, labels = load("data/hog_finger_click_dataset.joblib")

This file is intended to be used together with 'train_finger_svm.py', which trains a linear SVM classifier on the
sampled HoG features.
"""

import cv2
import numpy as np
import os
import shutil
from joblib import dump, load  # joblib is used to dump and load python objects to be
# used later by other files
from hog_utils import create_hog, compute_hog

# Select input mode below:
# "mouse"    -> Lft click / Right click to label positive/negative samples (better for mouse users)
# "keyboard" -> Use 'l' and 'r' keys to label positive/negative samples (better for laptops)
INPUT_MODE = "keyboard"

# HOG configuration parameters
HOG_WIN_SIZE = (64, 64)  # fixed patch size taken
BLOCK_SIZE = (16, 16)
BLOCK_STRIDE = (8, 8)
CELL_SIZE = (8, 8)
NBINS = 9

OUTPUT_DATASET_PATH = "data/hog_finger_click_dataset.joblib"
BACKUP_DATASET_PATH = "data/hog_finger_click_dataset_old.joblib"

last_frame = None  # current frame
X = []  # list of feature vectors
Y = []  # list of labels (1=positive, 0=negative)

hog = create_hog(
    win_size=HOG_WIN_SIZE,
    block_size=BLOCK_SIZE,
    block_stride=BLOCK_STRIDE,
    cell_size=CELL_SIZE,
    nbins=NBINS
)


def add_sample_at(x, y_img, label):
    """
    obtain a patch centered at (x, y_img) from last_frame,
    compute HOG, and add it to X (feature) and Y (label) dataset
    """
    global last_frame, X, Y

    if last_frame is None:
        return

    h, w = last_frame.shape[:2]
    win_w, win_h = HOG_WIN_SIZE

    # top left of patch
    x1 = int(x - win_w // 2)
    y1 = int(y_img - win_h // 2)
    x2 = x1 + win_w
    y2 = y1 + win_h

    # ensure patch is within frame
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        print("[WARN] Click too close to edge, patch would go out of frame. Ignoring.")
        return

    patch = last_frame[y1:y2, x1:x2]

    # compute HOG (no image saved)
    feat = compute_hog(patch, hog)
    X.append(feat) # X stores the computed hog feature,
    # Y stores the label (positive/negative)
    Y.append(label)

    label_str = "POSITIVE" if label == 1 else "NEGATIVE"
    y_arr = np.array(Y)
    n_pos = int((y_arr == 1).sum())
    n_neg = int((y_arr == 0).sum())
    print(f"[INFO] Saved {label_str} sample at ({x}, {y_img}). "
          f"Total: pos={n_pos}, neg={n_neg}, total={len(Y)}")


def mouse_callback(event, x, y_img, flags, param):
    """
    Mouse labeling only used when INPUT_MODE == "mouse".

    Left-click  -> positive sample (finger present at/near click)
    Right-click -> negative sample
    """
    if INPUT_MODE != "mouse":
        return

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
    global X, Y

    if not os.path.exists(OUTPUT_DATASET_PATH):
        print("[INFO] No existing dataset found. Starting fresh.")
        return

    ans = input(f"[PROMPT] Existing dataset found at '{OUTPUT_DATASET_PATH}'. "
                "Load and append new samples? (y/n): ").strip().lower()

    if ans == "y":
        x_arr, y_arr = load(OUTPUT_DATASET_PATH)
        print(f"[INFO] Loaded existing dataset: X={x_arr.shape}, y={y_arr.shape}")
        # Convert to lists to keep appending
        X = [row for row in x_arr]
        Y = list(y_arr)
        print(f"[INFO] Continuing from existing dataset. Current total samples: {len(Y)}")
    else:
        print("[INFO] Starting fresh dataset (existing file will be overwritten on save).")


def backup_existing_dataset():
    """
    If OUTPUT_DATASET_PATH exists, copy it to BACKUP_DATASET_PATH.
    This gives us a single backup from the last save.
    """
    if os.path.exists(OUTPUT_DATASET_PATH):
        try:
            shutil.copyfile(OUTPUT_DATASET_PATH, BACKUP_DATASET_PATH)
            print(f"[BACKUP] Existing dataset backed up to {BACKUP_DATASET_PATH}")
        except Exception as e:
            print(f"[WARN] Could not backup existing dataset: {e}")


def main():
    global last_frame, X, Y

    # optional: create data folder if not present
    data_dir = os.path.dirname(OUTPUT_DATASET_PATH) or "."
    os.makedirs(data_dir, exist_ok=True)

    # Ask whether to load existing data
    maybe_load_existing_dataset()

    cap = cv2.VideoCapture(0)  # adjust index if needed

    window_name = "Collect finger data (click)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    if INPUT_MODE == "mouse":
        print("[INFO] MODE: MOUSE")
        print("[INFO] Left-click = POSITIVE (finger present)")
        print("[INFO] Right-click = NEGATIVE (no finger / background)")
    else:
        print("[INFO] MODE: KEYBOARD")
        print("[INFO] 'l' = POSITIVE (finger present) at center crosshair")
        print("[INFO] 'r' = NEGATIVE (no finger / background) at center crosshair")

    print("[INFO] Press 's' to save dataset, 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # flip horizontally
        frame = cv2.flip(frame, 1)

        last_frame = frame.copy()

        # draw a crosshair so we have a sense of the center
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)
        cv2.line(frame, (0, h // 2), (w, h // 2), (0, 255, 0), 1)

        # instruction text depends on mode
        if INPUT_MODE == "mouse":
            instr = "L-click: POS | R-click: NEG | s: save | q: quit"
        else:
            instr = "'l': POS (center) | 'r': NEG (center) | s: save | q: quit"

        cv2.putText(frame, instr,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if len(X) == 0:
                print("[WARN] No samples collected; skipping save.")
            else:
                # always backup existing dataset (if any) before overwriting
                backup_existing_dataset()

                x_arr = np.vstack(X)  # shape (N, D)
                y_arr = np.array(Y)
                dump((x_arr, y_arr), OUTPUT_DATASET_PATH)
                print(f"[SAVED] Dataset with shape X={x_arr.shape}, y={y_arr.shape} "
                      f"saved to {OUTPUT_DATASET_PATH}")
        elif INPUT_MODE == "keyboard":
            # label at center crosshair using keys
            if key == ord('l'):
                add_sample_at(cx, cy, label=1)
            elif key == ord('r'):
                add_sample_at(cx, cy, label=0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
