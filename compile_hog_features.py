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

OUTPUT_DATASET_PATH = "data/hog_11k_dataset.joblib"

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


def main():
    global last_frame, X, y

    # Optional: create data folder if not present
    data_dir = os.path.dirname(OUTPUT_DATASET_PATH) or "."
    os.makedirs(data_dir, exist_ok=True)

    IMAGE_PATH = "Dataset_training/raw_data/PositiveSamples"
    IMAGES = sorted(os.listdir(IMAGE_PATH))

    for file in IMAGES:
        if file.endswith(".jpg"):
            path = os.path.join(IMAGE_PATH, file)
            # print("Reading:", path)
            frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            feat = compute_hog(frame, hog)
            X.append(feat)
            y.append(1) # all positive

    IMAGE_PATH = "Dataset_training/raw_data/NegativeSamples"
    IMAGES = sorted(os.listdir(IMAGE_PATH))

    for file in IMAGES:
        if file.endswith(".JPEG"):
            path = os.path.join(IMAGE_PATH, file)
            # print("Reading:", path)
            frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            feat = compute_hog(frame, hog)
            X.append(feat)
            y.append(0) #all negative

    X_arr = np.vstack(X)   # shape (N, D)
    y_arr = np.array(y)

    dump((X_arr, y_arr), OUTPUT_DATASET_PATH)
    print("done")


    # while True:
    

    #     # # Optional: flip horizontally
    #     # frame = cv2.flip(frame, 1)

    #     last_frame = frame.copy()

    #     # # Draw a crosshair so you have a sense of the center
    #     # H, W = frame.shape[:2]
    #     # cv2.line(frame, (W//2, 0), (W//2, H), (0, 255, 0), 1)
    #     # cv2.line(frame, (0, H//2), (W, H//2), (0, 255, 0), 1)

    #     # cv2.putText(frame, "L-click: POS | R-click: NEG | s: save | q: quit",
    #     #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
    #     #             (0, 255, 0), 2)

    #     # cv2.imshow("Collect finger data (click)", frame)

    #     key = cv2.waitKey(1) & 0xFF
    #     if key == ord('q'):
    #         break
    #     elif key == ord('s'):
    #         if len(X) == 0:
    #             print("[WARN] No samples collected; skipping save.")
    #         else:
    #             X_arr = np.vstack(X)   # shape (N, D)
    #             y_arr = np.array(y)
    #             dump((X_arr, y_arr), OUTPUT_DATASET_PATH)
    #             print(f"[SAVED] Dataset with shape X={X_arr.shape}, y={y_arr.shape} "
    #                   f"saved to {OUTPUT_DATASET_PATH}")



if __name__ == "__main__":
    main()
