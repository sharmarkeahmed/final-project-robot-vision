import cv2
import time
import numpy as np

fps_last_time = time.time()
fps_counter = 0
current_fps = 0


# def apply_kmeans(frame, K=8):
#     # Convert the frame to a 2D float32 array of pixels
#     Z = frame.reshape((-1, 3))
#     Z = np.float32(Z)

#     # K-means criteria and run
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#     ret, labels, centers = cv2.kmeans(
#         Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
#     )

#     # Convert centers back to uint8 and rebuild segmented image
#     centers = np.uint8(centers)
#     segmented = centers[labels.flatten()]
#     segmented = segmented.reshape(frame.shape)
#     return segmented


def _get_FPS():
    """
        Return FPS counter and the returned FPS is updated every second
    """
    global fps_counter
    global fps_last_time
    global current_fps

    now = time.time()
    # FPS Update
    fps_counter += 1
    # Update FPS counter value every second
    if now - fps_last_time >= 1.0:
        current_fps = fps_counter
        fps_counter = 0
        fps_last_time = now

    return current_fps


def startCamera(cam):
    """
    Start the camera of the system

    Arguments:
        cam: choice of camera in the system (defult 0)
    Return:
        The videofeed
    """

    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        raise RuntimeError("Error: Cannot open camera.")
    return cap


def readFrame(videostream):
    """
        Read the camera stream and returns a single frame
    """

    ret, frame = videostream.read()
    if not ret:
        raise RuntimeError("The frames cannot be read")
    return frame


def main():
    capture = startCamera(1)

    while True:
        frame = readFrame(capture)

        # Overlay FPS text
        cv2.putText(frame, f"FPS: {_get_FPS()}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        # Show camerafeed frames
        cv2.imshow("Camera feed", frame)

        # Quit on q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


# def main():
#     while True:


if __name__ == "__main__":
    main()