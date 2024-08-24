import cv2

def test_camera(index):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"Camera initialized at index {index}")
        # Try to capture a frame to ensure it's working
        ret, frame = cap.read()
        if ret:
            print(f"Captured a frame from camera at index {index}")
            cv2.imshow(f"Camera {index}", frame)
            cv2.waitKey(0)
        else:
            print(f"Failed to capture frame from camera at index {index}")
        cap.release()
    else:
        print(f"Failed to initialize camera at index {index}")

# Test indices from 0 to 4 (or a larger range if needed)
for i in range(5):
    test_camera(i)

cv2.destroyAllWindows()
