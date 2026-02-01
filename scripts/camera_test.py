# scripts/camera_test.py
import cv2

def test_camera():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    print("📷 Testing camera...")
    print("⏹️  Press 'q' to quit")
    
    while True:
        success, frame = cap.read()
        if not success:
            print("❌ Failed to read from camera")
            break
        
        cv2.imshow('Camera Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Camera test complete")

if __name__ == "__main__":
    test_camera()