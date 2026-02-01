# THIS IS PURELY TO TEST IF THE MODEL CAN RECOGNIZE YOUR GESTURES. THIS IS NOT THE CONTROLLER FILE
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import os
import time

class CustomModelTester:
    def __init__(self, confidence_threshold=0.7):
        self.confidence_threshold = confidence_threshold
        
        # Load custom model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
       
        models_dir = os.path.join(parent_dir, 'models')
        
        model_path = os.path.join(models_dir, 'custom_gesture_model.h5')
        metadata_path = os.path.join(models_dir, 'custom_gesture_metadata.pkl')
        
        
        print(f" Looking for model at: {model_path}")
        print(f" Model exists: {os.path.exists(model_path)}")
        print(f" Looking for metadata at: {metadata_path}")
        print(f" Metadata exists: {os.path.exists(metadata_path)}")
        
        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            print(" Custom model or metadata not found! Run quick_custom_train.py first")
            print(" Check that files exist in the models folder")
            self.model = None
            self.metadata = None
            return
        
        try:
            self.model = tf.keras.models.load_model(model_path)
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            print("✅ Custom model loaded! (Trained on YOUR data)")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
            self.metadata = None
            return
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2)
        
    def preprocess_hand_roi(self, hand_roi):
        if hand_roi is None or hand_roi.size == 0:
            return None
            
        img = cv2.resize(hand_roi, (64, 64))
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)
        return img
    
    def get_hand_roi(self, image, hand_landmarks):
        h, w = image.shape[:2]
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        
        padding = 25
        x_min = max(0, int(min(x_coords) - padding))
        x_max = min(w, int(max(x_coords) + padding))
        y_min = max(0, int(min(y_coords) - padding))
        y_max = min(h, int(max(y_coords) + padding))
        
        hand_roi = image[y_min:y_max, x_min:x_max]
        return hand_roi if hand_roi.size > 0 else None
    
    def run(self):
        if self.model is None or self.metadata is None:
            print("🚫 Cannot run test - model not loaded properly")
            return
        
        cap = cv2.VideoCapture(0)
        
        print("🎮 CUSTOM MODEL TEST")
        print("💡 This model was trained on YOUR hand gestures")
        print("If models are not found, move the .pkl and .h5 files to the models folder")
        print("⏹️  Press 'q' to quit")
        
        while True:
            success, image = cap.read()
            if not success:
                break
            
            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = self.hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    hand_roi = self.get_hand_roi(image, hand_landmarks)
                    if hand_roi is not None:
                        processed_img = self.preprocess_hand_roi(hand_roi)
                        if processed_img is not None:
                            predictions = self.model.predict(processed_img, verbose=0)
                            predicted_class = np.argmax(predictions[0])
                            confidence = np.max(predictions[0])
                            
                            gesture_name = self.metadata['gesture_mapping'][predicted_class]
                            control = self.metadata['sonic_controls'][gesture_name]
                            
                            color = (0, 255, 0) if confidence > self.confidence_threshold else (0, 165, 255)
                            cv2.putText(image, f"{gesture_name} → {control} ({confidence:.2f})", 
                                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            
                            if confidence > self.confidence_threshold:
                                print(f"🎮 {gesture_name} → {control} (conf: {confidence:.2f})")
            
            cv2.imshow('Custom Model Test', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tester = CustomModelTester(confidence_threshold=0.7)
    tester.run()