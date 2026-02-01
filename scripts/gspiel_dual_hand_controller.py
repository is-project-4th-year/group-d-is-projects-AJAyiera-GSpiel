# IF IT CANNOT FIND THE MODEL OR DATA AFTER TESTING, MOVE THE H5 FILE AND THE PKL FILE TO THE MODELS FOLDER
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import os
import time
import pydirectinput

class CustomModelTester:
    def __init__(self, confidence_threshold=0.7):
        #YOU CAN CHANGE THE COOLDOWN HERE (0 IS RECOMMENDED FOR OPTIMUM EXPERIENCE, BUT FEEL FREE TO EXPERIMENT)
        self.confidence_threshold = confidence_threshold
        self.last_action_time = 0
        self.action_cooldown = 0
        self.last_direction_time = 0
        self.direction_cooldown = 0
        self.current_direction = None
        
        # Load custom model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        models_dir = os.path.join(parent_dir, 'models')
        
        model_path = os.path.join(models_dir, 'custom_gesture_model.h5')
        metadata_path = os.path.join(models_dir, 'custom_gesture_metadata.pkl')
        
        print(f" Looking for model at: {model_path}")
        print(f" Model exists: {os.path.exists(model_path)}")
        
        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            print(" Custom model or metadata not found! Run quick_custom_trainer.py first")
            self.model = None
            self.metadata = None
            return
        
        try:
            self.model = tf.keras.models.load_model(model_path)
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            print("✅ Custom model loaded! (Trained on YOUR data)")
            print(f"📊 Available gestures: {list(self.metadata['gesture_mapping'].values())}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
            self.metadata = None
            return
        
        # TWO-HAND CONTROL SCHEME 
# =============================================================================
#  CUSTOM CONTROL MAPPING [You can edit here to change controls :)]
# =============================================================================

# REMEMBER!!!!! THE SAME GESTURES DO DIFFERENT THINGS ON EACH HAND

# LEFT HAND CONTROLS (Directional):
#   - fist: left arrow
#   - palm: right arrow  
#   - thumbs_up: up arrow
#   - peace_sign: down arrow
#
# RIGHT HAND CONTROLS (Actions):
#   - fist: A key
#   - palm: S key
#   - thumbs_up: D key  
#   - peace_sign: Enter key
#
# TO CHANGE CONTROLS:
#   1. Edit the keystroke values below (e.g., change 'left' to 'a')
#   2. Make sure game window is active when testing
#   3. Common keys: 'a','s','d','w','space','enter','ctrl','shift'
#   4. Arrow keys: 'left','right','up','down'

#You can add a comment to remember what you changed it to

# =============================================================================
        self.left_hand_controls = {
            'fist': 'left', #Comment controls here
            'palm': 'right',  #Comment controls here
            'thumbs_up': 'up', #Comment controls here
            'peace': 'down', #Comment controls here
        }
        
        self.right_hand_controls = {
            'fist': 'a', #Comment controls here
            'palm': 's',      #Comment controls here
            'thumbs_up': 'd', #Comment controls here
            'peace': 'enter', #Comment controls here
        }
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
    def trigger_keystroke(self, keystroke, hand_type):
        """Simulate keystroke based on gesture and hand type"""
        current_time = time.time()
        
        print(f"🔑 Attempting keystroke: '{keystroke}' for {hand_type} hand")
        
        try:
            if hand_type == "left":
                # Directional controls
                if current_time - self.last_direction_time < self.direction_cooldown:
                    print("⏳ Direction cooldown active")
                    return
                
                # Release previous direction if changed
                if self.current_direction and self.current_direction != keystroke:
                    print(f"🔄 Releasing previous direction: '{self.current_direction}'")
                    pydirectinput.keyUp(self.current_direction)
                    self.current_direction = None
                
                # Press new direction
                print(f"⬇️ Pressing direction: '{keystroke}'")
                pydirectinput.keyDown(keystroke)
                self.current_direction = keystroke
                self.last_direction_time = current_time
                
            else:  # right hand
                # Action buttons - single press
                if current_time - self.last_action_time < self.action_cooldown:
                    print("⏳ Action cooldown active")
                    return
                
                print(f"🎯 Pressing action: '{keystroke}'")
                pydirectinput.press(keystroke)
                self.last_action_time = current_time
                
            print(f"✅ SUCCESS: Keystroke '{keystroke}' executed")
                
        except Exception as e:
            print(f"❌ ERROR in keystroke: {e}")
    
    def release_direction(self):
        """Release any held directional keys"""
        if self.current_direction:
            print(f"🔼 Releasing direction: '{self.current_direction}'")
            pydirectinput.keyUp(self.current_direction)
            self.current_direction = None
    
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
    
    def determine_handedness(self, hand_landmarks, image_width):
        """Is the hand used left or right"""
        wrist_x = hand_landmarks.landmark[0].x * image_width
        
        if wrist_x < image_width / 2:
            return "left"
        else:
            return "right"
    
    def run(self):
        if self.model is None or self.metadata is None:
            print("🚫 Cannot run test - model not loaded properly")
            return
        
        cap = cv2.VideoCapture(0)
        
        print("GSpiel DUAL-HAND GESTURE CONTROLLER")
        print("⏹️  Press 'q' to quit")
        print("🔧 Press 'r' to release all keys (emergency)")
       #Emergency release stops all simulated keystrokes
        

        print("\n Testing keystrokes...")
        try:
            pydirectinput.press('a')
            print("✅ Keystroke test passed - 'a' key should have been pressed")
        except Exception as e:
            print(f"❌ Keystroke test failed: {e}")
        
        frame_count = 0
        hands_detected_count = 0
        
        while True:
            success, image = cap.read()
            if not success:
                print("❌ Failed to read from camera")
                break
            
            frame_count += 1
            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_height, image_width, _ = image.shape
            
            results = self.hands.process(rgb_image)
            
            # Reset if no hands detected
            if not results.multi_hand_landmarks:
                if frame_count % 30 == 0:  # Print every ~1 second
                    print(" No hands detected...")
                self.release_direction()
            
            if results.multi_hand_landmarks:
                hands_detected_count += 1
                print(f"👐 Hands detected: {len(results.multi_hand_landmarks)}")
                
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    hand_type = self.determine_handedness(hand_landmarks, image_width)
                    
                    landmark_color = (255, 0, 0) if hand_type == "left" else (0, 0, 255)
                    mp.solutions.drawing_utils.draw_landmarks(
                        image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=landmark_color, thickness=3))
                    
                    wrist = hand_landmarks.landmark[0]
                    label_x = int(wrist.x * image_width)
                    label_y = int(wrist.y * image_height) - 10
                    cv2.putText(image, f"{hand_type.upper()} HAND", 
                               (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, landmark_color, 2)
                    
                    hand_roi = self.get_hand_roi(image, hand_landmarks)
                    if hand_roi is not None:
                        processed_img = self.preprocess_hand_roi(hand_roi)
                        if processed_img is not None:
                            predictions = self.model.predict(processed_img, verbose=0)
                            predicted_class = np.argmax(predictions[0])
                            confidence = np.max(predictions[0])
                            
                            gesture_name = self.metadata['gesture_mapping'][predicted_class]
                            
                            if hand_type == "left":
                                control_mapping = self.left_hand_controls
                            else:
                                control_mapping = self.right_hand_controls
                            
                            keystroke = control_mapping.get(gesture_name)
                            
                            if keystroke:
                                color = (0, 255, 0) if confidence > self.confidence_threshold else (0, 165, 255)
                                
                                if hand_type == "left":
                                    cv2.putText(image, f"LEFT: {gesture_name} → {keystroke} ({confidence:.2f})", 
                                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                else:
                                    cv2.putText(image, f"RIGHT: {gesture_name} → {keystroke} ({confidence:.2f})", 
                                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                
                                if confidence > self.confidence_threshold:
                                    print(f"🎯 {hand_type.upper()} HAND: {gesture_name} → '{keystroke}' (conf: {confidence:.2f})")
                                    self.trigger_keystroke(keystroke, hand_type)
                            else:
                                print(f"⚠️  No keystroke mapping for {gesture_name} on {hand_type} hand")
            
            cv2.imshow('GSpiel Dual-Hand Gesture Controller', image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.release_direction()
                print("(EM) Emergency release - all keys released")
        
        self.release_direction()
        cap.release()
        cv2.destroyAllWindows()
        print(f"📊 Session stats: {hands_detected_count} hand detections in {frame_count} frames")
        #Tracking how many gestures were detected ^
if __name__ == "__main__":
    gspiel = CustomModelTester(confidence_threshold=0.7)
    gspiel.run()