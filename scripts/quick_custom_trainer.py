# RUN THIS TO TRAIN YOUR MODEL. 50 IMAGES PER GESTURE. CAPTURE IN DIFFERENT POSITIONS IF POSSIBLE
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import pickle
import time
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

class QuickCustomTrainer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2)
        
        # Core gestures we actually need
        self.gestures = {
            0: 'palm',       # RIGHT
            1: 'l',          # LEFT
            2: 'fist',       # CROUCH
            3: 'thumb',      # LOOK UP
            4: 'peace',      # JUMP
            5: 'index',      # SPIN
            6: 'ok',         # ACTION
            7: 'palm_moved', # RUN
            8: 'c_shape',    # PAUSE
            9: 'down'        # ROLL
        }
        
        self.collected_data = {i: [] for i in range(10)}
        
    def collect_training_data(self, samples_per_gesture=50): #CHANGE VALUE IF YOU WOULD WANT MORE OR LESS SAMPLES AS WELL AS AT THE BOTTOM OF THE FILE.
        """Quick data collection with YOUR camera and hands"""
        cap = cv2.VideoCapture(0)
        
        print("🎯 QUICK CUSTOM TRAINING")
        print(f"📝 We'll collect {samples_per_gesture} samples for each gesture")
        print("💡 Use YOUR camera, lighting, and hand style")
        print("⏰ Takes about 5-8 minutes total")
        
        current_gesture = 0
        start_time = time.time()
        
        while True:
            success, image = cap.read()
            if not success:
                break
            
            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = self.hands.process(rgb_image)
            
            # Display current state
            gesture_name = self.gestures[current_gesture]
            cv2.putText(image, f"Gesture: {gesture_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Samples: {len(self.collected_data[current_gesture])}/{samples_per_gesture}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, "0-9: Change gesture, SPACE: Capture", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(image, "'s': Save & Train, 'q': Quit", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            cv2.imshow('Quick Custom Training', image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                total_samples = sum(len(samples) for samples in self.collected_data.values())
                if total_samples >= 150:
                    self.train_custom_model()
                    break
                else:
                    print(f"❌ Need more data! Currently have {total_samples} samples")
            elif ord('0') <= key <= ord('9'):
                current_gesture = key - ord('0')
                print(f"📝 Switched to: {self.gestures[current_gesture]}")
            elif key == ord(' ') and results.multi_hand_landmarks:
                hand_roi = self.get_hand_roi(image, results.multi_hand_landmarks[0])
                if hand_roi is not None:
                    current_samples = self.collected_data[current_gesture]
                    if len(current_samples) < samples_per_gesture:
                        processed = self.preprocess_image(hand_roi)
                        self.collected_data[current_gesture].append(processed)
                        print(f"✅ {self.gestures[current_gesture]}: {len(current_samples)}/{samples_per_gesture}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        print(f"⏱️  Data collection completed in {total_time:.1f} seconds")
    
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
    
    def preprocess_image(self, hand_roi):
        img = cv2.resize(hand_roi, (64, 64))
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype('float32') / 255.0
        return img
    
    def evaluate_model(self, model, X_test, y_test):
        """Comprehensive model evaluation with multiple metrics"""
        print("\n" + "="*50)
        print("📊 COMPREHENSIVE MODEL EVALUATION")
        print("="*50)
        
        # Get predictions
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_test)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # The overall metrics
        print(f"🎯 Overall Metrics:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        
        print(f"\n📈 Detailed Classification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=[self.gestures[i] for i in range(10)],
                                  zero_division=0))
        
        print(f"🎭 Per-Class Metrics:")
        class_precision = precision_score(y_test, y_pred, average=None, zero_division=0)
        class_recall = recall_score(y_test, y_pred, average=None, zero_division=0)
        class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
        
        for i in range(10):
            print(f"   {self.gestures[i]:12} - Precision: {class_precision[i]:.3f}, "
                  f"Recall: {class_recall[i]:.3f}, F1: {class_f1[i]:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n🔄 Confusion Matrix:")
        print(cm)
        
        # Plot confusion matrix (optional - requires matplotlib)
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=[self.gestures[i] for i in range(10)],
                       yticklabels=[self.gestures[i] for i in range(10)])
            plt.title('Confusion Matrix - Custom Gesture Model')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("📊 Confusion matrix saved as 'confusion_matrix.png'")
        except Exception as e:
            print(f"⚠️  Could not save confusion matrix plot: {e}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': classification_report(y_test, y_pred, 
                                                         target_names=[self.gestures[i] for i in range(10)],
                                                         output_dict=True,
                                                         zero_division=0)
        }
    
    def train_custom_model(self):
        """Train a simple model on YOUR data with comprehensive evaluation"""
        print("🚀 Training custom model on your data...")
        
        # Prepare data
        X_train = []
        y_train = []
        
        for gesture_idx, samples in self.collected_data.items():
            for sample in samples:
                X_train.append(sample)
                y_train.append(gesture_idx)
        
        if len(X_train) == 0:
            print("❌ No training data collected!")
            return
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_train = np.expand_dims(X_train, axis=-1)
        
        print(f"📊 Training on {len(X_train)} samples from your camera")
        
        # Split data for training and evaluation
        from sklearn.model_selection import train_test_split
        X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        print(f"   Training samples: {len(X_train_split)}")
        print(f"   Test samples: {len(X_test_split)}")
        
        
        model = models.Sequential([
            layers.Input(shape=(64, 64, 1)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Extended training
        print("⏱️  Training (70 epochs)...")
        history = model.fit(
            X_train_split, y_train_split,
            epochs=70,
            batch_size=16,
            validation_split=0.2,
            verbose=1
        )
        
        # Comprehensive evaluation
        evaluation_results = self.evaluate_model(model, X_test_split, y_test_split)
        
        # Save model
        model.save('custom_gesture_model.h5')
        
        # Create metadata with evaluation results
        metadata = {
            'gesture_mapping': {i: name for i, name in self.gestures.items()},
            'sonic_controls': {
                'palm': 'RIGHT',
                'l': 'LEFT',
                'fist': 'CROUCH',
                'thumb': 'LOOK UP', 
                'peace': 'JUMP',
                'index': 'SPIN',
                'ok': 'ACTION',
                'palm_moved': 'RUN',
                'c_shape': 'PAUSE',
                'down': 'ROLL'
            },
            'input_shape': (64, 64, 1),
            'model_type': 'custom_trained',
            'evaluation_metrics': {
                'overall_accuracy': float(evaluation_results['accuracy']),
                'overall_precision': float(evaluation_results['precision']),
                'overall_recall': float(evaluation_results['recall']),
                'overall_f1_score': float(evaluation_results['f1_score']),
            },
            'training_history': {
                'final_train_accuracy': float(history.history['accuracy'][-1]),
                'final_val_accuracy': float(history.history['val_accuracy'][-1] if 'val_accuracy' in history.history else 0),
                'final_train_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1] if 'val_loss' in history.history else 0),
            }
        }
        
        with open('custom_gesture_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"\n✅ Custom model trained and saved!")
        print(f"📈 Final Training Accuracy: {history.history['accuracy'][-1]:.3f}")
        print(f"🎯 Test Accuracy: {evaluation_results['accuracy']:.3f}")
        print(f"📊 F1-Score: {evaluation_results['f1_score']:.3f}")
        print("🎮 Ready to test with your custom model!")

if __name__ == "__main__":
    trainer = QuickCustomTrainer()
    trainer.collect_training_data(samples_per_gesture=50) #CHANGE VALUE LIKE ABOVE