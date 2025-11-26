# GSpiel - Dual-Handed Game Gesture Controller

A computer vision program that utilizes machine learning to use user-generated hand gestures converting them to virtual keystrokes through their webcam. It is a self-contained program that runs locally on your device meaning there are no network latency issues.

##Features

- Real time gesture capturing
- Dual-Handed controls
- Personalized model creation (unique model for each user)
- User interface for gesture sampling and gesture capturing for gameplay

##Setup

###Prerequisites (Hardware and Software)

-Python version 3.9 (specifically)
-Functioning webcam

###Installation

1. Clone the repository

2. Install dependencies
```bash
pip install -r requirements.txt

pip install pydirectinput
```

##Usage

###Camera Testing

Run:
```bash
python camera_test.py
```

###Data collection and model training

To train the model, run:
```bash
python quick_custom_trainer.py
```

Follow its instrutions in the terminal to capture the images for each gesture. Press the numbers 0-9 to switch between the different gestures. Move the .h5 file and the .pkl file to the models folder(DO NOT FORGET THIS). After you're done, run:
```bash
python test_custom_model.py
```

This will open a window where you can test whether different gestures can be recognized through your webcam.

###Running the Program

After the above steps, if everything is in order, run:
```bash
python gspiel_dual_hand_controller.py
```

##Limitations

This project was tested with simpler games. Specifically Sonic the Hedgehog 2(1992) and an emulated version of the original Super Mario Bros.(1985)
It is not compatible with more complex games as the control scheme cannot handle that many actions to perform.
Kindly test with a relatively simple game.

##Project Structure

```
GSPIEL/
├── scripts/                    # Houses the model files and metadata
│   ├── camera_test.py            # Testing camera functionality
│   ├── gspiel_dual_hand_controller.py     # Gesture controller program
│   ├── quick_custom_trainer.py     # Trains user-captured images to create model and metadata
│   └── test_custom_model.py         # Tests gesture recognition through webcam
└── models              # Houses the .h5 and .pkl files for the controller and test files to locate.
```

##License

MIT License

##Contributors

- GSPIEL Development Team


