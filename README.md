# Body Detection App

This is a simple Python application for detecting human bodies in images using the Mediapipe library. The app allows you to select an input folder containing image frames and detect full body, top body, or bottom body in each image.

## Installation

1. Clone the repository to your local machine:

2. Select an input folder containing image frames (the folder should contain images with filenames starting with "frame" and ending with ".jpg").

3. Choose the type of body detection (Full Body, Top Body, or Bottom Body) using the radio buttons.

4. Click the "Detect Bodies" button to start the detection process.

5. The app will process each image and save the detected body images in the same input folder with filenames prefixed as "fb_" (for Full Body), "tb_" (for Top Body), or "bb_" (for Bottom Body).

6. The detected images will also be displayed in the GUI.

7. The status bar will show messages regarding the detection process.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Requirements

The following packages are required to run the app. You can install them using `pip`:

opencv-python==4.5.3.56
PySimpleGUI==4.58.2
Pillow==8.3.2
mediapipe==0.8.10
tqdm==4.62.1



