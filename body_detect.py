# Detects human forms from an image and additionally save them as separate files. Use the Frame_Extractor project to extract frames from a video file.
######################################################################################################################################################
import io
import os
import cv2
import PySimpleGUI as sg
from PIL import Image
from tqdm import tqdm
import mediapipe as mp

# Suppress warning caused by google.protobuf
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

file_types = [("JPEG (*.jpg)", "*.jpg"),
              ("All files (*.*)", "*.*")]

class BodyDetectionApp:
    def __init__(self):
        self.input_folder = ""
        self.body_detection_choice = "Full Body"

        # Define the layout for the GUI
        layout = [
            [sg.Text("Select Input Folder: "), sg.Input(size=(25, 1), key="-FOLDER-"), sg.FolderBrowse()],
            [sg.Radio("Full Body", "body_detection", default=True, key="-FULL-", enable_events=True),
             sg.Radio("Top Body", "body_detection", key="-TOP-", enable_events=True),
             sg.Radio("Bottom Body", "body_detection", key="-BOTTOM-", enable_events=True)],
            [sg.Text("Images found: "), sg.Text(" ", key="-IMG-")],
            [sg.Text("Total Full Body Files: "), sg.Text("0", key="-TOTAL-FB-")],
            [sg.Text("Total Top Body Files: "), sg.Text("0", key="-TOTAL-TOP-")],
            [sg.Text("Total Bottom Body Files: "), sg.Text("0", key="-TOTAL-BOTTOM-")],
            [sg.Button("Detect Bodies", key="-DETECT-")],
            [sg.ProgressBar(100, orientation='h', size=(20, 20), key='-PROGRESS-', visible=False)],
            [sg.Image(key="-IMAGE-", size=(600, 600), background_color="white", pad=(20, 20))],
            [sg.Button("Exit")],
            [sg.StatusBar("Ready", size=(40, 1), key="-STATUS-")]
        ]

        # Create the GUI window
        self.window = sg.Window("Body Detection App", layout, finalize=True)

        # Initialize mediapipe
        self.mp_holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def run(self):
        while True:
            event, values = self.window.read()
            if event == sg.WINDOW_CLOSED or event == "Exit":
                break
            elif event in ["-FULL-", "-TOP-", "-BOTTOM-"]:
                self.body_detection_choice = values[event]
            elif event == "-DETECT-":
                self.detect_bodies()

        self.window.close()

    def detect_bodies(self):
        self.input_folder = self.window["-FOLDER-"].get()

        if not self.input_folder:
            self.window["-STATUS-"].update("Please select an input folder.")
            return

        # Use os.scandir to search only for files starting with "frame" and ending with ".jpg"
        image_files = [entry.name for entry in os.scandir(self.input_folder) if entry.name.startswith("frame") and entry.name.endswith(".jpg")]

        if len(image_files) == 0:
            self.window["-STATUS-"].update("Unable to find any frames. Frame images start with 'frame_xxx.jpg'")
            self.window["-IMG-"].update("None")
            return
        else:
            self.window["-IMG-"].update(str(len(image_files)))

        # Reset the progress bar
        self.window["-PROGRESS-"].update(visible=True)
        self.window["-PROGRESS-"].update(0)
        self.window["-PROGRESS-"].update(max=len(image_files))

        # Set output folder to input folder if not specified
        output_folder = self.input_folder

        total_fb_files = 0
        total_top_files = 0
        total_bottom_files = 0

        for i, image_file in enumerate(tqdm(image_files, disable=True)):
            image_path = os.path.join(self.input_folder, image_file)
            image = cv2.imread(image_path)
            image_height, image_width, _ = image.shape

            # Convert the image from BGR to RGB (mediapipe requires RGB format)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Make the detection
            results = self.mp_holistic.process(image_rgb)

            # Check if the human body was detected
            if results.pose_landmarks:
                # Get the y-coordinate of the landmarks
                landmarks_y = [landmark.y for landmark in results.pose_landmarks.landmark]

                # Decide the type of body based on the user's choice
                if self.body_detection_choice == "Full Body":
                    is_full_body = True
                elif self.body_detection_choice == "Top Body":
                    is_full_body = all(y < 0.45 * image_height for y in landmarks_y[0:22])
                else:  # Bottom Body
                    is_full_body = all(y > 0.7 * image_height for y in landmarks_y[23:33])

                # Decide the prefix for the output file
                if is_full_body:
                    total_fb_files += 1
                    output_file = "fb_" + image_file
                else:
                    total_bottom_files += 1
                    output_file = "tb_" + image_file

                total_top_files += 1 if not is_full_body else 0

                # Save the image with the appropriate prefix in the output folder
                output_path = os.path.join(output_folder, output_file)
                cv2.imwrite(output_path, image)

                # Display the processed image in the GUI
                self.display_image(output_path)

            # Update the progress bar
            self.window["-PROGRESS-"].update(i + 1)

        self.window["-PROGRESS-"].update(visible=False)
        self.window["-STATUS-"].update("Body detection process completed successfully.")
        self.window["-TOTAL-FB-"].update(total_fb_files)
        self.window["-TOTAL-TOP-"].update(total_top_files)
        self.window["-TOTAL-BOTTOM-"].update(total_bottom_files)
        sg.popup("Body Detection", "Body detection process completed successfully.")


    def display_image(self, image_path):
        try:
            image = Image.open(image_path)

            # Calculate scaling factor to fit the image in the image element
            window_width, window_height = self.window["-IMAGE-"].Widget.winfo_width(), self.window["-IMAGE-"].Widget.winfo_height()
            image_width, image_height = image.size
            scaling_factor = min(window_width / image_width, window_height / image_height)

            new_width = int(image_width * scaling_factor)
            new_height = int(image_height * scaling_factor)

            image = image.resize((new_width, new_height), Image.ANTIALIAS)

            bio = io.BytesIO()
            image.save(bio, format="PNG")
            self.window["-IMAGE-"].update(data=bio.getvalue())

        except Exception as e:
            self.window["-STATUS-"].update(f"Failed to display image: {e}")

if __name__ == "__main__":
    app = BodyDetectionApp()
    app.run()
