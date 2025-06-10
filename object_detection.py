import cv2
import numpy as np
import pyttsx3
import threading
import pytesseract
import mediapipe as mp
from ultralytics import YOLO
from tkinter import Tk, Label, Button, StringVar, OptionMenu, Canvas, font, Frame, messagebox
from PIL import Image, ImageTk

# Initialize Text-to-Speech (TTS)
engine = pyttsx3.init()
engine.setProperty("rate", 175)  # Adjust speaking speed

# Initialize Tkinter GUI
root = Tk()
root.title("Object Detection, Distance & OCR")
root.geometry("1200x800")
root.configure(bg="#2c3e50")  # Set background color

# Load background image
try:
    bg_image = Image.open("background.jpg")  # Replace with your image path
    bg_image = bg_image.resize((1200, 800), Image.LANCZOS)
    bg_image_tk = ImageTk.PhotoImage(bg_image)
except Exception as e:
    messagebox.showerror("Error", f"Failed to load background image: {e}")
    bg_image_tk = None

# Background Canvas
background_canvas = Canvas(root, width=1200, height=800, highlightthickness=0)
background_canvas.place(x=0, y=0)

if bg_image_tk:
    background_canvas.create_image(0, 0, image=bg_image_tk, anchor="nw")

# Variable to store selected language
selected_language = StringVar(root)
selected_language.set("English")  # Default language

# Function to set TTS language based on user input
def set_language():
    lang_name = selected_language.get()
    lang_map = {"English": "en", "French": "fr", "Spanish": "es", "Japanese": "ja"}
    return lang_map.get(lang_name, "en"), lang_name

# Function to set the TTS voice
def set_voice(lang_code, lang_name):
    voices = engine.getProperty('voices')
    for voice in voices:
        if lang_name.lower() in voice.name.lower():
            engine.setProperty('voice', voice.id)
            print(f"Voice set to {voice.name} for {lang_name}.")
            return
    engine.setProperty('voice', voices[0].id)  # Default to first English voice

# Function to speak in a separate thread
def speak(text):
    threading.Thread(target=lambda: engine.say(text) or engine.runAndWait(), daemon=True).start()

# Function to start the main application
def start_application():
    global lang_code, lang_name
    lang_code, lang_name = set_language()
    set_voice(lang_code, lang_name)
    speak(f"Welcome! Your selected language is {lang_name}.")
    threading.Thread(target=start_object_detection, daemon=True).start()

# Function to start object detection
def start_object_detection():
    global cap
    model = YOLO("yolov8m.pt")  # Load YOLO model
    cap = cv2.VideoCapture(0)  # Open webcam
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def estimate_distance(bbox_width, known_width=50, focal_length=500):
        return (known_width * focal_length) / bbox_width

    def preprocess_for_ocr(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return thresh

    def extract_text_from_region(frame, x1, y1, x2, y2):
        roi = frame[y1:y2, x1:x2]
        preprocessed_roi = preprocess_for_ocr(roi)
        text = pytesseract.image_to_string(preprocessed_roi, config="--psm 6").strip()
        return text

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        detected_objects = {}

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_hands = hands.process(rgb_frame)

        index_finger_x, index_finger_y = None, None
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                index_finger_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
                index_finger_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
                cv2.circle(frame, (index_finger_x, index_finger_y), 15, (255, 255, 0), -1)

        results = model(frame, conf=0.3)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls[0].item())]

                bbox_width = x2 - x1
                distance = estimate_distance(bbox_width)
                detected_objects[label] = int(distance)

                bbox_color = (255, 0, 0)
                text_color = (0, 255, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
                text = f"{label}: {int(distance)} cm"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

                if label in ["sign", "text", "board", "poster"] and index_finger_x and index_finger_y:
                    if x1 < index_finger_x < x2 and y1 < index_finger_y < y2:
                        text = extract_text_from_region(frame, x1, y1, x2, y2)
                        if text:
                            speak(f"Sign reads: {text}")

        if detected_objects:
            spoken_text = ", ".join([f"{obj} at {dist} cm" for obj, dist in detected_objects.items()])
            speak(f"Detected: {spoken_text}")

        cv2.imshow("Object Detection, Distance & OCR", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break  

    cap.release()
    cv2.destroyAllWindows()

# Create a frame for GUI elements
gui_frame = Frame(root, bg="#2c3e50", bd=5, relief="ridge")
gui_frame.place(relx=0.5, rely=0.1, anchor="n")

# Create a sub-frame for aligning elements in one row
button_frame = Frame(gui_frame, bg="#2c3e50")
button_frame.pack(pady=10)

button_font = font.Font(family="Helvetica", size=14, weight="bold")

# Styling for buttons
button_style = {"width": 15, "height": 2, "bg": "#4CAF50", "fg": "white", "font": button_font, "relief": "raised"}

# Language selection dropdown
language_menu = OptionMenu(button_frame, selected_language, "English", "French", "Spanish", "Japanese")
language_menu.config(**button_style)
language_menu.grid(row=0, column=0, padx=10)

# Start Application button
start_button = Button(button_frame, text="Start Application", command=start_application, **button_style)
start_button.grid(row=0, column=1, padx=10)

# Run the Tkinter loop
root.mainloop()
