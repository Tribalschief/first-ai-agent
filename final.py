# PYTHON SCRIPT START
import cv2
import numpy as np
import pytesseract
import RPi.GPIO as GPIO
import time
from picamera2 import Picamera2, Preview # Added Preview for potential debug
from RPLCD.gpio import CharLCD
import re
import os
from datetime import datetime
import json # For saving/loading active parkers
import math # For ceil function for billing

# ==================================
#        CONFIGURATION & TUNING
# ==================================
DEBUG_MODE = True  # SET TO True FOR ANPR TUNING AND DEBUG IMAGE SAVING
DEBUG_IMG_PATH_ENTRY = "debug_images_entry"
DEBUG_IMG_PATH_EXIT = "debug_images_exit"
LOG_FILE_PATH = "parking_log.csv"
ACTIVE_PARKERS_FILE = "active_unregistered_parkers.json"

# --- Camera (ANPR) ---
IMG_WIDTH = 1024  # Consider 640 or 800 if RPi struggles with 1024
IMG_HEIGHT = 576 # Consider 480 or 600 respectively

# --- Plate Extraction Tuning (ANPR - apply to both entry & exit) ---
# ** START TUNING THESE FIRST for plate_extraction_anpr_generic **
CANNY_LOW_THRESH = 50    # Try 30-80
CANNY_HIGH_THRESH = 180  # Try 120-200 (often 2-3x CANNY_LOW_THRESH)
CONTOUR_APPROX_FACTOR = 0.02 # Usually 0.01 to 0.04
MIN_PLATE_AREA = 700     # Adjust based on plate size in image (increase if picking up small false positives)
MIN_ASPECT_RATIO = 2.0   # Tighten this based on your local plate W/H (e.g., 2.0 for wider plates)
MAX_ASPECT_RATIO = 5.0   # Tighten this (e.g., 4.5 or 5.0)

# --- OCR Preprocessing Tuning (ANPR - apply to both entry & exit) ---
# ** TUNE THESE AFTER plate_extraction is good, for ocr_processing_anpr_generic **
OCR_RESIZE_HEIGHT = 60      # Tesseract likes char height ~30-50px. This is warped plate height.
THRESHOLD_METHOD = 'ADAPTIVE' # 'ADAPTIVE' or 'OTSU'. Experiment!
ADAPT_THRESH_BLOCK_SIZE = 19  # Must be odd. Try 11, 15, 19, 25. (for ADAPTIVE)
ADAPT_THRESH_C = 9            # Try 5-15. (for ADAPTIVE)
# Note: For THRESH_BINARY_INV, Tesseract expects light text on dark background.
# If Tesseract performs better with black text on white, either use THRESH_BINARY
# or invert the binary image: binary_plate = cv2.bitwise_not(binary_plate) before pytesseract

# --- Tesseract Tuning (ANPR - apply to both entry & exit) ---
TESS_LANG = 'eng'
TESS_OEM = 3  # Default LSTM engine, usually best.
TESS_PSM = '7' # '7': single line of text. Try '8' (single word), '6' (uniform block), '13' (raw line).
TESS_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" # IMPORTANT
EXPECTED_PLATE_PATTERN = "" # e.g., r"^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$" # CAREFUL: Must be correct!

# --- OCR Post-Processing / Validation (ANPR) ---
MIN_PLATE_LENGTH = 5 # Min expected characters for a valid plate

# --- ANPR Main Loop Timing ---
PROCESS_COOLDOWN_ANPR = 8
RESET_TIMEOUT_ANPR = 20 # Increased slightly

# --- Parking System Configuration ---
CAR_PRESENT_THRESHOLD_CM = 25 # Increase if entry US sensor is too sensitive, decrease if not detecting
SERVO_FREQUENCY = 50
SERVO_CLOSED_ANGLE = 0
SERVO_OPEN_ANGLE = 90
SERVO_MOVE_DELAY = 1.2 # Time for servo to complete its move
GATE_OPEN_DURATION_PARKING = 4.0 # How long gate stays open
US_POLLING_INTERVAL = 1.0
MAIN_LOOP_POLLING_INTERVAL = 0.1

# --- Pay-as-you-go ---
HOURLY_RATE = 5

# ==================================
#           PIN DEFINITIONS (BCM Mode) - VERIFY ALL ARE CORRECT & SAFE!
# ==================================
# --- LCD Pins ---
LCD_RS_PIN = 7
LCD_E_PIN = 8
LCD_D4_PIN = 25
LCD_D5_PIN = 24
LCD_D6_PIN = 23
LCD_D7_PIN = 12

# --- Servo Pins ---
SERVO_ENTRY_PIN = 17 # Physical 11
SERVO_EXIT_PIN = 4   # Physical 7

# --- Buzzer Pin (MUST BE A SAFE, GENERAL-PURPOSE GPIO) ---
BUZZ_PIN = 14 # Example: BCM 14 (Physical 8, TXD). Ensure it's not BCM 2 or 3!

# --- Ultrasonic Sensor Pins (Entry & Exit Detection) ---
US_ENTRY_SENSOR = {"name": "Entry Detect", "trig": 0, "echo": 1} # Phys 27, 28
US_EXIT_SENSOR = {"name": "Exit Detect", "trig": 27, "echo": 22}  # Phys 13, 15 -- PICK DIFFERENT, FREE PINS! Example: BCM 5, BCM 6

# --- Ultrasonic Sensor Pins (Parking System Slots) ---
US_SENSORS = [
    {"name": "Slot 1", "trig": 19, "echo": 26},  # Phys 35, 37 (SPI1_MISO, SPI1_CE2_N)
    {"name": "Slot 2", "trig": 20, "echo": 21},  # Phys 38, 40 (SPI1_MOSI, SPI1_SCLK)
    # Add more if you have them, ensuring no conflicts with other hardware (SPI, I2C etc.)
    # {"name": "Slot 3", "trig": xx, "echo": yy},
]
TOTAL_PARKING_SPOTS = len(US_SENSORS) # Update if you change number of slot sensors

# ==================================
#           GLOBAL VARIABLES
# ==================================
lcd = None
lcd_ready = False
picam2_entry = None
picam2_exit = None
camera_entry_ready = False
camera_exit_ready = False
servo_entry_pwm = None
servo_exit_pwm = None
occupied_spots_status = [False] * max(1, TOTAL_PARKING_SPOTS) # Ensure list is not empty
previous_occupied_spots_status = [False] * max(1, TOTAL_PARKING_SPOTS)
available_spots_count = TOTAL_PARKING_SPOTS
last_us_poll_time = 0

entry_gate_busy = False
exit_gate_busy = False

entry_anpr_last_processed_plate = ""
entry_anpr_last_process_time = 0
entry_anpr_last_plate_contour_detection_time = 0
entry_anpr_processing_active = False

exit_anpr_last_processed_plate = ""
exit_anpr_last_process_time = 0
exit_anpr_last_plate_contour_detection_time = 0
exit_anpr_processing_active = False

active_unregistered_parkers = {}

# --- Debug Directory Creation ---
def create_debug_dir(path):
    if DEBUG_MODE and not os.path.exists(path):
        try:
            os.makedirs(path)
            print(f"Created debug directory: {path}")
        except OSError as e:
            print(f"[ERROR] Could not create debug directory '{path}': {e}")

# --- Load/Save Active Unregistered Parkers ---
def load_active_parkers():
    global active_unregistered_parkers
    try:
        if os.path.exists(ACTIVE_PARKERS_FILE):
            with open(ACTIVE_PARKERS_FILE, 'r') as f:
                active_unregistered_parkers = json.load(f)
            print(f"Loaded {len(active_unregistered_parkers)} active unregistered parkers.")
    except Exception as e:
        print(f"[ERROR] Could not load active parkers: {e}")
        active_unregistered_parkers = {}

def save_active_parkers():
    try:
        with open(ACTIVE_PARKERS_FILE, 'w') as f:
            json.dump(active_unregistered_parkers, f, indent=4) # Added indent for readability
        if DEBUG_MODE: print(f"Saved {len(active_unregistered_parkers)} active parkers.")
    except Exception as e:
        print(f"[ERROR] Could not save active parkers: {e}")

# --- LCD Setup Function ---
def setup_lcd():
    global lcd, lcd_ready
    try:
        lcd = CharLCD(
            numbering_mode=GPIO.BCM, cols=16, rows=2,
            pin_rs=LCD_RS_PIN, pin_e=LCD_E_PIN,
            pins_data=[LCD_D4_PIN, LCD_D5_PIN, LCD_D6_PIN, LCD_D7_PIN],
            charmap='A00', auto_linebreaks=True
        )
        lcd.clear()
        lcd_ready = True
        print("LCD Initialized Successfully.")
        lcd_display_merged("System Booting", "Please Wait...")
    except Exception as e:
        print(f"[ERROR] Failed to initialize LCD: {e}")
        lcd_ready = False
        class DummyLCD: # Fallback if LCD fails
            def write_string(self, text): print(f"LCD_DUMMY: {text}")
            def clear(self): print("LCD_DUMMY: clear()")
            def cursor_pos(self, pos): print(f"LCD_DUMMY: cursor_pos({pos})")
        lcd = DummyLCD()

def lcd_display_merged(line1, line2="", clear_first=True):
    if not lcd_ready and not isinstance(lcd, DummyLCD): return # Don't try if LCD failed AND no dummy
    try:
        if clear_first: lcd.clear()
        lcd.cursor_pos = (0, 0)
        lcd.write_string(str(line1)[:16])
        if line2:
            lcd.cursor_pos = (1, 0)
            lcd.write_string(str(line2)[:16])
    except Exception as e:
        print(f"[ERROR] LCD display error: {e}")

# --- Camera Setup (for two cameras if pay-as-you-go is used) ---
def setup_cameras():
    global picam2_entry, camera_entry_ready, picam2_exit, camera_exit_ready
    camera_config_main = {"size": (IMG_WIDTH, IMG_HEIGHT), "format": "RGB888"}
    camera_config_lores = {"size": (320,240), "format": "YUV420"} # For faster preview if needed
    
    # Entry Camera
    try:
        print("Initializing Entry Camera (cam_num=0)...")
        picam2_entry = Picamera2(camera_num=0)
        config_entry = picam2_entry.create_preview_configuration(main=camera_config_main, lores=camera_config_lores)
        picam2_entry.configure(config_entry)
        # picam2_entry.start_preview(Preview.QTGL) # UNCOMMENT FOR FOCUS/AIMING
        picam2_entry.start()
        # Optional: Set camera controls for exposure, gain etc.
        # picam2_entry.set_controls({"ExposureTime": 20000, "AnalogueGain": 1.0})
        time.sleep(2.0)
        camera_entry_ready = True
        print("Entry Camera Initialized.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Entry Camera: {e}")
        camera_entry_ready = False

    # Exit Camera (Only if pay-as-you-go is a core feature, otherwise conditional)
    # For simplicity, assume it's always attempted. Check camera_exit_ready before use.
    try:
        print("Initializing Exit Camera (cam_num=1)...")
        picam2_exit = Picamera2(camera_num=1)
        config_exit = picam2_exit.create_preview_configuration(main=camera_config_main, lores=camera_config_lores)
        picam2_exit.configure(config_exit)
        # picam2_exit.start_preview(Preview.QTGL) # UNCOMMENT FOR FOCUS/AIMING
        picam2_exit.start()
        # picam2_exit.set_controls({"ExposureTime": 20000, "AnalogueGain": 1.0})
        time.sleep(2.0)
        camera_exit_ready = True
        print("Exit Camera Initialized.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Exit Camera: {e}. Pay-as-you-go exit might be affected.")
        camera_exit_ready = False


# --- GPIO Setup ---
def setup_gpio():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)

    # Ultrasonic Sensors
    for us_dev_info in [US_ENTRY_SENSOR, US_EXIT_SENSOR] + US_SENSORS:
        if "trig" in us_dev_info and "echo" in us_dev_info: # Check if full dict
            GPIO.setup(us_dev_info["trig"], GPIO.OUT)
            GPIO.setup(us_dev_info["echo"], GPIO.IN)
            GPIO.output(us_dev_info["trig"], False)
    
    # Servos
    GPIO.setup(SERVO_ENTRY_PIN, GPIO.OUT)
    GPIO.setup(SERVO_EXIT_PIN, GPIO.OUT)
    # Buzzer
    GPIO.setup(BUZZ_PIN, GPIO.OUT, initial=GPIO.LOW)
    print("GPIO Initialized.")


# --- Servo Control ---
def setup_servos():
    global servo_entry_pwm, servo_exit_pwm
    print("SERVODBG: Initializing Servos...")
    try:
        servo_entry_pwm = GPIO.PWM(SERVO_ENTRY_PIN, SERVO_FREQUENCY)
        servo_exit_pwm = GPIO.PWM(SERVO_EXIT_PIN, SERVO_FREQUENCY)

        # Start PWM. Duty cycle 0 means no pulse. We'll set angle explicitly.
        servo_entry_pwm.start(0)
        servo_exit_pwm.start(0)
        print("SERVODBG: PWM started for both servos.")

        set_servo_angle_parking(servo_entry_pwm, SERVO_CLOSED_ANGLE, "Entry Initial")
        set_servo_angle_parking(servo_exit_pwm, SERVO_CLOSED_ANGLE, "Exit Initial")
        print("Servos Initialized and explicitly set to Closed.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize servos: {e}")

def set_servo_angle_parking(servo_pwm, angle, gate_name_debug=""):
    if servo_pwm is None:
        print(f"SERVODBG [WARN]: Servo PWM object for {gate_name_debug} is None. Cannot set angle.")
        return
    duty = (angle / 18.0) + 2.0 # Standard formula for 0-180 to 2-12% duty
    if duty < 2.0: duty = 2.0 # Min duty
    if duty > 12.0: duty = 12.0 # Max duty
    
    print(f"SERVODBG: {gate_name_debug} - Setting angle: {angle} deg, Calculated Duty: {duty:.2f}%")
    servo_pwm.ChangeDutyCycle(duty)
    time.sleep(SERVO_MOVE_DELAY) # Give servo time to physically move
    servo_pwm.ChangeDutyCycle(0) # Stop sending PWM signal. Crucial to prevent jitter/overheating
                                 # and allows manual movement if servo power is cut.
    print(f"SERVODBG: {gate_name_debug} - PWM signal stopped (Duty set to 0).")

def open_gate_parking(servo_pwm_obj, gate_name, associated_plate=""):
    print(f"SERVODBG: Attempting to OPEN gate: {gate_name}")
    line1_text = f"{gate_name} Gate"
    if associated_plate: line1_text = associated_plate[:8].ljust(8) # Pad for consistent look

    is_entry_process = (gate_name == "Entry" and entry_anpr_processing_active)
    is_exit_process = (gate_name == "Exit" and exit_anpr_processing_active)
    clear_lcd = not (is_entry_process or is_exit_process)

    lcd_display_merged(line1_text, "Opening...", clear_first=clear_lcd)
    print(f"Opening {gate_name} gate...")
    set_servo_angle_parking(servo_pwm_obj, SERVO_OPEN_ANGLE, gate_name)
    print(f"{gate_name} gate OPENED.")

def close_gate_parking(servo_pwm_obj, gate_name):
    print(f"SERVODBG: Attempting to CLOSE gate: {gate_name}")
    lcd_display_merged(f"{gate_name} Gate", "Closing...")
    print(f"Closing {gate_name} gate...")
    set_servo_angle_parking(servo_pwm_obj, SERVO_CLOSED_ANGLE, gate_name)
    print(f"{gate_name} gate CLOSED.")

# --- Ultrasonic Sensor Function ---
def measure_distance(trig_pin, echo_pin, sensor_name="US"):
    # ... (Same stable version from previous responses)
    try:
        GPIO.output(trig_pin, False)
        time.sleep(0.02) # Settle

        GPIO.output(trig_pin, True)
        time.sleep(0.00001) # 10us pulse
        GPIO.output(trig_pin, False)

        pulse_start_timeout = time.time()
        start_time_echo = pulse_start_timeout

        while GPIO.input(echo_pin) == 0:
            start_time_echo = time.time()
            if start_time_echo - pulse_start_timeout > 0.05: # 50ms timeout for pulse start
                if DEBUG_MODE: print(f"US TIMEOUT ({sensor_name}): No echo pulse start.")
                return float('inf')

        pulse_end_timeout = time.time()
        end_time_echo = pulse_end_timeout

        while GPIO.input(echo_pin) == 1:
            end_time_echo = time.time()
            if end_time_echo - pulse_end_timeout > 0.05: # 50ms timeout for pulse end
                if DEBUG_MODE: print(f"US TIMEOUT ({sensor_name}): Echo pulse too long.")
                return float('inf')

        duration = end_time_echo - start_time_echo
        distance = (duration * 34300) / 2
        return distance if distance >= 0 else float('inf')
    except RuntimeError: # Catches if GPIO access fails after cleanup during shutdown
        if DEBUG_MODE: print(f"US ({sensor_name}): GPIO runtime error (likely during shutdown).")
        return float('inf')
    except Exception as e:
        if DEBUG_MODE: print(f"US ({sensor_name}) Error: {e}")
        return float('inf')

# --- ANPR Image Processing & OCR (Generic Functions) ---
def capture_image_anpr_generic(picam_instance, camera_name="UnknownCam"):
    if picam_instance is None or not hasattr(picam_instance, 'started') or not picam_instance.started:
        print(f"[ERROR-ANPR-{camera_name}] Camera not ready, not an instance, or not started.")
        return None
    try:
        frame = picam_instance.capture_array("main")
        # Picamera2 often gives RGB, convert to BGR for OpenCV
        # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # If capture_array gives RGB
        # If frame is already BGR or you configured RGB888 and handle it, ensure correct format
        # Assuming main format is RGB888, need to convert to BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame_bgr
    except Exception as e:
        print(f"[ERROR-ANPR-{camera_name}] Image capture failed: {e}")
        return None

def plate_extraction_anpr_generic(image_color, debug_path_prefix="anpr_debug"):
    if image_color is None or image_color.size == 0: return None
    ts_for_debug = int(time.time())
    current_debug_img_path = DEBUG_IMG_PATH_ENTRY if "entry" in debug_path_prefix.lower() else DEBUG_IMG_PATH_EXIT
    
    gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    if DEBUG_MODE: cv2.imwrite(os.path.join(current_debug_img_path, f"{ts_for_debug}_{debug_path_prefix}_00_gray.png"), gray)
    
    # Experiment with blurring
    # blur = cv2.bilateralFilter(gray, 9, 75, 75) # Smaller d for less blur, larger sigma for more color/space smoothing
    blur = cv2.GaussianBlur(gray, (5,5), 0) # Simpler, often effective
    if DEBUG_MODE: cv2.imwrite(os.path.join(current_debug_img_path, f"{ts_for_debug}_{debug_path_prefix}_00b_blur.png"), blur)
    
    edges = cv2.Canny(blur, CANNY_LOW_THRESH, CANNY_HIGH_THRESH)
    if DEBUG_MODE: cv2.imwrite(os.path.join(current_debug_img_path, f"{ts_for_debug}_{debug_path_prefix}_01_edges.png"), edges)
    
    cnts, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # RETR_EXTERNAL better for outer contours
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10] # Get largest 10
    
    plate_contour_found = None
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, CONTOUR_APPROX_FACTOR * perimeter, True)
        if len(approx) == 4: # Is it a quadrilateral?
            (x, y, w, h) = cv2.boundingRect(approx)
            if h == 0: continue # Avoid division by zero
            aspect_ratio = w / float(h)
            area = cv2.contourArea(approx)

            # Print contour info if in deep debug
            if DEBUG_MODE and area > MIN_PLATE_AREA / 3 : # Print some candidates
                print(f"  Contour ({debug_path_prefix}): Area={area:.0f}, AR={aspect_ratio:.2f} (W:{w}, H:{h})")

            if MIN_PLATE_AREA < area and MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO:
                # Additional checks (e.g. solidity, extent) can be added here for robustness
                plate_contour_found = approx
                if DEBUG_MODE:
                     debug_frame_contour = image_color.copy()
                     cv2.drawContours(debug_frame_contour, [plate_contour_found], -1, (0, 255, 0), 2)
                     cv2.putText(debug_frame_contour, f"A:{area:.0f} AR:{aspect_ratio:.1f}", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)
                     cv2.imwrite(os.path.join(current_debug_img_path, f"{ts_for_debug}_{debug_path_prefix}_02_contour_SELECTED.png"), debug_frame_contour)
                break
            elif DEBUG_MODE and area > MIN_PLATE_AREA /2 : # Log rejected good candidates
                 debug_frame_contour_rej = image_color.copy()
                 cv2.drawContours(debug_frame_contour_rej, [approx], -1, (0, 0, 255), 2) # Red for rejected
                 cv2.putText(debug_frame_contour_rej, f"REJ: A:{area:.0f} AR:{aspect_ratio:.1f}", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
                 cv2.imwrite(os.path.join(current_debug_img_path, f"{ts_for_debug}_{debug_path_prefix}_02_contour_REJECTED_{area:.0f}.png"), debug_frame_contour_rej)


    if plate_contour_found is None:
        if DEBUG_MODE: print(f"  ANPR Plate Extraction ({debug_path_prefix}): No suitable 4-sided contour found matching criteria.")
        return None
    
    # Perspective Transform
    # Ensure pts are ordered: tl, tr, br, bl for getPerspectiveTransform
    rect = np.zeros((4, 2), dtype = "float32")
    s = plate_contour_found.sum(axis = 2) # Sum over the inner arrays
    rect[0] = plate_contour_found[np.argmin(s)]
    rect[2] = plate_contour_found[np.argmax(s)]
    diff = np.diff(plate_contour_found, axis = 2)
    rect[1] = plate_contour_found[np.argmin(diff)]
    rect[3] = plate_contour_found[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    if maxWidth <=0 or maxHeight <=0:
        if DEBUG_MODE: print(f"  ANPR Perspective ({debug_path_prefix}): Invalid dimensions for warp ({maxWidth}x{maxHeight}).")
        return None

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped_plate_gray = cv2.warpPerspective(gray, M, (maxWidth, maxHeight)) # Use gray image

    if warped_plate_gray is None or warped_plate_gray.size == 0:
        if DEBUG_MODE: print(f"  ANPR plate_extraction ({debug_path_prefix}): Warped plate is empty.")
        return None
    if DEBUG_MODE: cv2.imwrite(os.path.join(current_debug_img_path, f"{ts_for_debug}_{debug_path_prefix}_03_warped.png"), warped_plate_gray)
    return warped_plate_gray

def ocr_processing_anpr_generic(plate_image_gray, debug_path_prefix="anpr_debug"):
    global entry_anpr_last_plate_contour_detection_time, exit_anpr_last_plate_contour_detection_time

    if plate_image_gray is None or plate_image_gray.size == 0: return ""
    current_debug_img_path = DEBUG_IMG_PATH_ENTRY if "entry" in debug_path_prefix.lower() else DEBUG_IMG_PATH_EXIT

    current_time_ocr = time.time()
    if "entry" in debug_path_prefix.lower(): entry_anpr_last_plate_contour_detection_time = current_time_ocr
    elif "exit" in debug_path_prefix.lower(): exit_anpr_last_plate_contour_detection_time = current_time_ocr
    
    # Resize to a fixed height for OCR, preserving aspect ratio
    try:
        h_orig, w_orig = plate_image_gray.shape[:2]
        aspect_ratio_orig = w_orig / float(h_orig) if h_orig > 0 else 1.0
        target_width = int(OCR_RESIZE_HEIGHT * aspect_ratio_orig)
        if target_width > 10 and OCR_RESIZE_HEIGHT > 10: # Sanity check dimensions
            plate_image_resized = cv2.resize(plate_image_gray, (target_width, OCR_RESIZE_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
        else:
            if DEBUG_MODE: print(f"  OCR ({debug_path_prefix}): Original plate too small for reliable resize. Using original.")
            plate_image_resized = plate_image_gray
    except Exception as e:
        print(f"[WARN-ANPR] Plate resize ({debug_path_prefix}) failed: {e}. Using original.")
        plate_image_resized = plate_image_gray

    if DEBUG_MODE: cv2.imwrite(os.path.join(current_debug_img_path, f"{time.time()}_{debug_path_prefix}_04a_resized.png"), plate_image_resized)

    # Thresholding
    if THRESHOLD_METHOD == 'ADAPTIVE':
        binary_plate = cv2.adaptiveThreshold(plate_image_resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPT_THRESH_BLOCK_SIZE, ADAPT_THRESH_C)
    elif THRESHOLD_METHOD == 'OTSU':
        blurred_for_otsu = cv2.GaussianBlur(plate_image_resized, (5,5), 0)
        _, binary_plate = cv2.threshold(blurred_for_otsu, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else: # Default to simple global threshold if unknown method
        print(f"[WARN-ANPR] Unknown THRESHOLD_METHOD ({debug_path_prefix}). Defaulting to simple global threshold.")
        _, binary_plate = cv2.threshold(plate_image_resized, 127, 255, cv2.THRESH_BINARY_INV)

    # Optional: Invert if Tesseract expects black text on white background and binary_plate is white text on black
    # if TESS_PSM in ['7','8','6','13']: # These PSMs often work better with black text on white
    #     binary_plate = cv2.bitwise_not(binary_plate)
    #     if DEBUG_MODE: print(f"  OCR ({debug_path_prefix}): Inverted binary image for Tesseract.")


    ts_for_debug_ocr = int(time.time())
    if DEBUG_MODE: cv2.imwrite(os.path.join(current_debug_img_path, f"{ts_for_debug_ocr}_{debug_path_prefix}_04b_binary.png"), binary_plate)

    custom_config = f'--oem {TESS_OEM} --psm {TESS_PSM} -l {TESS_LANG}'
    if TESS_WHITELIST: custom_config += f' -c tessedit_char_whitelist={TESS_WHITELIST}'
    # Add more configs if needed: e.g. -c load_system_dawg=0 -c load_freq_dawg=0 (to speed up by disabling dictionaries if whitelist is very strict)
    
    try:
        raw_text = pytesseract.image_to_string(binary_plate, config=custom_config)
        if DEBUG_MODE: print(f"  ANPR Raw OCR ({debug_path_prefix}): '{raw_text.strip()}'")
        
        # Post-processing OCR result
        cleaned_text = ''.join(filter(str.isalnum, raw_text)).upper()
        
        if len(cleaned_text) < MIN_PLATE_LENGTH:
            if DEBUG_MODE and cleaned_text: print(f"  ANPR Reject ({debug_path_prefix}) (short): '{cleaned_text}' len {len(cleaned_text)}")
            return ""
        if EXPECTED_PLATE_PATTERN and not re.fullmatch(EXPECTED_PLATE_PATTERN, cleaned_text):
            if DEBUG_MODE: print(f"  ANPR Reject ({debug_path_prefix}) (pattern mismatch): '{cleaned_text}' vs pattern '{EXPECTED_PLATE_PATTERN}'")
            return ""
        
        # Basic sanity check: e.g., too many consecutive same characters (often OCR error)
        if len(cleaned_text) > 2:
            for i in range(len(cleaned_text) - 2):
                if cleaned_text[i] == cleaned_text[i+1] == cleaned_text[i+2] and cleaned_text[i].isalpha():
                    if DEBUG_MODE: print(f"  ANPR Reject ({debug_path_prefix}) (too many consec. letters): '{cleaned_text}'")
                    return "" # Reject "OOO", "III" which are common OCR noise
                if cleaned_text[i] == cleaned_text[i+1] == cleaned_text[i+2] and cleaned_text[i].isdigit() and cleaned_text[i] in '01':
                    if DEBUG_MODE: print(f"  ANPR Reject ({debug_path_prefix}) (too many consec. 0/1): '{cleaned_text}'")
                    return "" # Reject "000", "111" 
        
        if DEBUG_MODE: print(f"  ANPR Cleaned OCR ({debug_path_prefix}): '{cleaned_text}'")
        return cleaned_text
    except pytesseract.TesseractError as te: # More specific error
        print(f"[ERROR-ANPR ({debug_path_prefix})] Tesseract Error: {te}")
        return ""
    except Exception as e:
        print(f"[ERROR-ANPR ({debug_path_prefix})] OCR processing failed: {e}")
        return ""

def check_database_anpr(plate_text):
    # ... (same as before)
    if not plate_text: return False
    try:
        with open('Database.txt', 'r') as f:
            database_plates = {line.strip().upper() for line in f if line.strip()}
        return plate_text in database_plates
    except FileNotFoundError:
        print("[ERROR-ANPR] Database.txt not found.")
        return False # Cannot validate if file doesn't exist
    except Exception as e:
        print(f"[ERROR-ANPR] Database read error: {e}")
        return False

# --- Parking System Logic ---
def update_parking_spots_status():
    global occupied_spots_status, previous_occupied_spots_status, available_spots_count, last_us_poll_time
    current_time_us_poll = time.time()
    if current_time_us_poll - last_us_poll_time < US_POLLING_INTERVAL:
        return

    if not US_SENSORS: # If no parking slot sensors are defined
        available_spots_count = TOTAL_PARKING_SPOTS # Assume all are free or handle differently
        last_us_poll_time = current_time_us_poll
        return

    new_occupied_count = 0
    changed_slots_this_poll = False
    for i, sensor_info in enumerate(US_SENSORS):
        dist = measure_distance(sensor_info["trig"], sensor_info["echo"], sensor_info["name"])
        current_spot_is_occupied = (dist < CAR_PRESENT_THRESHOLD_CM + 10) # Add small buffer to slot sensor
        
        if current_spot_is_occupied != previous_occupied_spots_status[i]:
            changed_slots_this_poll = True
            print(f"PARKING: Slot {sensor_info['name']} is now {'OCCUPIED' if current_spot_is_occupied else 'EMPTY'} (Dist: {dist:.1f}cm)")
        
        occupied_spots_status[i] = current_spot_is_occupied
        if current_spot_is_occupied: new_occupied_count += 1
    
    physically_occupied_count = new_occupied_count
    available_spots_count = TOTAL_PARKING_SPOTS - physically_occupied_count
    if available_spots_count < 0: available_spots_count = 0 # Cannot be negative

    if changed_slots_this_poll or DEBUG_MODE: # Print if changed or always in debug
        print(f"PARKING UPDATE: Physically occupied by US: {physically_occupied_count}, Available (by US): {available_spots_count}")
        print(f"PARKING UPDATE: Logged unregistered vehicles: {len(active_unregistered_parkers)}")

    if changed_slots_this_poll: previous_occupied_spots_status = list(occupied_spots_status)
    last_us_poll_time = current_time_us_poll

def display_parking_main_status_lcd():
    global available_spots_count
    # Don't update if any major process is active to avoid flickering important messages
    if entry_anpr_processing_active or exit_anpr_processing_active or entry_gate_busy or exit_gate_busy:
        return

    # Calculate overall availability more comprehensively
    # This is tricky: available_spots_count is from physical US sensors.
    # True availability for NEW entries also depends on how many are logged (registered/unregistered).
    # For LCD, displaying US-based is simpler for now. Gate logic should be smarter.
    display_spots = available_spots_count 

    status_line1 = "Spots Available"
    status_line2 = f"{display_spots} Free"

    if display_spots >= TOTAL_PARKING_SPOTS : status_line1 = "Parking Empty" # Can be > if sensors error
    elif display_spots <= 0:
        status_line1 = "Parking Full!"
        status_line2 = "No Spots Free"
    lcd_display_merged(status_line1, status_line2, clear_first=True)


# --- Parking Log File Setup & Logging ---
def setup_log_file():
    if not os.path.exists(LOG_FILE_PATH):
        try:
            with open(LOG_FILE_PATH, 'w') as f:
                f.write("Timestamp,PlateNumber,Action,Details\n")
            print(f"Created log file: {LOG_FILE_PATH}")
        except IOError as e: print(f"[ERROR] Could not create log file '{LOG_FILE_PATH}': {e}")

def log_parking_event(plate_number, action, details=""):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE_PATH, 'a') as f:
            f.write(f"{timestamp},{plate_number},{action},{details}\n")
    except IOError as e: print(f"[ERROR] Could not write to log file '{LOG_FILE_PATH}': {e}")

# --- Beeper ---
def beep(duration=0.1, count=1, delay_between=0.1):
    try:
        for _ in range(count):
            GPIO.output(BUZZ_PIN, GPIO.HIGH)
            time.sleep(duration)
            GPIO.output(BUZZ_PIN, GPIO.LOW)
            if count > 1 and _ < count -1 : time.sleep(delay_between)
    except RuntimeError: # If GPIO cleaned up during beep
        pass
    except Exception as e:
        print(f"Buzzer Error: {e}")

# --- ANPR Entry Sequence ---
def run_anpr_entry_sequence():
    # ... (Largely same as previous, with more SERVODBG focus)
    global entry_anpr_last_processed_plate, entry_anpr_last_process_time, available_spots_count
    global active_unregistered_parkers, entry_gate_busy # ensure entry_gate_busy is modifiable

    print("SERVODBG: run_anpr_entry_sequence BEGIN")
    entry_gate_busy = True # Mark as busy for this sequence

    if not camera_entry_ready:
        print("ANPR Entry: Entry camera not ready.")
        lcd_display_merged("Entry Cam Error", "Service Down", clear_first=True); time.sleep(2)
        entry_gate_busy = False; return False

    print("\nANPR Entry: Car detected. Reading plate...")
    lcd_display_merged("Car at Entry", "Scanning Plate...", clear_first=True)

    frame_color = capture_image_anpr_generic(picam2_entry, "EntryCam")
    if frame_color is None:
        lcd_display_merged("Entry Cam Error", "Capture Failed", clear_first=True); time.sleep(2)
        entry_gate_busy = False; return False

    extracted_plate_img_gray = plate_extraction_anpr_generic(frame_color, "entry")
    if extracted_plate_img_gray is None:
        print("ANPR Entry: No plate-like contour detected.")
        lcd_display_merged("No Plate Found", "Reposition Car", clear_first=True); time.sleep(2)
        entry_gate_busy = False; return False

    plate_text_ocr = ocr_processing_anpr_generic(extracted_plate_img_gray, "entry")
    if not plate_text_ocr:
        print("ANPR Entry: Contour found, but OCR failed.")
        lcd_display_merged("Plate Found", "OCR Failed", clear_first=True); time.sleep(2)
        entry_gate_busy = False; return False

    current_time = time.time()
    if (plate_text_ocr == entry_anpr_last_processed_plate) and \
       (current_time - entry_anpr_last_process_time < PROCESS_COOLDOWN_ANPR):
        print(f"ANPR Entry: Ignoring '{plate_text_ocr}' (recently processed / cooldown).")
        entry_gate_busy = False; return False

    entry_anpr_last_processed_plate = plate_text_ocr
    entry_anpr_last_process_time = current_time

    print(f"ANPR Entry Detected: '{plate_text_ocr}'")
    lcd_display_merged(f"{plate_text_ocr[:8]} Checking", "Please Wait...", clear_first=True)
    time.sleep(0.5)

    is_registered = check_database_anpr(plate_text_ocr)
    
    # Smart capacity check:
    # Number of physically free spots from US_SENSORS should be > 0
    # AND number of "logged in" cars (unregistered + known registered if tracked) < TOTAL_PARKING_SPOTS
    physically_free_spots = available_spots_count # From US sensor polling
    # For now, simplified: entry allowed if US sensors show space.
    # A more robust system would reconcile US sensor data with logged entries.
    system_has_capacity = (physically_free_spots > 0)
    if DEBUG_MODE: print(f"ENTRY CHECK: Plate '{plate_text_ocr}', Registered: {is_registered}, US_Available_Spots: {physically_free_spots}, SystemHasCapacity: {system_has_capacity}")


    gate_action_taken = False
    if is_registered:
        if system_has_capacity:
            print("[REGISTERED] ✔️ Access Granted!")
            lcd_display_merged(plate_text_ocr[:8].ljust(8), "Access Granted", clear_first=True)
            beep(duration=0.3, count=1)
            log_parking_event(plate_text_ocr, "ENTRY_REGISTERED_GRANTED")
            open_gate_parking(servo_entry_pwm, "Entry", plate_text_ocr)
            gate_action_taken = True
        else:
            print("[REGISTERED] BUT PARKING FULL ❌ Access Denied!")
            lcd_display_merged(plate_text_ocr[:8].ljust(8), "Parking Full!", clear_first=True)
            beep(duration=0.1, count=3)
            log_parking_event(plate_text_ocr, "ENTRY_REGISTERED_DENIED_FULL")
            time.sleep(1.5)
    else: # Unregistered User
        if system_has_capacity:
            print(f"[UNREGISTERED] '{plate_text_ocr}'. Granting temporary access.")
            lcd_display_merged(plate_text_ocr[:8].ljust(8), "Temp. Access", clear_first=True)
            beep(duration=0.15, count=2)
            active_unregistered_parkers[plate_text_ocr] = current_time
            save_active_parkers()
            log_parking_event(plate_text_ocr, "ENTRY_UNREGISTERED_GRANTED_TEMP", f"EntryTime:{current_time}")
            open_gate_parking(servo_entry_pwm, "Entry", plate_text_ocr)
            gate_action_taken = True
        else:
            print(f"[UNREGISTERED] '{plate_text_ocr}' BUT PARKING FULL ❌ Access Denied!")
            lcd_display_merged(plate_text_ocr[:8].ljust(8), "Parking Full!", clear_first=True)
            beep(duration=0.1, count=3)
            log_parking_event(plate_text_ocr, "ENTRY_UNREGISTERED_DENIED_FULL")
            time.sleep(1.5)

    if gate_action_taken:
        time.sleep(GATE_OPEN_DURATION_PARKING)
        close_gate_parking(servo_entry_pwm, "Entry")
        entry_gate_busy = False
        print("SERVODBG: run_anpr_entry_sequence END (Gate Cycled)")
        return True
    else:
        entry_gate_busy = False
        print("SERVODBG: run_anpr_entry_sequence END (No Gate Action)")
        return False

# --- ANPR Exit Sequence ---
def run_anpr_exit_sequence():
    # ... (Largely same as previous)
    global exit_anpr_last_processed_plate, exit_anpr_last_process_time, active_unregistered_parkers
    global exit_gate_busy

    print("SERVODBG: run_anpr_exit_sequence BEGIN")
    exit_gate_busy = True

    if not camera_exit_ready: # Only proceed if exit camera is functional
        print("ANPR Exit: Exit camera not ready. Cannot process ANPR exit.")
        lcd_display_merged("Exit Cam Error", "Manual Exit?", clear_first=True); time.sleep(2)
        # Fallback for NO EXIT CAMERA: Just open the gate based on US_EXIT_SENSOR.
        # This removes billing for unregistered, but allows exit.
        # print("ANPR Exit: Camera unavailable. Opening gate for generic exit.")
        # log_parking_event("NO_EXIT_CAM_ANPR", "EXIT_GENERIC_NO_CAM")
        # open_gate_parking(servo_exit_pwm, "Exit")
        # time.sleep(GATE_OPEN_DURATION_PARKING)
        # close_gate_parking(servo_exit_pwm, "Exit")
        # exit_gate_busy = False; return True # Or False if strict ANPR exit is required
        exit_gate_busy = False; return False


    print("\nANPR Exit: Car detected. Reading plate...")
    lcd_display_merged("Car at Exit", "Scanning Plate...", clear_first=True)

    frame_color = capture_image_anpr_generic(picam2_exit, "ExitCam")
    if frame_color is None:
        lcd_display_merged("Exit Cam Error", "Capture Failed", clear_first=True); time.sleep(2)
        exit_gate_busy = False; return False

    extracted_plate_img_gray = plate_extraction_anpr_generic(frame_color, "exit")
    if extracted_plate_img_gray is None:
        print("ANPR Exit: No plate-like contour detected.")
        lcd_display_merged("No Plate Found", "Reposition Car", clear_first=True); time.sleep(2)
        exit_gate_busy = False; return False

    plate_text_ocr = ocr_processing_anpr_generic(extracted_plate_img_gray, "exit")
    if not plate_text_ocr:
        print("ANPR Exit: Contour found, but OCR failed.")
        lcd_display_merged("Plate Found", "OCR Failed", clear_first=True); time.sleep(2)
        # CRITICAL: What to do if OCR fails for a paying customer? Manual override needed in real world.
        # For now, no gate action if OCR fails to prevent unpaid exits.
        exit_gate_busy = False; return False

    current_time = time.time()
    if (plate_text_ocr == exit_anpr_last_processed_plate) and \
       (current_time - exit_anpr_last_process_time < PROCESS_COOLDOWN_ANPR):
        print(f"ANPR Exit: Ignoring '{plate_text_ocr}' (recently processed / cooldown).")
        # Option: if they are still there after cooldown, maybe reshow bill?
        # For now, just ignore and wait for car to move or new detection cycle.
        exit_gate_busy = False; return False

    exit_anpr_last_processed_plate = plate_text_ocr
    exit_anpr_last_process_time = current_time

    print(f"ANPR Exit Detected: '{plate_text_ocr}'")
    lcd_display_merged(f"{plate_text_ocr[:8]} Processing", "Please Wait...", clear_first=True)
    time.sleep(0.5)

    is_registered_db = check_database_anpr(plate_text_ocr)
    is_unregistered_parker = plate_text_ocr in active_unregistered_parkers
    gate_action_taken = False

    if is_unregistered_parker:
        entry_time = active_unregistered_parkers[plate_text_ocr]
        parked_duration_seconds = current_time - entry_time
        parked_duration_hours = parked_duration_seconds / 3600.0
        
        billed_hours = math.ceil(parked_duration_hours) if parked_duration_hours > 0 else 1
        if parked_duration_hours <= (10/60): billed_hours = 0 # Grace period e.g. 10 mins, no charge
        elif billed_hours == 0 and parked_duration_hours > (10/60): billed_hours = 1 # Min 1 hour charge if over grace
        
        total_bill = billed_hours * HOURLY_RATE

        if total_bill > 0 :
            print(f"[UNREGISTERED] '{plate_text_ocr}' Exiting. Parked: {parked_duration_hours:.2f} hrs. Bill: Rs. {total_bill}")
            lcd_display_merged(f"Bill: Rs.{total_bill}", f"{plate_text_ocr[:8]} Pay", clear_first=True)
            beep(duration=0.08, count=5, delay_between=0.08) # Billing beep
            log_parking_event(plate_text_ocr, "EXIT_UNREGISTERED_BILLED", f"Bill:{total_bill},Hours:{billed_hours:.0f}")
            print("SIMULATING PAYMENT / DISPLAY BILL (5s)...")
            time.sleep(5) # Simulate payment / display bill duration before opening gate
        else: # Grace period
            print(f"[UNREGISTERED] '{plate_text_ocr}' Exiting within grace period. No charge.")
            lcd_display_merged(f"{plate_text_ocr[:8]} Exiting", "Grace Period", clear_first=True)
            beep(duration=0.2, count=2)
            log_parking_event(plate_text_ocr, "EXIT_UNREGISTERED_GRACE", f"Duration:{parked_duration_hours:.2f}hrs")
            time.sleep(2)

        open_gate_parking(servo_exit_pwm, "Exit", plate_text_ocr)
        gate_action_taken = True
        del active_unregistered_parkers[plate_text_ocr]
        save_active_parkers()
    elif is_registered_db:
        print(f"[REGISTERED] '{plate_text_ocr}' Exiting. Access Granted.")
        lcd_display_merged(plate_text_ocr[:8].ljust(8), "Exit Granted", clear_first=True)
        beep(duration=0.3, count=1)
        log_parking_event(plate_text_ocr, "EXIT_REGISTERED_GRANTED")
        open_gate_parking(servo_exit_pwm, "Exit", plate_text_ocr)
        gate_action_taken = True
    else:
        print(f"[UNKNOWN PLATE] '{plate_text_ocr}' at exit. Not in DB, not in temp list.")
        lcd_display_merged(plate_text_ocr[:8].ljust(8), "Plate Unknown!", clear_first=True)
        beep(duration=0.1, count=5, delay_between=0.05)
        log_parking_event(plate_text_ocr, "EXIT_UNKNOWN_PLATE_DENIED")
        time.sleep(2)

    if gate_action_taken:
        time.sleep(GATE_OPEN_DURATION_PARKING)
        close_gate_parking(servo_exit_pwm, "Exit")
        exit_gate_busy = False
        print("SERVODBG: run_anpr_exit_sequence END (Gate Cycled)")
        return True
    else:
        exit_gate_busy = False
        print("SERVODBG: run_anpr_exit_sequence END (No Gate Action)")
        return False


# --- MERGED Main Logic ---
def merged_main_loop():
    global entry_gate_busy, exit_gate_busy, entry_anpr_processing_active, exit_anpr_processing_active
    global entry_anpr_last_processed_plate, entry_anpr_last_plate_contour_detection_time, entry_anpr_last_process_time
    global exit_anpr_last_processed_plate, exit_anpr_last_plate_contour_detection_time, exit_anpr_last_process_time

    current_time_loop = time.time()

    # ANPR Cooldown Reset Logic
    if current_time_loop - entry_anpr_last_plate_contour_detection_time > RESET_TIMEOUT_ANPR:
        if entry_anpr_last_processed_plate != "":
            print(f"ANPR Entry: Resetting lock for '{entry_anpr_last_processed_plate}' (inactivity).")
            entry_anpr_last_processed_plate = ""
        entry_anpr_last_plate_contour_detection_time = current_time_loop
    if current_time_loop - exit_anpr_last_plate_contour_detection_time > RESET_TIMEOUT_ANPR:
        if exit_anpr_last_processed_plate != "":
            print(f"ANPR Exit: Resetting lock for '{exit_anpr_last_processed_plate}' (inactivity).")
            exit_anpr_last_processed_plate = ""
        exit_anpr_last_plate_contour_detection_time = current_time_loop

    # Gate Sensor Detections
    entry_us_distance = measure_distance(US_ENTRY_SENSOR["trig"], US_ENTRY_SENSOR["echo"], "Entry US")
    entry_vehicle_detected = (entry_us_distance < CAR_PRESENT_THRESHOLD_CM)

    exit_us_distance = measure_distance(US_EXIT_SENSOR["trig"], US_EXIT_SENSOR["echo"], "Exit US")
    exit_vehicle_detected = (exit_us_distance < CAR_PRESENT_THRESHOLD_CM)
    if DEBUG_MODE and entry_us_distance != float('inf') : print(f"Sensor Readings: Entry US: {entry_us_distance:.1f}cm, Exit US: {exit_us_distance:.1f}cm", end='\r')


    # Prioritize managing an already active gate/ANPR process
    if entry_gate_busy or entry_anpr_processing_active or exit_gate_busy or exit_anpr_processing_active:
        pass # Let the active process complete its course; don't start new one.
    else: # If no major processes are active, check for new vehicle detections
        if entry_vehicle_detected:
            print(f"\nMAIN_LOOP: Entry vehicle detected ({entry_us_distance:.1f}cm). Starting entry sequence.")
            entry_anpr_processing_active = True # Set flag *before* calling sequence
            run_anpr_entry_sequence()
            entry_anpr_processing_active = False # Clear flag after sequence returns
        elif exit_vehicle_detected: # Use elif to prevent processing both entry and exit in same loop tick
            print(f"\nMAIN_LOOP: Exit vehicle detected ({exit_us_distance:.1f}cm). Starting exit sequence.")
            exit_anpr_processing_active = True
            run_anpr_exit_sequence()
            exit_anpr_processing_active = False

    # Update parking spot occupancy status (polls US_SENSORS for slots)
    update_parking_spots_status()

    # Display general parking status on LCD if no other actions are ongoing
    if not (entry_gate_busy or exit_gate_busy or entry_anpr_processing_active or exit_anpr_processing_active):
        display_parking_main_status_lcd()


# --- Main Execution ---
if __name__ == "__main__":
    try:
        print("System Starting Up...")
        create_debug_dir(DEBUG_IMG_PATH_ENTRY)
        create_debug_dir(DEBUG_IMG_PATH_EXIT)

        try:
            tesseract_version = pytesseract.get_tesseract_version()
            print(f"Tesseract version: {tesseract_version} found.")
        except pytesseract.TesseractNotFoundError:
            print("[FATAL ERROR] Tesseract OCR not installed or not found in PATH. `sudo apt install tesseract-ocr`")
            exit(1)
        except Exception as e:
            print(f"Error getting Tesseract version: {e}")


        if not os.path.exists('Database.txt'):
            print("[WARNING] Database.txt not found. Creating an empty one.")
            with open('Database.txt', 'w') as f: pass # Create empty file
        else:
            with open('Database.txt', 'r') as f: db_lines = sum(1 for line in f if line.strip())
            print(f"Database.txt found with {db_lines} entries.")

        setup_log_file()
        load_active_parkers()

        setup_gpio()
        setup_lcd() # Sets up DummyLCD if real LCD fails
        setup_cameras()
        setup_servos() # This will print SERVODBG messages

        if not camera_entry_ready:
            lcd_display_merged("Entry Cam FAIL!", "Entry Disabled", clear_first=True); time.sleep(2)
        if not camera_exit_ready: # Applicable if pay-as-you-go is primary
            lcd_display_merged("Exit Cam FAIL!", "Billing N/A", clear_first=True); time.sleep(2)
        if not (camera_entry_ready or camera_exit_ready): # Or simply 'and' if both are vital
            print("[CRITICAL] One or more cameras failed. Functionality will be limited.")


        current_main_time = time.time()
        entry_anpr_last_plate_contour_detection_time = current_main_time
        entry_anpr_last_process_time = current_main_time
        exit_anpr_last_plate_contour_detection_time = current_main_time
        exit_anpr_last_process_time = current_main_time
        last_us_poll_time = 0 # Force immediate first poll

        print("SERVODBG: Initial available_spots_count:", available_spots_count)
        update_parking_spots_status() # Initial poll for parking spots
        print("SERVODBG: available_spots_count after first poll:", available_spots_count)

        lcd_display_merged("System Ready", f"{available_spots_count} Spots Free", clear_first=True)
        print("\nSmart Parking System with Pay-as-you-go Ready. Press Ctrl+C to quit.")
        print("--- ANPR Tuning Parameters ---")
        print(f"  Canny: Low={CANNY_LOW_THRESH}, High={CANNY_HIGH_THRESH}")
        print(f"  Contour: ApproxFactor={CONTOUR_APPROX_FACTOR}, MinArea={MIN_PLATE_AREA}, AR={MIN_ASPECT_RATIO}-{MAX_ASPECT_RATIO}")
        print(f"  OCR Preproc: ResizeH={OCR_RESIZE_HEIGHT}, ThreshMethod='{THRESHOLD_METHOD}', AdaptBlock={ADAPT_THRESH_BLOCK_SIZE}, AdaptC={ADAPT_THRESH_C}")
        print(f"  Tesseract: PSM='{TESS_PSM}', Whitelist='{TESS_WHITELIST}', Pattern='{EXPECTED_PLATE_PATTERN}'")
        print("--- System Live ---")
        time.sleep(1)

        while True:
            merged_main_loop()
            time.sleep(MAIN_LOOP_POLLING_INTERVAL)

    except KeyboardInterrupt:
        print("\nCtrl+C Detected. Shutting down system gracefully...")
    except Exception as e:
        print(f"\n[FATAL UNHANDLED ERROR IN MAIN BLOCK] An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Performing final cleanup...")
        save_active_parkers()
        
        if picam2_entry and hasattr(picam2_entry, 'started') and picam2_entry.started:
            try: picam2_entry.stop_preview(); picam2_entry.stop(); print("Entry Camera stopped.")
            except Exception as cam_e: print(f"Error stopping entry camera: {cam_e}")
        if picam2_exit and hasattr(picam2_exit, 'started') and picam2_exit.started:
            try: picam2_exit.stop_preview(); picam2_exit.stop(); print("Exit Camera stopped.")
            except Exception as cam_e: print(f"Error stopping exit camera: {cam_e}")
        
        if lcd_ready and lcd is not None and not isinstance(lcd, DummyLCD) and hasattr(lcd, 'clear'):
           try:
             lcd_display_merged("System Offline", "Goodbye!", clear_first=True)
             time.sleep(1)
             lcd.clear()
           except: pass
        
        # Stop PWMs before GPIO cleanup
        if servo_entry_pwm:
            try: servo_entry_pwm.stop(); print("Entry Servo PWM stopped.")
            except: pass
        if servo_exit_pwm:
            try: servo_exit_pwm.stop(); print("Exit Servo PWM stopped.")
            except: pass
            
        GPIO.cleanup()
        print("GPIO Cleaned Up. System Exited.")
# PYTHON SCRIPT END
