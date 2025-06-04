# ==============================================================================
#               MERGED SMART PARKING & ANPR BARRIER SYSTEM
#               (Entry IR replaced with 7th Ultrasonic Sensor)
# ==============================================================================
import cv2
import numpy as np
import pytesseract
import RPi.GPIO as GPIO
import time
from picamera2 import Picamera2
from RPLCD.gpio import CharLCD
import re
import os
import datetime

# ==================================
#        CONFIGURATION & TUNING
# ==================================
DEBUG_MODE = True
DEBUG_IMG_PATH = "debug_images_merged"
DATABASE_FILE_PATH = "Database.txt"
ACTIVITY_LOG_FILE_PATH = "activity_log.csv"

# --- Camera (ANPR) ---
IMG_WIDTH = 1024
IMG_HEIGHT = 576

# --- Plate Extraction Tuning (ANPR) ---
# ... (Keep previous values)
CANNY_LOW_THRESH = 50
CANNY_HIGH_THRESH = 180
CONTOUR_APPROX_FACTOR = 0.02
MIN_PLATE_AREA = 500
MIN_ASPECT_RATIO = 1.8
MAX_ASPECT_RATIO = 5.5

# --- OCR Preprocessing Tuning (ANPR) ---
# ... (Keep previous values)
OCR_RESIZE_HEIGHT = 60
THRESHOLD_METHOD = 'ADAPTIVE' # 'ADAPTIVE' or 'OTSU'
ADAPT_THRESH_BLOCK_SIZE = 19
ADAPT_THRESH_C = 9


# --- Tesseract Tuning (ANPR) ---
# ... (Keep previous values)
TESS_LANG = 'eng'
TESS_OEM = 3
TESS_PSM = '7'
TESS_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
EXPECTED_PLATE_PATTERN = ""


# --- OCR Post-Processing / Validation (ANPR) ---
MIN_PLATE_LENGTH = 5

# --- ANPR Main Loop Timing ---
PROCESS_COOLDOWN_ANPR = 8
RESET_TIMEOUT_ANPR = 15

# --- Parking System Configuration ---
CAR_PRESENT_THRESHOLD_CM = 30
SERVO_FREQUENCY = 50

# === SERVO ANGLE CONFIGURATION (PHYSICAL MEANING) ===
# These define the *physical state* of the gate.
# Calibrate these for your barrier's physical movement.
SERVO_PHYSICALLY_CLOSED_ANGLE = 0   # Servo angle that makes the barrier ARM physically DOWN
SERVO_PHYSICALLY_OPEN_ANGLE = 90    # Servo angle that makes the barrier ARM physically UP
# ======================================================

# === SERVO INVERSION FLAGS ===
# Set to True if the servo needs reversed angles to achieve the physical open/closed state
# For example, if to make the EXIT gate arm physically UP, the servo needs to go to SERVO_PHYSICALLY_CLOSED_ANGLE
ENTRY_GATE_SERVO_INVERTED = False # Usually False for the primary/reference gate
EXIT_GATE_SERVO_INVERTED = False  # <<<<<<< CHANGE THIS TO True IF YOUR EXIT SERVO IS MOUNTED REVERSEDLY
# ============================

SERVO_MOVE_DELAY = 1.0
GATE_OPEN_DURATION_PARKING = 3.0
US_POLLING_INTERVAL = 1.0
IR_POLLING_INTERVAL_MAIN_LOOP = 0.1

# ==================================
#           PIN DEFINITIONS (BCM Mode)
# ==================================
# ... (Keep previous pin definitions)
LCD_RS_PIN = 7
LCD_E_PIN = 8
LCD_D4_PIN = 25
LCD_D5_PIN = 24
LCD_D6_PIN = 23
LCD_D7_PIN = 12
IR_EXIT_PIN = 22
SERVO_ENTRY_PIN = 17
SERVO_EXIT_PIN = 4
BUZZ_PIN_ANPR = 2
US_ENTRY_SENSOR = {"name": "Entry Detect", "trig": 0, "echo": 1}
US_SENSORS = [
    {"name": "Slot 1", "trig": 5,  "echo": 6},
    {"name": "Slot 2", "trig": 19, "echo": 26},
    {"name": "Slot 3", "trig": 20, "echo": 21},
    {"name": "Slot 4", "trig": 16, "echo": 13},
    {"name": "Slot 5", "trig": 10, "echo": 9},
    {"name": "Slot 6", "trig": 11, "echo": 18}
]
TOTAL_PARKING_SPOTS = len(US_SENSORS)
# ==================================

# --- Global Variables & Setup Code (largely unchanged) ---
if DEBUG_MODE and not os.path.exists(DEBUG_IMG_PATH):
    try:
        os.makedirs(DEBUG_IMG_PATH)
        print(f"Created debug directory: {DEBUG_IMG_PATH}")
    except OSError as e:
        print(f"[ERROR] Could not create debug directory '{DEBUG_IMG_PATH}': {e}")
        DEBUG_MODE = False

lcd = None
lcd_ready = False
picam2 = None
camera_ready = False
servo_entry_pwm = None
servo_exit_pwm = None
occupied_spots_status = [False] * TOTAL_PARKING_SPOTS
previous_occupied_spots_status = [False] * TOTAL_PARKING_SPOTS
available_spots_count = TOTAL_PARKING_SPOTS
last_us_poll_time = 0
entry_gate_busy = False
exit_gate_busy = False
anpr_last_processed_plate = ""
anpr_last_process_time = 0
anpr_last_plate_contour_detection_time = 0
anpr_processing_active = False

# ... (initialize_log_file, log_event, setup_lcd, lcd_display_merged, setup_camera, setup_gpio - are the same)
def initialize_log_file():
    if not os.path.exists(ACTIVITY_LOG_FILE_PATH):
        try:
            with open(ACTIVITY_LOG_FILE_PATH, "w") as f:
                f.write("Timestamp,EventType,Plate,Status\n")
            print(f"Activity log file created: {ACTIVITY_LOG_FILE_PATH}")
        except IOError as e:
            print(f"[ERROR] Could not create activity log file {ACTIVITY_LOG_FILE_PATH}: {e}")

def log_event(event_type, plate_text="N/A", status_message=""):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plate_text_csv = str(plate_text).replace(',', ';')
    status_message_csv = str(status_message).replace(',', ';')
    log_entry = f"{timestamp},{event_type},{plate_text_csv},{status_message_csv}\n"
    try:
        with open(ACTIVITY_LOG_FILE_PATH, "a") as f:
            f.write(log_entry)
        if DEBUG_MODE:
            print(f"LOGGED: {timestamp} | {event_type} | Plate: {plate_text} | Status: {status_message}")
    except Exception as e:
        print(f"[ERROR] Could not write to log file {ACTIVITY_LOG_FILE_PATH}: {e}")

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
        class DummyLCD:
            def write_string(self, text): pass
            def clear(self): pass
            def cursor_pos(self, pos): pass
        lcd = DummyLCD()

def lcd_display_merged(line1, line2="", clear_first=True):
    if not lcd_ready: return
    try:
        if clear_first: lcd.clear()
        lcd.cursor_pos = (0, 0)
        lcd.write_string(str(line1)[:16])
        if line2:
            lcd.cursor_pos = (1, 0)
            lcd.write_string(str(line2)[:16])
    except Exception as e:
        print(f"[ERROR] LCD display error: {e}")

def setup_camera():
    global picam2, camera_ready
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (IMG_WIDTH, IMG_HEIGHT)})
        picam2.configure(config)
        picam2.start()
        time.sleep(2.0)
        camera_ready = True
        print("Camera Initialized for ANPR.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Camera: {e}")
        camera_ready = False

def setup_gpio():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(IR_EXIT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(US_ENTRY_SENSOR["trig"], GPIO.OUT)
    GPIO.setup(US_ENTRY_SENSOR["echo"], GPIO.IN)
    GPIO.output(US_ENTRY_SENSOR["trig"], False)
    for sensor in US_SENSORS:
        GPIO.setup(sensor["trig"], GPIO.OUT)
        GPIO.setup(sensor["echo"], GPIO.IN)
        GPIO.output(sensor["trig"], False)
    GPIO.setup(SERVO_ENTRY_PIN, GPIO.OUT)
    GPIO.setup(SERVO_EXIT_PIN, GPIO.OUT)
    GPIO.setup(BUZZ_PIN_ANPR, GPIO.OUT, initial=GPIO.LOW)
    print("GPIO Initialized.")

# --- Servo Control (Parking System Based) ---
def setup_servos():
    global servo_entry_pwm, servo_exit_pwm
    try:
        servo_entry_pwm = GPIO.PWM(SERVO_ENTRY_PIN, SERVO_FREQUENCY)
        servo_exit_pwm = GPIO.PWM(SERVO_EXIT_PIN, SERVO_FREQUENCY)
        servo_entry_pwm.start(0)
        servo_exit_pwm.start(0)
        
        print("Initializing servos to physically closed positions...")
        
        # Determine initial closed angle for Entry gate
        initial_entry_closed_angle = SERVO_PHYSICALLY_CLOSED_ANGLE
        if ENTRY_GATE_SERVO_INVERTED:
            initial_entry_closed_angle = SERVO_PHYSICALLY_OPEN_ANGLE # If inverted, "closed" means go to "open" angle
        set_servo_angle_parking(servo_entry_pwm, initial_entry_closed_angle, "Entry Initial Close")

        # Determine initial closed angle for Exit gate
        initial_exit_closed_angle = SERVO_PHYSICALLY_CLOSED_ANGLE
        if EXIT_GATE_SERVO_INVERTED:
            initial_exit_closed_angle = SERVO_PHYSICALLY_OPEN_ANGLE
        set_servo_angle_parking(servo_exit_pwm, initial_exit_closed_angle, "Exit Initial Close")
        
        print("Servos Initialized.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize servos: {e}")

def set_servo_angle_parking(servo_pwm, angle, gate_name_debug=""):
    if servo_pwm is None:
        print(f"[WARN] Servo {gate_name_debug} not available for angle set.")
        return
    if not (0 <= angle <= 180): # Basic servo angle range
        print(f"[WARN] Servo command angle {angle} for {gate_name_debug} is out of 0-180 range. Clamping.")
        angle = max(0, min(180, angle))
    duty = (angle / 180.0) * 5.0 + 5.0
    servo_pwm.ChangeDutyCycle(duty)
    if DEBUG_MODE:
        print(f"SERVO {gate_name_debug}: Commanded to angle {angle} -> duty {duty:.2f}%")
    time.sleep(SERVO_MOVE_DELAY)

# Gate control functions now consider inversion
def open_gate_parking(servo_pwm_obj, gate_name, is_inverted_flag):
    target_servo_angle = SERVO_PHYSICALLY_OPEN_ANGLE
    if is_inverted_flag:
        target_servo_angle = SERVO_PHYSICALLY_CLOSED_ANGLE # To physically open, an inverted servo goes to the "closed" angle definition
    
    display_line1 = f"{gate_name} Gate"
    display_line2 = "Opening..."
    lcd_display_merged(display_line1, display_line2, clear_first=True)
    print(f"Opening {gate_name} gate (Servo target: {target_servo_angle})...")
    set_servo_angle_parking(servo_pwm_obj, target_servo_angle, f"{gate_name} Open")
    print(f"{gate_name} gate PHYSICALLY OPEN.")

def close_gate_parking(servo_pwm_obj, gate_name, is_inverted_flag):
    target_servo_angle = SERVO_PHYSICALLY_CLOSED_ANGLE
    if is_inverted_flag:
        target_servo_angle = SERVO_PHYSICALLY_OPEN_ANGLE # To physically close, an inverted servo goes to the "open" angle definition

    lcd_display_merged(f"{gate_name} Gate", "Closing...", clear_first=True)
    print(f"Closing {gate_name} gate (Servo target: {target_servo_angle})...")
    set_servo_angle_parking(servo_pwm_obj, target_servo_angle, f"{gate_name} Close")
    print(f"{gate_name} gate PHYSICALLY CLOSED.")

# ... (Keep: read_ir_sensor, measure_distance, perspective_transform, capture_image_anpr, plate_extraction_anpr, ocr_processing_anpr, check_database_anpr, update_parking_spots_status, display_parking_main_status_lcd)
def read_ir_sensor(pin):
    return GPIO.input(pin) == GPIO.LOW

def measure_distance(trig_pin, echo_pin):
    GPIO.output(trig_pin, False)
    time.sleep(0.01)
    GPIO.output(trig_pin, True)
    time.sleep(0.00001)
    GPIO.output(trig_pin, False)
    start_time, end_time = time.time(), time.time()
    timeout_limit = 0.1
    loop_start_time = time.time()
    while GPIO.input(echo_pin) == 0:
        start_time = time.time()
        if start_time - loop_start_time > timeout_limit: return float('inf')
    loop_start_time = time.time()
    while GPIO.input(echo_pin) == 1:
        end_time = time.time()
        if end_time - loop_start_time > timeout_limit: return float('inf')
    duration = end_time - start_time
    distance = (duration * 34300) / 2
    return distance if distance >= 0 else float('inf')

def perspective_transform(image, pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    if maxWidth <= 0 or maxHeight <= 0: return None
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def capture_image_anpr():
    if not camera_ready or picam2 is None:
        print("[ERROR-ANPR] Camera not ready for capture.")
        return None
    try:
        frame = picam2.capture_array("main")
        if frame.shape[2] == 4:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        elif frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            print(f"[ERROR-ANPR] Unexpected frame format: {frame.shape}")
            return None
        return frame_bgr
    except Exception as e:
        print(f"[ERROR-ANPR] Image capture failed: {e}")
        return None

def plate_extraction_anpr(image_color):
    if image_color is None or image_color.size == 0: return None
    ts_for_debug = int(time.time())
    gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    if DEBUG_MODE: cv2.imwrite(os.path.join(DEBUG_IMG_PATH, f"{ts_for_debug}_anpr_00_gray.png"), gray)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    if DEBUG_MODE: cv2.imwrite(os.path.join(DEBUG_IMG_PATH, f"{ts_for_debug}_anpr_00b_blur.png"), blur)
    edges = cv2.Canny(blur, CANNY_LOW_THRESH, CANNY_HIGH_THRESH)
    if DEBUG_MODE: cv2.imwrite(os.path.join(DEBUG_IMG_PATH, f"{ts_for_debug}_anpr_01_edges.png"), edges)
    cnts, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    plate_contour_found = None
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, CONTOUR_APPROX_FACTOR * perimeter, True)
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h) if h > 0 else 0
            area = cv2.contourArea(approx)
            if MIN_PLATE_AREA < area and MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO:
                plate_contour_found = approx
                if DEBUG_MODE:
                     debug_frame = image_color.copy()
                     cv2.drawContours(debug_frame, [plate_contour_found], -1, (0, 255, 0), 3)
                     cv2.imwrite(os.path.join(DEBUG_IMG_PATH, f"{ts_for_debug}_anpr_02_contour.png"), debug_frame)
                break
    if plate_contour_found is None:
        if DEBUG_MODE: print("  ANPR plate_extraction: No suitable contour found.")
        return None
    pts = plate_contour_found.reshape(4, 2)
    warped_plate_gray = perspective_transform(gray, pts)
    if warped_plate_gray is None or warped_plate_gray.size == 0:
        if DEBUG_MODE: print("  ANPR plate_extraction: Warped plate is empty.")
        return None
    if DEBUG_MODE: cv2.imwrite(os.path.join(DEBUG_IMG_PATH, f"{ts_for_debug}_anpr_03_warped.png"), warped_plate_gray)
    return warped_plate_gray

def ocr_processing_anpr(plate_image_gray):
    global anpr_last_plate_contour_detection_time
    if plate_image_gray is None or plate_image_gray.size == 0: return ""
    anpr_last_plate_contour_detection_time = time.time()
    try:
        h_orig, w_orig = plate_image_gray.shape[:2]
        aspect_ratio_orig = w_orig / float(h_orig) if h_orig > 0 else 1.0
        target_width = int(OCR_RESIZE_HEIGHT * aspect_ratio_orig)
        if target_width > 0 and OCR_RESIZE_HEIGHT > 0:
            plate_image_resized = cv2.resize(plate_image_gray, (target_width, OCR_RESIZE_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
        else: plate_image_resized = plate_image_gray
    except Exception as e:
        print(f"[WARN-ANPR] Plate resize failed: {e}. Using original.")
        plate_image_resized = plate_image_gray
    if THRESHOLD_METHOD == 'ADAPTIVE':
        binary_plate = cv2.adaptiveThreshold(plate_image_resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPT_THRESH_BLOCK_SIZE, ADAPT_THRESH_C)
    elif THRESHOLD_METHOD == 'OTSU':
        _, binary_plate = cv2.threshold(plate_image_resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, binary_plate = cv2.threshold(plate_image_resized, 127, 255, cv2.THRESH_BINARY_INV)
    ts_for_debug = int(time.time())
    if DEBUG_MODE: cv2.imwrite(os.path.join(DEBUG_IMG_PATH, f"{ts_for_debug}_anpr_04_binary.png"), binary_plate)
    custom_config = f'--oem {TESS_OEM} --psm {TESS_PSM} -l {TESS_LANG}'
    if TESS_WHITELIST: custom_config += f' -c tessedit_char_whitelist={TESS_WHITELIST}'
    try:
        raw_text = pytesseract.image_to_string(binary_plate, config=custom_config)
        if DEBUG_MODE: print(f"  ANPR Raw OCR: '{raw_text.strip()}'")
        cleaned_text = ''.join(filter(str.isalnum, raw_text)).upper()
        if not cleaned_text:
             if DEBUG_MODE: print(f"  ANPR Reject (empty after clean): '{raw_text.strip()}'")
             return ""
        if len(cleaned_text) < MIN_PLATE_LENGTH:
            if DEBUG_MODE: print(f"  ANPR Reject (short): '{cleaned_text}' len {len(cleaned_text)}")
            return ""
        if EXPECTED_PLATE_PATTERN and not re.fullmatch(EXPECTED_PLATE_PATTERN, cleaned_text):
            if DEBUG_MODE: print(f"  ANPR Reject (pattern mismatch): '{cleaned_text}' vs pattern '{EXPECTED_PLATE_PATTERN}'")
            return ""
        return cleaned_text
    except pytesseract.TesseractNotFoundError:
        print("[ERROR-ANPR] Tesseract not found. Please ensure it's installed and in PATH.")
        return ""
    except Exception as e:
        print(f"[ERROR-ANPR] OCR processing failed: {e}")
        return ""

def check_database_anpr(plate_text):
    if not plate_text: return False
    try:
        with open(DATABASE_FILE_PATH, 'r') as f:
            database_plates = {line.strip().upper() for line in f if line.strip()}
        return plate_text in database_plates
    except FileNotFoundError:
        print(f"[ERROR-ANPR] {DATABASE_FILE_PATH} not found! Attempting to create an empty one.")
        try:
            with open(DATABASE_FILE_PATH, 'w') as f: pass
            print(f"[INFO-ANPR] Created empty {DATABASE_FILE_PATH}. Add plate numbers to it.")
        except IOError: print(f"[ERROR-ANPR] Could not create {DATABASE_FILE_PATH} due to IO error.")
        return False
    except Exception as e:
        print(f"[ERROR-ANPR] Database read error: {e}")
        return False

def update_parking_spots_status():
    global occupied_spots_status, previous_occupied_spots_status, available_spots_count, last_us_poll_time
    current_time_us_poll = time.time()
    if current_time_us_poll - last_us_poll_time < US_POLLING_INTERVAL:
        return
    new_occupied_count = 0
    changed_slots_this_poll = False
    for i, sensor_info in enumerate(US_SENSORS):
        dist = measure_distance(sensor_info["trig"], sensor_info["echo"])
        current_spot_is_occupied = (dist < CAR_PRESENT_THRESHOLD_CM)
        if current_spot_is_occupied != previous_occupied_spots_status[i]:
            changed_slots_this_poll = True
            print(f"PARKING: Slot {sensor_info['name']} is now {'OCCUPIED' if current_spot_is_occupied else 'EMPTY'} (Dist: {dist:.1f}cm)")
        occupied_spots_status[i] = current_spot_is_occupied
        if current_spot_is_occupied: new_occupied_count += 1
    new_available_spots = TOTAL_PARKING_SPOTS - new_occupied_count
    if new_available_spots != available_spots_count:
        print(f"PARKING: Available spots updated from {available_spots_count} to {new_available_spots}")
        available_spots_count = new_available_spots
    if changed_slots_this_poll:
        previous_occupied_spots_status = list(occupied_spots_status)
    last_us_poll_time = current_time_us_poll

def display_parking_main_status_lcd():
    global available_spots_count
    if anpr_processing_active or entry_gate_busy or exit_gate_busy: return
    status_line1 = "Spots Available"
    status_line2 = f"{available_spots_count} Free"
    if available_spots_count == TOTAL_PARKING_SPOTS:
        status_line1 = "Parking Empty"
        status_line2 = f"{available_spots_count} Free"
    elif available_spots_count <= 0:
        available_spots_count = 0
        status_line1 = "Parking Full!"
        status_line2 = "No Spots Free"
    lcd_display_merged(status_line1, status_line2, clear_first=True)

# Gate operations now pass the relevant inversion flag
def run_anpr_entry_sequence():
    global anpr_last_processed_plate, anpr_last_process_time, available_spots_count
    print("\nANPR: Car detected at entry (US). Initiating plate recognition...")
    lcd_display_merged("Car at Entry", "Reading Plate...", clear_first=True)
    frame_color = capture_image_anpr()
    if frame_color is None:
        lcd_display_merged("Camera Error", "Try Again Soon", clear_first=True)
        print("ANPR: Failed to capture frame for entry.")
        log_event("ENTRY", status_message="Camera Error")
        time.sleep(2); return False
        
    extracted_plate_img_gray = plate_extraction_anpr(frame_color)
    if extracted_plate_img_gray is not None:
        plate_text_ocr = ocr_processing_anpr(extracted_plate_img_gray)
        if plate_text_ocr:
            current_time = time.time()
            if (plate_text_ocr == anpr_last_processed_plate) and \
               (current_time - anpr_last_process_time < PROCESS_COOLDOWN_ANPR):
                print(f"ANPR: Ignoring '{plate_text_ocr}' (recently processed / cooldown).")
                return False
            print(f"ANPR Detected: '{plate_text_ocr}'", end=' ')
            lcd_display_merged("Plate: " + plate_text_ocr, "Checking DB...", clear_first=True)
            time.sleep(0.5)
            is_registered = check_database_anpr(plate_text_ocr)
            anpr_last_processed_plate = plate_text_ocr
            anpr_last_process_time = time.time()
            if is_registered:
                if available_spots_count > 0:
                    print("[REGISTERED] ✅ Access Granted!")
                    lcd_display_merged(plate_text_ocr, "Access Granted", clear_first=True)
                    log_event("ENTRY", plate_text_ocr, "Access Granted")
                    GPIO.output(BUZZ_PIN_ANPR, GPIO.HIGH); time.sleep(0.5); GPIO.output(BUZZ_PIN_ANPR, GPIO.LOW)
                    
                    open_gate_parking(servo_entry_pwm, "Entry", ENTRY_GATE_SERVO_INVERTED)
                    time.sleep(GATE_OPEN_DURATION_PARKING)
                    close_gate_parking(servo_entry_pwm, "Entry", ENTRY_GATE_SERVO_INVERTED)
                    
                    print(f"ANPR -> Granted '{plate_text_ocr}'. Gate operated. Cooldown started.")
                    return True
                else:
                    # ... (parking full logic)
                    print("[REGISTERED] BUT PARKING FULL ❌ Access Denied!")
                    lcd_display_merged(plate_text_ocr, "Parking Full!", clear_first=True)
                    log_event("ENTRY", plate_text_ocr, "Denied - Parking Full")
                    for _ in range(3): GPIO.output(BUZZ_PIN_ANPR, GPIO.HIGH); time.sleep(0.1); GPIO.output(BUZZ_PIN_ANPR, GPIO.LOW); time.sleep(0.1)
                    time.sleep(1.5)
                    return False
            else:
                # ... (unregistered logic)
                print("[UNREGISTERED] ❌ Access Denied!")
                lcd_display_merged(plate_text_ocr, "Access Denied", clear_first=True)
                log_event("ENTRY", plate_text_ocr, "Denied - Unregistered")
                for _ in range(3): GPIO.output(BUZZ_PIN_ANPR, GPIO.HIGH); time.sleep(0.1); GPIO.output(BUZZ_PIN_ANPR, GPIO.LOW); time.sleep(0.1)
                time.sleep(1.5)
                return False
        else: # OCR Failed
            print("ANPR: Contour found, but OCR failed to read plate text.")
            lcd_display_merged("Plate Found", "OCR Failed", clear_first=True)
            log_event("ENTRY", status_message="Denied - OCR Failed")
            time.sleep(1.5)
            return False
    else: # No plate contour
        print("ANPR: No plate-like contour detected in the image.")
        lcd_display_merged("No Plate Found", "Try Reposition", clear_first=True)
        log_event("ENTRY", status_message="Denied - No Plate Detected")
        time.sleep(1.5)
        return False

def merged_main_loop():
    global entry_gate_busy, exit_gate_busy, anpr_processing_active
    global anpr_last_processed_plate, anpr_last_plate_contour_detection_time, anpr_last_process_time, available_spots_count
    current_time = time.time()

    if anpr_last_processed_plate and (current_time - anpr_last_plate_contour_detection_time > RESET_TIMEOUT_ANPR):
        print(f"\nANPR: Resetting lock for plate '{anpr_last_processed_plate}' due to no plate contour detection for {RESET_TIMEOUT_ANPR}s.")
        anpr_last_processed_plate = ""
        if not (anpr_processing_active or entry_gate_busy or exit_gate_busy):
             display_parking_main_status_lcd()
    
    entry_us_distance = measure_distance(US_ENTRY_SENSOR["trig"], US_ENTRY_SENSOR["echo"])
    entry_vehicle_detected = (entry_us_distance < CAR_PRESENT_THRESHOLD_CM)

    if entry_vehicle_detected and not entry_gate_busy and not anpr_processing_active and not exit_gate_busy:
        if DEBUG_MODE: print(f"DEBUG: Entry US detected vehicle at {entry_us_distance:.1f}cm. Threshold: {CAR_PRESENT_THRESHOLD_CM}cm")
        anpr_processing_active = True
        entry_gate_busy = True
        run_anpr_entry_sequence() # This now uses ENTRY_GATE_SERVO_INVERTED
        anpr_processing_active = False
        entry_gate_busy = False
        display_parking_main_status_lcd()
    elif not entry_vehicle_detected and DEBUG_MODE and entry_us_distance != float('inf') and entry_us_distance < (CAR_PRESENT_THRESHOLD_CM + 70):
        pass

    exit_ir_active = read_ir_sensor(IR_EXIT_PIN)
    if exit_ir_active and not exit_gate_busy and not entry_gate_busy and not anpr_processing_active:
        exit_gate_busy = True
        print("\nPARKING: Car detected at exit (IR).")
        lcd_display_merged("Car Exiting...", "Gate Opening", clear_first=True)
        
        open_gate_parking(servo_exit_pwm, "Exit", EXIT_GATE_SERVO_INVERTED)
        time.sleep(GATE_OPEN_DURATION_PARKING)
        close_gate_parking(servo_exit_pwm, "Exit", EXIT_GATE_SERVO_INVERTED)
        
        log_event("EXIT", status_message="Vehicle Exited - Gate Cycled")
        lcd_display_merged("Car Exited", "Thank You!", clear_first=True); time.sleep(1)
        exit_gate_busy = False
        display_parking_main_status_lcd()

    update_parking_spots_status()
    if not (entry_gate_busy or exit_gate_busy or anpr_processing_active):
        display_parking_main_status_lcd()


# --- Main Execution ---
if __name__ == "__main__":
    try:
        # ... (Tesseract check, Database file check - same as before)
        try:
            tesseract_version = pytesseract.get_tesseract_version()
            print(f"Tesseract version: {tesseract_version} found.")
        except pytesseract.TesseractNotFoundError:
            print("[FATAL ERROR] Tesseract OCR not installed or not found in PATH.")
            print("Please install Tesseract: sudo apt-get install tesseract-ocr")
            exit(1)

        if not os.path.exists(DATABASE_FILE_PATH):
            print(f"[WARNING] {DATABASE_FILE_PATH} not found. Creating an empty one.")
            try:
                with open(DATABASE_FILE_PATH, 'w') as f: pass
                print(f"[INFO] Created empty {DATABASE_FILE_PATH}. Add whitelisted plates to it, one per line.")
            except IOError as e: print(f"[ERROR] Could not create {DATABASE_FILE_PATH}: {e}")
        else:
            with open(DATABASE_FILE_PATH, 'r') as f: db_lines = len(f.readlines())
            print(f"{DATABASE_FILE_PATH} found with {db_lines} entries.")


        initialize_log_file()
        setup_gpio()
        setup_lcd()
        setup_camera()
        setup_servos() # This now considers inversion for initial setup

        if not camera_ready:
            lcd_display_merged("ANPR Cam FAIL!", "Entry Disabled", clear_first=True)
            print("[CRITICAL] ANPR Camera failed to initialize. Entry via ANPR will not work.")

        anpr_last_plate_contour_detection_time = time.time()
        anpr_last_process_time = time.time()

        log_event("SYSTEM", status_message="System Startup")
        lcd_display_merged("System Ready", f"{available_spots_count} Spots Free", clear_first=True)
        print("\nCombined Parking & ANPR System Ready. Press Ctrl+C to quit.")
        print(f"Entry gate servo inverted: {ENTRY_GATE_SERVO_INVERTED}") # Info
        print(f"Exit gate servo inverted: {EXIT_GATE_SERVO_INVERTED}")   # Info
        time.sleep(1)

        while True:
            merged_main_loop()
            time.sleep(IR_POLLING_INTERVAL_MAIN_LOOP)

    except KeyboardInterrupt:
        print("\nCtrl+C Detected. Shutting down system...")
        log_event("SYSTEM", status_message="System Shutdown Initiated")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        log_event("SYSTEM", status_message=f"CRITICAL ERROR: {e}")
    finally:
        print("Cleaning up all resources...")
        if lcd_ready and lcd is not None and hasattr(lcd, 'clear') and not isinstance(lcd, DummyLCD):
           try:
             lcd_display_merged("System Offline", "Goodbye!", clear_first=True)
             time.sleep(1)
             lcd.clear()
           except Exception as lcd_e: print(f"Error clearing LCD: {lcd_e}")
        
        # Command servos to their respective physical closed positions before stopping
        if servo_entry_pwm:
            print("Closing entry servo before exit...")
            final_entry_close_angle = SERVO_PHYSICALLY_CLOSED_ANGLE
            if ENTRY_GATE_SERVO_INVERTED:
                final_entry_close_angle = SERVO_PHYSICALLY_OPEN_ANGLE
            set_servo_angle_parking(servo_entry_pwm, final_entry_close_angle, "Entry Final Close")
            time.sleep(0.5)
            servo_entry_pwm.stop()
            
        if servo_exit_pwm:
            print("Closing exit servo before exit...")
            final_exit_close_angle = SERVO_PHYSICALLY_CLOSED_ANGLE
            if EXIT_GATE_SERVO_INVERTED:
                final_exit_close_angle = SERVO_PHYSICALLY_OPEN_ANGLE
            set_servo_angle_parking(servo_exit_pwm, final_exit_close_angle, "Exit Final Close")
            time.sleep(0.5)
            servo_exit_pwm.stop()
            
        if camera_ready and picam2:
            try:
                print("Stopping camera...")
                picam2.stop()
                print("Camera stopped.")
            except Exception as cam_e: print(f"Error stopping camera: {cam_e}")
        
        GPIO.cleanup()
        print("GPIO Cleaned Up. System Exited.")
        if os.path.exists(ACTIVITY_LOG_FILE_PATH):
             log_event("SYSTEM", status_message="System Exited Cleanly")
