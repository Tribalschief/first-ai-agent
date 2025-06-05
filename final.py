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

# ==================================
#        CONFIGURATION & TUNING
# ==================================
DEBUG_MODE = True
DEBUG_IMG_PATH = "debug_images_merged"

# --- Camera (ANPR) ---
IMG_WIDTH = 1024
IMG_HEIGHT = 576

# --- Plate Extraction Tuning (ANPR) ---
CANNY_LOW_THRESH = 50
CANNY_HIGH_THRESH = 180
CONTOUR_APPROX_FACTOR = 0.02
MIN_PLATE_AREA = 500
MIN_ASPECT_RATIO = 1.8
MAX_ASPECT_RATIO = 5.5

# --- OCR Preprocessing Tuning (ANPR) ---
OCR_RESIZE_HEIGHT = 60
THRESHOLD_METHOD = 'ADAPTIVE' # 'ADAPTIVE' or 'OTSU'
ADAPT_THRESH_BLOCK_SIZE = 19 # Must be odd
ADAPT_THRESH_C = 9
# SERVO_CLOSED_ANGLE = 90 # Removed duplicate/conflicting definition
# SERVO_OPEN_ANGLE = 0   # Removed duplicate/conflicting definition
CAR_PRESENT_THRESHOLD_CM = 10 # Used for general car presence, main threshold


# --- Tesseract Tuning (ANPR) ---
TESS_LANG = 'eng'
TESS_OEM = 3
TESS_PSM = '7' # Try '8' for single word, '7' for single line of text
TESS_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
EXPECTED_PLATE_PATTERN = "" # Example: r"^[A-Z]{2}[0-9]{2}[A-Z]{3}$" (UK Style)

# --- OCR Post-Processing / Validation (ANPR) ---
MIN_PLATE_LENGTH = 5

# --- ANPR Main Loop Timing ---
PROCESS_COOLDOWN_ANPR = 8 # Seconds before processing the SAME plate again
RESET_TIMEOUT_ANPR = 15   # Seconds of no plate contour detection before resetting anpr_last_processed_plate lock

# --- Parking System Configuration ---
# CAR_PRESENT_THRESHOLD_CM already defined above, now used for parking slots AND the new entry US sensor
SERVO_FREQUENCY = 50
SERVO_CLOSED_ANGLE = 0  # Actual used definition
SERVO_OPEN_ANGLE = 90   # Actual used definition
SERVO_MOVE_DELAY = 1.0      # Delay for servo to reach position
GATE_OPEN_DURATION_PARKING = 3.0 # How long gate stays open for parking events
US_POLLING_INTERVAL = 1.0      # How often to poll parking spot ultrasonic sensors
IR_POLLING_INTERVAL_MAIN_LOOP = 0.1 # Main loop tick rate, also polls entry US sensor

# ==================================
#           PIN DEFINITIONS (BCM Mode) - MATCHING USER IMAGES
# ==================================
# --- LCD Pins ---
LCD_RS_PIN = 7    # BCM 7 (Physical 26)
LCD_E_PIN = 8     # BCM 8 (Physical 24)
LCD_D4_PIN = 25   # BCM 25 (Physical 22)
LCD_D5_PIN = 24   # BCM 24 (Physical 18)
LCD_D6_PIN = 23   # BCM 23 (Physical 16)
LCD_D7_PIN = 12   # BCM 12 (Physical 32)

# --- IR Sensor Pin (Exit Only) ---
# IR_ENTRY_PIN = 27 # BCM 27 (Physical 13) -- REPLACED BY US_ENTRY_SENSOR
IR_EXIT_PIN = 22  # BCM 22 (Physical 15)

# --- Servo Pins ---
SERVO_ENTRY_PIN = 17 # BCM 17 (Physical 11) - (Entry Gate)
SERVO_EXIT_PIN = 4   # BCM 4  (Physical 7)  - (Exit Gate)

# --- ANPR Buzzer Pin ---
BUZZ_PIN_ANPR = 2    # BCM 2 (Physical 3)

# --- Ultrasonic Sensor Pin (FOR ENTRY DETECTION) ---
US_ENTRY_SENSOR = {"name": "Entry Detect", "trig": 0, "echo": 1}

# --- Ultrasonic Sensor Pins (Parking System Slots) ---
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

# --- Create Debug Directory ---
if DEBUG_MODE and not os.path.exists(DEBUG_IMG_PATH):
    try:
        os.makedirs(DEBUG_IMG_PATH)
        print(f"Created debug directory: {DEBUG_IMG_PATH}")
    except OSError as e:
        print(f"[ERROR] Could not create debug directory '{DEBUG_IMG_PATH}': {e}")
        DEBUG_MODE = False

# --- Global Variables (Consolidated) ---
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
entry_us_last_detected = False
entry_us_confirmed = False


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
        class DummyLCD:
            def write_string(self, text): pass
            def clear(self): pass
            def cursor_pos(self, pos): pass
            def lcd_display_merged(self, line1, line2, clear_first=False): pass # Add dummy method
        lcd = DummyLCD()

# Dummy display function if LCD fails or for global use before lcd is confirmed
def lcd_display_merged(line1, line2, clear_first=False):
    global lcd, lcd_ready
    if lcd_ready and lcd:
        try:
            if clear_first: lcd.clear()
            lcd.cursor_pos = (0, 0)
            lcd.write_string(line1.ljust(16)[:16])
            lcd.cursor_pos = (1, 0)
            lcd.write_string(line2.ljust(16)[:16])
        except Exception as e:
            print(f"[ERROR] LCD write error: {e}")
            # lcd_ready = False # Optionally mark LCD as not ready on error
    else:
        # Fallback print if LCD not working
        print(f"LCD_Simulate: L1: {line1} | L2: {line2}")


# --- Camera Setup (ANPR) ---
def setup_camera():
    global picam2, camera_ready
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (IMG_WIDTH, IMG_HEIGHT)})
        picam2.configure(config)
        picam2.start()
        time.sleep(2.0) # Allow camera to initialize
        camera_ready = True
        print("Camera Initialized for ANPR.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Camera: {e}")
        camera_ready = False
        # lcd_display_merged("Camera InitFail", "ANPR May Fail") # Potentially display on LCD

# --- GPIO Setup ---
def setup_gpio():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    # IR Sensor (Exit only)
    GPIO.setup(IR_EXIT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    # Ultrasonic Sensor (Entry Detection)
    GPIO.setup(US_ENTRY_SENSOR["trig"], GPIO.OUT)
    GPIO.setup(US_ENTRY_SENSOR["echo"], GPIO.IN)
    GPIO.output(US_ENTRY_SENSOR["trig"], False)

    # Ultrasonic Sensors (Parking Slots)
    for sensor in US_SENSORS:
        GPIO.setup(sensor["trig"], GPIO.OUT)
        GPIO.setup(sensor["echo"], GPIO.IN)
        GPIO.output(sensor["trig"], False)
    # Servos
    GPIO.setup(SERVO_ENTRY_PIN, GPIO.OUT)
    GPIO.setup(SERVO_EXIT_PIN, GPIO.OUT)
    # ANPR Buzzer
    GPIO.setup(BUZZ_PIN_ANPR, GPIO.OUT, initial=GPIO.LOW)
    print("GPIO Initialized.")

# --- Servo Control ---
def setup_servos():
    global servo_entry_pwm, servo_exit_pwm
    try:
        servo_entry_pwm = GPIO.PWM(SERVO_ENTRY_PIN, SERVO_FREQUENCY)
        servo_exit_pwm = GPIO.PWM(SERVO_EXIT_PIN, SERVO_FREQUENCY)
        servo_entry_pwm.start(0)
        servo_exit_pwm.start(0)
        set_servo_angle_parking(servo_entry_pwm, SERVO_CLOSED_ANGLE, "Entry Initial")
        set_servo_angle_parking(servo_exit_pwm, SERVO_CLOSED_ANGLE, "Exit Initial")
        print("Servos Initialized and Closed.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize servos: {e}")

def set_servo_angle_parking(servo_pwm, angle, gate_name_debug=""):
    if servo_pwm is None:
        print(f"[WARN] Servo {gate_name_debug} not available for angle set.")
        return
    duty = (angle / 18.0) + 2.0
    servo_pwm.ChangeDutyCycle(duty)
    time.sleep(SERVO_MOVE_DELAY)
    servo_pwm.ChangeDutyCycle(0)

def open_gate_parking(servo_pwm_obj, gate_name):
    global anpr_processing_active # to control LCD display logic
    display_line1 = f"{gate_name} Gate"
    display_line2 = "Opening..."
    # Avoid clearing LCD if ANPR is active and this is entry gate, to preserve ANPR messages
    clear_lcd = not (anpr_processing_active and gate_name == "Entry")
    lcd_display_merged(display_line1, display_line2, clear_first=clear_lcd)
    print(f"Opening {gate_name} gate...")
    set_servo_angle_parking(servo_pwm_obj, SERVO_OPEN_ANGLE, gate_name)
    print(f"{gate_name} gate OPEN.")

def close_gate_parking(servo_pwm_obj, gate_name):
    lcd_display_merged(f"{gate_name} Gate", "Closing...")
    print(f"Closing {gate_name} gate...")
    set_servo_angle_parking(servo_pwm_obj, SERVO_CLOSED_ANGLE, gate_name)
    print(f"{gate_name} gate CLOSED.")

# --- IR Sensor Functions ---
def read_ir_sensor(pin):
    return GPIO.input(pin) == GPIO.LOW

# --- Ultrasonic Sensor Functions ---
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

# --- ANPR Image Processing & OCR ---
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
        # Capture as an array, then convert color space if needed
        frame = picam2.capture_array("main") #  Picamera2 captures in RGB by default for "main"
        # OpenCV expects BGR, so convert RGB to BGR
        if frame.shape[2] == 4: # RGBA
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        else: # RGB
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
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
            area = cv2.contourArea(approx) # Use area of the polygon itself
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
    warped_plate_gray = perspective_transform(gray, pts) # Use original gray, not image_color
    if warped_plate_gray is None or warped_plate_gray.size == 0:
        if DEBUG_MODE: print("  ANPR plate_extraction: Warped plate is empty.")
        return None
    if DEBUG_MODE: cv2.imwrite(os.path.join(DEBUG_IMG_PATH, f"{ts_for_debug}_anpr_03_warped.png"), warped_plate_gray)
    return warped_plate_gray

def ocr_processing_anpr(plate_image_gray):
    global anpr_last_plate_contour_detection_time
    if plate_image_gray is None or plate_image_gray.size == 0: return ""
    anpr_last_plate_contour_detection_time = time.time() # Update time if we attempt OCR

    try:
        h_orig, w_orig = plate_image_gray.shape[:2]
        aspect_ratio_orig = w_orig / float(h_orig) if h_orig > 0 else 1.0
        target_width = int(OCR_RESIZE_HEIGHT * aspect_ratio_orig)
        if target_width > 0: # Ensure target_width is positive
            plate_image_resized = cv2.resize(plate_image_gray, (target_width, OCR_RESIZE_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
        else: # If target_width calculation fails, use original image
            plate_image_resized = plate_image_gray
    except Exception as e:
        print(f"[WARN-ANPR] Plate resize failed: {e}. Using original.")
        plate_image_resized = plate_image_gray # Fallback to original if resize fails

    if THRESHOLD_METHOD == 'ADAPTIVE':
        binary_plate = cv2.adaptiveThreshold(plate_image_resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPT_THRESH_BLOCK_SIZE, ADAPT_THRESH_C)
    elif THRESHOLD_METHOD == 'OTSU':
        # Apply Gaussian blur before Otsu for better results
        blurred_for_otsu = cv2.GaussianBlur(plate_image_resized, (5,5), 0)
        _, binary_plate = cv2.threshold(blurred_for_otsu, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else: # Default
        print(f"[WARN-ANPR] Unknown THRESHOLD_METHOD '{THRESHOLD_METHOD}'. Defaulting to ADAPTIVE.")
        binary_plate = cv2.adaptiveThreshold(plate_image_resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPT_THRESH_BLOCK_SIZE, ADAPT_THRESH_C)

    ts_for_debug = int(time.time())
    if DEBUG_MODE: cv2.imwrite(os.path.join(DEBUG_IMG_PATH, f"{ts_for_debug}_anpr_04_binary.png"), binary_plate)

    custom_config = f'--oem {TESS_OEM} --psm {TESS_PSM} -l {TESS_LANG}'
    if TESS_WHITELIST: custom_config += f' -c tessedit_char_whitelist={TESS_WHITELIST}'
    try:
        raw_text = pytesseract.image_to_string(binary_plate, config=custom_config)
        if DEBUG_MODE: print(f"  ANPR Raw OCR: '{raw_text.strip()}'")
        cleaned_text = ''.join(filter(str.isalnum, raw_text)).upper() # Clean and uppercase

        if len(cleaned_text) < MIN_PLATE_LENGTH:
            if DEBUG_MODE and cleaned_text: print(f"  ANPR Reject (short): '{cleaned_text}' len {len(cleaned_text)}")
            return "" # Too short

        if EXPECTED_PLATE_PATTERN and not re.fullmatch(EXPECTED_PLATE_PATTERN, cleaned_text):
            if DEBUG_MODE: print(f"  ANPR Reject (pattern mismatch): '{cleaned_text}' vs pattern '{EXPECTED_PLATE_PATTERN}'")
            return "" # Does not match expected pattern

        return cleaned_text
    except pytesseract.TesseractNotFoundError:
        print("[ERROR-ANPR] Tesseract not found. Please ensure it's installed and in PATH.")
        return ""
    except Exception as e:
        print(f"[ERROR-ANPR] OCR processing failed: {e}")
        return ""

def check_database_anpr(plate_text):
    if not plate_text: return False # Cannot check an empty plate
    try:
        with open('Database.txt', 'r') as f:
            database_plates = {line.strip().upper() for line in f if line.strip()} # Read and uppercase
        return plate_text in database_plates
    except FileNotFoundError:
        print("[ERROR-ANPR] Database.txt not found! Attempting to create an empty one.")
        try:
            with open('Database.txt', 'w') as f: pass # Create empty file
            print("[INFO-ANPR] Created empty Database.txt. Add plate numbers to it.")
        except IOError: print("[ERROR-ANPR] Could not create Database.txt due to IO error.")
        return False # Assume not registered if DB not found
    except Exception as e:
        print(f"[ERROR-ANPR] Database read error: {e}")
        return False

# --- Parking System Logic ---
def update_parking_spots_status():
    global occupied_spots_status, previous_occupied_spots_status, available_spots_count, last_us_poll_time
    current_time_us_poll = time.time()
    if current_time_us_poll - last_us_poll_time < US_POLLING_INTERVAL:
        return # Not time to poll parking spots yet

    new_occupied_count = 0
    changed_slots_this_poll = False
    for i, sensor_info in enumerate(US_SENSORS): # Iterate ONLY parking slot sensors
        dist = measure_distance(sensor_info["trig"], sensor_info["echo"])
        current_spot_is_occupied = (dist < CAR_PRESENT_THRESHOLD_CM and dist != float('inf')) # Check for valid reading too
        if current_spot_is_occupied != previous_occupied_spots_status[i]:
            changed_slots_this_poll = True
            print(f"PARKING: Slot {sensor_info['name']} is now {'OCCUPIED' if current_spot_is_occupied else 'EMPTY'} (Dist: {dist:.1f}cm)")
        occupied_spots_status[i] = current_spot_is_occupied
        if current_spot_is_occupied: new_occupied_count += 1

    available_spots_count = TOTAL_PARKING_SPOTS - new_occupied_count
    if changed_slots_this_poll:
        previous_occupied_spots_status = list(occupied_spots_status) # Update previous status
        # Optionally, display update immediately if status changed and no other critical message is on LCD
        if not (anpr_processing_active or entry_gate_busy or exit_gate_busy):
            display_parking_main_status_lcd() # Update LCD if parking status changed

    last_us_poll_time = current_time_us_poll

def display_parking_main_status_lcd():
    global available_spots_count
    # This function should only update LCD if no other higher priority message is being shown
    if anpr_processing_active or entry_gate_busy or exit_gate_busy:
        return

    status_line1 = "Spots Available"
    status_line2 = f"{available_spots_count} Free"
    if available_spots_count == TOTAL_PARKING_SPOTS:
        status_line1 = "Parking Empty"
    elif available_spots_count == 0:
        status_line1 = "Parking Full!"
        status_line2 = "No Spots Free"
    lcd_display_merged(status_line1, status_line2, clear_first=True)


def run_anpr_entry_sequence():
    global anpr_last_processed_plate, anpr_last_process_time, available_spots_count, camera_ready

    if not camera_ready:
        print("ANPR: Camera not ready. Attempting entry based on availability only.")
        lcd_display_merged("Camera Issue", "Checking Spots", clear_first=True)
        if available_spots_count <= 0:
            lcd_display_merged("Parking Full!", "Entry Blocked", clear_first=True)
            print("[PARKING FULL] No access due to no space (camera inactive).")
            time.sleep(1.5) # Give time to read message
            return False

        lcd_display_merged("Space Available", "Access Granted", clear_first=True)
        GPIO.output(BUZZ_PIN_ANPR, GPIO.HIGH); time.sleep(0.5); GPIO.output(BUZZ_PIN_ANPR, GPIO.LOW)
        open_gate_parking(servo_entry_pwm, "Entry")
        time.sleep(GATE_OPEN_DURATION_PARKING)
        close_gate_parking(servo_entry_pwm, "Entry")

        entry_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open("procedure.txt", "a") as log:
            log.write(f"[ENTRY] Plate: UNKNOWN (No Camera) | Entry Time: {entry_time_str}\n")
        print(f"[NO CAMERA] Generic entry logged at {entry_time_str}.")
        # anpr_last_processed_plate remains as it was (likely empty or stale)
        # No timer file specific to this "UNKNOWN" entry is created to avoid conflicts
        return True # Sequence handled, car allowed in

    # --- Normal ANPR flow if camera is ready ---
    print("\nANPR: Car detected at entry (US). Initiating plate recognition...")
    lcd_display_merged("Car at Entry", "Reading Plate...", clear_first=True)

    frame_color = capture_image_anpr()
    if frame_color is None:
        lcd_display_merged("Camera Error", "Try Again Soon", clear_first=True)
        print("ANPR: Failed to capture frame.")
        time.sleep(2)
        return False

    extracted_plate_img_gray = plate_extraction_anpr(frame_color)
    if extracted_plate_img_gray is not None:
        plate_text = ocr_processing_anpr(extracted_plate_img_gray)
        if plate_text: # A non-empty plate text was recognized
            current_time = time.time()
            if plate_text == anpr_last_processed_plate and (current_time - anpr_last_process_time < PROCESS_COOLDOWN_ANPR):
                print(f"ANPR: Ignoring {plate_text} (cooldown). Plate still locked.")
                # Potentially show "Plate recognized, waiting cooldown"
                lcd_display_merged(f"Plate: {plate_text[:8]}..", "In Cooldown", clear_first=True) # Display part of plate
                time.sleep(1.5)
                return False # Indicate that processing was intentionally skipped

            print(f"ANPR Detected: {plate_text}")
            lcd_display_merged(f"Plate: {plate_text}", "Checking DB...", clear_first=True)
            time.sleep(0.5)

            is_registered = check_database_anpr(plate_text)
            registration_status_msg = "Registered" if is_registered else "Unregistered"
            print(f"ANPR: Plate {plate_text} is {registration_status_msg}.")


            if available_spots_count <= 0:
                lcd_display_merged("Parking Full!", f"{plate_text[:8]}.. Denied", clear_first=True)
                print(f"[PARKING FULL] No access for {plate_text}.")
                time.sleep(1.5)
                return False # Gate does not open if full

            # Open gate (regardless of registration, as per original logic, but now we have plate info)
            lcd_display_merged(f"{plate_text[:9]} {registration_status_msg[:6]}", "Access Granted", clear_first=True)
            GPIO.output(BUZZ_PIN_ANPR, GPIO.HIGH); time.sleep(0.5); GPIO.output(BUZZ_PIN_ANPR, GPIO.LOW) # Buzzer
            open_gate_parking(servo_entry_pwm, "Entry")
            time.sleep(GATE_OPEN_DURATION_PARKING)
            close_gate_parking(servo_entry_pwm, "Entry")

            entry_time_val = time.time()
            entry_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry_time_val))
            
            log_message_suffix = "Registered" if is_registered else "Unregistered"
            with open("procedure.txt", "a") as log:
                log.write(f"[ENTRY] Plate: {plate_text} ({log_message_suffix}) | Entry Time: {entry_time_str}\n")

            if not is_registered and plate_text: # Ensure plate_text is valid before creating file
                timer_file_path = f"{plate_text}_start.txt"
                try:
                    with open(timer_file_path, "w") as f:
                        f.write(str(entry_time_val))
                    print(f"[UNREGISTERED] Timer started for {plate_text} and entry logged.")
                except IOError as e:
                    print(f"[ERROR] Could not write timer file {timer_file_path}: {e}")
            else:
                print(f"[REGISTERED] Entry logged for {plate_text}. No timer file needed or plate invalid for timer.")


            anpr_last_processed_plate = plate_text # Update last processed plate
            anpr_last_process_time = current_time # Update time of this processing
            return True # Success
        else: # OCR failed to produce text
            lcd_display_merged("Plate Found", "OCR Failed", clear_first=True)
            print("ANPR: OCR failed or plate invalid after cleaning.")
            time.sleep(1.5)
    else: # Plate contour not found
        lcd_display_merged("No Plate Found", "Try Reposition", clear_first=True)
        print("ANPR: No plate contour found.")
        time.sleep(1.5)

    return False # ANPR sequence did not lead to gate opening for a specific plate


# --- MERGED Main Loop ---
def merged_main_loop():
    global entry_gate_busy, exit_gate_busy, anpr_processing_active
    global anpr_last_processed_plate, anpr_last_plate_contour_detection_time, anpr_last_process_time
    global entry_us_last_detected, entry_us_confirmed, available_spots_count

    current_time = time.time()

    # --- ANPR Cooldown/Reset Logic ---
    if (anpr_last_processed_plate != "" and # Only if a plate is locked
        current_time - anpr_last_plate_contour_detection_time > RESET_TIMEOUT_ANPR):
        print(f"\nANPR: Resetting lock for plate '{anpr_last_processed_plate}' due to ANPR inactivity ({RESET_TIMEOUT_ANPR}s).")
        anpr_last_processed_plate = ""
        anpr_last_process_time = 0 # Reset last process time as well
        # Update LCD if no other activity is ongoing
        if not (anpr_processing_active or entry_gate_busy or exit_gate_busy):
             lcd_display_merged("System Ready", f"Spots: {available_spots_count}", clear_first=True)
    # Always update contour detection time if it's stale to avoid immediate reset on next cycle if OCR didn't run
    if anpr_last_plate_contour_detection_time == 0 : anpr_last_plate_contour_detection_time = current_time


    # --- Entry US Sensor & ANPR/Entry Sequence ---
    entry_us_distance = measure_distance(US_ENTRY_SENSOR["trig"], US_ENTRY_SENSOR["echo"])
    entry_vehicle_present_at_us = (entry_us_distance < CAR_PRESENT_THRESHOLD_CM and entry_us_distance != float('inf'))

    # Debounce logic for entry ultrasonic sensor
    if entry_vehicle_present_at_us:
        if entry_us_last_detected: # Vehicle was detected in the previous poll too
            entry_us_confirmed = True
        else: # First time vehicle is detected in this potential sequence
            entry_us_last_detected = True
            entry_us_confirmed = False # Requires a second confirmation
    else: # No vehicle detected (or sensor error)
        entry_us_last_detected = False
        entry_us_confirmed = False

    if entry_us_confirmed and not entry_gate_busy and not anpr_processing_active and not exit_gate_busy:
        if DEBUG_MODE: print(f"DEBUG: Entry US confirmed vehicle at {entry_us_distance:.1f}cm.")
        anpr_processing_active = True # Set flag before calling ANPR
        entry_gate_busy = True      # Mark entry gate as busy

        run_anpr_entry_sequence() # This function handles LCD messages during its operation

        anpr_processing_active = False # Clear flag after ANPR
        entry_gate_busy = False       # Clear busy flag
        # After ANPR, anpr_last_processed_plate might be reset by timeout logic above, or main status displayed.
        # Re-display main status if appropriate (will be handled by logic at end of loop)

    elif not entry_vehicle_present_at_us and DEBUG_MODE and entry_us_distance != float('inf'):
        # Optional: print entry US distance when no vehicle is detected for tuning
        # print(f"DEBUG: Entry US clear, dist: {entry_us_distance:.1f}cm")
        pass


    # --- Exit IR & Gate Sequence ---
    exit_ir_active = read_ir_sensor(IR_EXIT_PIN)
    if exit_ir_active and not exit_gate_busy and not entry_gate_busy and not anpr_processing_active:
        exit_gate_busy = True
        print("\nPARKING: Car detected at exit (IR).")
        lcd_display_merged("Car Exiting...", "Gate Opening", clear_first=True)
        open_gate_parking(servo_exit_pwm, "Exit")
        time.sleep(GATE_OPEN_DURATION_PARKING)
        close_gate_parking(servo_exit_pwm, "Exit")
        
        # --- MOVED AND CORRECTED EXIT LOGGING LOGIC ---
        current_exit_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        log_file_name = "procedure.txt"
        timer_file_path = f"{anpr_last_processed_plate}_start.txt" if anpr_last_processed_plate else ""

        try:
            if anpr_last_processed_plate and os.path.exists(timer_file_path):
                with open(timer_file_path, "r") as f:
                    start_time_val = float(f.read().strip())
                end_time_val = time.time()
                duration_seconds = end_time_val - start_time_val
                
                with open(log_file_name, "a") as log:
                    log.write(f"[EXIT] Plate: {anpr_last_processed_plate} | Duration: {duration_seconds:.1f} sec | Exit Time: {current_exit_time_str}\n")
                
                try: # Try to remove the timer file
                    os.remove(timer_file_path)
                    print(f"[LOGGED & CLEARED] {anpr_last_processed_plate} stayed for {duration_seconds:.1f} seconds. Timer file removed.")
                except OSError as e_remove:
                    print(f"[WARNING] Could not remove timer file {timer_file_path}: {e_remove}")

                # Reset anpr_last_processed_plate after successful exit and logging of a timed car
                # to prevent re-logging the same car if it somehow triggers exit again quickly
                # and to free up the "lock" for new entries.
                anpr_last_processed_plate = ""
                anpr_last_process_time = 0

            else: # No valid plate from ANPR for this exit, or timer file not found
                with open(log_file_name, "a") as log:
                    plate_to_log = anpr_last_processed_plate if anpr_last_processed_plate else "UNKNOWN"
                    log.write(f"[EXIT] Plate: {plate_to_log} | Exit Time: {current_exit_time_str} | Duration: N/A (timer file not found or plate unknown)\n")
                print(f"[LOGGED] Generic exit for plate '{plate_to_log}' or unknown. Duration N/A.")
                if not anpr_last_processed_plate:
                     print(f"    (No specific plate was associated with this exit)")
                elif timer_file_path:
                     print(f"    (Timer file '{timer_file_path}' was not found)")


        except Exception as e:
            print(f"[WARNING] Exit log or timer processing failed: {e}")
        # --- END OF MOVED EXIT LOGGING ---

        lcd_display_merged("Car Exited", "Thank You!", clear_first=True)
        time.sleep(1.5) # Display message a bit longer
        exit_gate_busy = False


    # --- Parking Spot Status Update ---
    update_parking_spots_status() # This will now also call display_parking_main_status_lcd if appropriate

    # --- Default LCD Display (if no other activity) ---
    if not (entry_gate_busy or exit_gate_busy or anpr_processing_active):
        display_parking_main_status_lcd()


# --- Main Execution ---
if __name__ == "__main__":
    try:
        # --- Pre-run Checks ---
        try:
            tesseract_version = pytesseract.get_tesseract_version()
            print(f"Tesseract version: {tesseract_version} found.")
        except pytesseract.TesseractNotFoundError:
            print("[FATAL ERROR] Tesseract OCR not installed or not found in PATH.")
            print("Please install Tesseract and ensure 'tesseract' command is in your system's PATH.")
            exit(1)

        if not os.path.exists('Database.txt'):
            print("[WARNING] Database.txt not found. Creating an empty one.")
            try:
                with open('Database.txt', 'w') as f: pass
                print("[INFO] Created empty Database.txt. Add registered plate numbers to it, one per line.")
            except IOError as e: print(f"[ERROR] Could not create Database.txt: {e}")
        else:
            try:
                with open('Database.txt', 'r') as f: db_lines = len(f.readlines())
                print(f"Database.txt found with {db_lines} entries.")
            except IOError as e: print(f"[ERROR] Could not read Database.txt: {e}")


        # --- Setup routines ---
        setup_gpio()
        setup_lcd()  # LCD setup includes an initial display
        setup_camera() # Sets camera_ready global
        setup_servos()

        if not camera_ready: # Display persistent warning if camera failed
            lcd_display_merged("ANPR Cam FAIL!", "Entry by US Only", clear_first=True)
            print("[CRITICAL] ANPR Camera failed to initialize. ANPR features will be disabled. Entry will rely on US sensor and availability.")
            time.sleep(2) # Allow message to be read

        # Initialize ANPR timers
        anpr_last_plate_contour_detection_time = time.time() # Initialize to current time
        anpr_last_process_time = time.time() # Initialize to prevent immediate cooldown on first real plate

        # Initial system status display
        lcd_display_merged("System Ready", f"{available_spots_count} Spots Free", clear_first=True)
        print("\nCombined Parking & ANPR System Ready. Press Ctrl+C to quit.")
        time.sleep(1) # Short delay before main loop

        # --- Main Loop ---
        while True:
            merged_main_loop()
            time.sleep(IR_POLLING_INTERVAL_MAIN_LOOP) # Main loop tick rate

    except KeyboardInterrupt:
        print("\nCtrl+C Detected. Shutting down system...")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] An unexpected error occurred in main execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up all resources...")
        if lcd_ready and lcd is not None and not isinstance(lcd, DummyLCD): # Check if it's the real LCD
           try:
             lcd_display_merged("System Offline", "Goodbye!", clear_first=True)
             time.sleep(1)
             lcd.clear()
           except: pass # Ignore errors during final cleanup
        if servo_entry_pwm: servo_entry_pwm.stop()
        if servo_exit_pwm: servo_exit_pwm.stop()
        if picam2: # Check if picam2 object exists
            try:
                if camera_ready: picam2.stop() # Only stop if it was started
                print("Camera stopped (if it was running).")
            except Exception as e: print(f"Error stopping camera: {e}")
        GPIO.cleanup()
        print("GPIO Cleaned Up. System Exited.")
