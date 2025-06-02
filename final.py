# PYTHON SCRIPT START
import cv2
import numpy as np
import pytesseract
import RPi.GPIO as GPIO
import time
from picamera2 import Picamera2, Preview
from RPLCD.gpio import CharLCD
import re
import os
from datetime import datetime
import json
import math

# ==================================
#        CONFIGURATION & TUNING
# ==================================
DEBUG_MODE = True
DEBUG_IMG_PATH_ENTRY = "debug_images_entry"
DEBUG_IMG_PATH_EXIT = "debug_images_exit"
LOG_FILE_PATH = "parking_log.csv"
ACTIVE_PARKERS_FILE = "active_unregistered_parkers.json"

# --- Camera (ANPR) ---
IMG_WIDTH = 1024
IMG_HEIGHT = 576

# --- Plate Extraction Tuning ---
CANNY_LOW_THRESH = 50
CANNY_HIGH_THRESH = 180
CONTOUR_APPROX_FACTOR = 0.02
MIN_PLATE_AREA = 700
MIN_ASPECT_RATIO = 2.0
MAX_ASPECT_RATIO = 5.0

# --- OCR Preprocessing Tuning ---
OCR_RESIZE_HEIGHT = 60
THRESHOLD_METHOD = 'ADAPTIVE'
ADAPT_THRESH_BLOCK_SIZE = 19
ADAPT_THRESH_C = 9

# --- Tesseract Tuning ---
TESS_LANG = 'eng'
TESS_OEM = 3
TESS_PSM = '7'
TESS_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
EXPECTED_PLATE_PATTERN = ""

# --- OCR Post-Processing ---
MIN_PLATE_LENGTH = 5

# --- ANPR Main Loop Timing ---
PROCESS_COOLDOWN_ANPR = 8
RESET_TIMEOUT_ANPR = 20

# --- Parking System Configuration ---
CAR_PRESENT_THRESHOLD_CM = 25 # For entry/exit US sensors
CAR_PRESENT_THRESHOLD_SLOT_CM = 35 # Slightly higher for parking slot US sensors (can be same as above too)
CONSISTENCY_THRESHOLD_SLOT = 2 # For parking slot sensor debouncing (e.g., 2-3 readings)

SERVO_FREQUENCY = 50
SERVO_CLOSED_ANGLE = 0
SERVO_OPEN_ANGLE = 90
SERVO_MOVE_DELAY = 1.2
GATE_OPEN_DURATION_PARKING = 4.0
US_POLLING_INTERVAL = 0.75 # Poll parking slots a bit more frequently for debouncing
MAIN_LOOP_POLLING_INTERVAL = 0.1

# --- Pay-as-you-go ---
HOURLY_RATE = 5
GRACE_PERIOD_MINUTES = 10

# ==================================
#           PIN DEFINITIONS (BCM Mode) - VERIFY ALL!
# ==================================
LCD_RS_PIN = 7
LCD_E_PIN = 8
LCD_D4_PIN = 25
LCD_D5_PIN = 24
LCD_D6_PIN = 23
LCD_D7_PIN = 12

SERVO_ENTRY_PIN = 17
SERVO_EXIT_PIN = 4

BUZZ_PIN = 14 # ENSURE THIS IS A SAFE GPIO (NOT BCM 2 or 3)

US_ENTRY_SENSOR = {"name": "Entry Detect", "trig": 0, "echo": 1}
US_EXIT_SENSOR = {"name": "Exit Detect", "trig": 5, "echo": 6} # EXAMPLE - CHOOSE FREE PINS! (was 27,22 - check original choice)

# Parking Slot Sensors - EXAMPLE! ENSURE THESE ARE FREE AND NOT SPI IF SPI IS ACTIVE
US_SENSORS = [
    {"name": "Slot 1", "trig": 19, "echo": 26},
    {"name": "Slot 2", "trig": 20, "echo": 21},
    # Add more if needed
]
TOTAL_PARKING_SPOTS = len(US_SENSORS) if US_SENSORS else 0

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

# Parking Slot Sensor Debouncing States
occupied_spots_status = [False] * max(1, TOTAL_PARKING_SPOTS)
previous_occupied_spots_status = [False] * max(1, TOTAL_PARKING_SPOTS)
slot_consistent_occupied_count = [0] * max(1, TOTAL_PARKING_SPOTS)
slot_consistent_empty_count = [0] * max(1, TOTAL_PARKING_SPOTS)

available_spots_count = TOTAL_PARKING_SPOTS
last_us_poll_time = 0

entry_gate_busy = False
exit_gate_busy = False
entry_anpr_processing_active = False
exit_anpr_processing_active = False

entry_anpr_last_processed_plate = ""
entry_anpr_last_process_time = 0
entry_anpr_last_plate_contour_detection_time = 0
exit_anpr_last_processed_plate = ""
exit_anpr_last_process_time = 0
exit_anpr_last_plate_contour_detection_time = 0

active_unregistered_parkers = {}

# --- Debug Directory Creation --- (Same as before)
def create_debug_dir(path):
    if DEBUG_MODE and not os.path.exists(path):
        try: os.makedirs(path); print(f"Created debug directory: {path}")
        except OSError as e: print(f"[ERROR] Could not create debug directory '{path}': {e}")

# --- Load/Save Active Unregistered Parkers --- (Same as before)
def load_active_parkers():
    global active_unregistered_parkers
    try:
        if os.path.exists(ACTIVE_PARKERS_FILE):
            with open(ACTIVE_PARKERS_FILE, 'r') as f: active_unregistered_parkers = json.load(f)
            print(f"Loaded {len(active_unregistered_parkers)} active unregistered parkers.")
    except Exception as e: print(f"[ERROR] Could not load active parkers: {e}"); active_unregistered_parkers = {}
def save_active_parkers():
    try:
        with open(ACTIVE_PARKERS_FILE, 'w') as f: json.dump(active_unregistered_parkers, f, indent=4)
        if DEBUG_MODE: print(f"Saved {len(active_unregistered_parkers)} active parkers.")
    except Exception as e: print(f"[ERROR] Could not save active parkers: {e}")

# --- LCD Setup & Display --- (Same as before, with DummyLCD fallback)
def setup_lcd():
    global lcd, lcd_ready
    try:
        lcd = CharLCD(numbering_mode=GPIO.BCM, cols=16, rows=2, pin_rs=LCD_RS_PIN, pin_e=LCD_E_PIN, pins_data=[LCD_D4_PIN, LCD_D5_PIN, LCD_D6_PIN, LCD_D7_PIN], charmap='A00', auto_linebreaks=True)
        lcd.clear(); lcd_ready = True; print("LCD Initialized Successfully."); lcd_display_merged("System Booting", "Please Wait...")
    except Exception as e:
        print(f"[ERROR] Failed to initialize LCD: {e}"); lcd_ready = False
        class DummyLCD:
            def write_string(self, text): print(f"LCD_DUMMY: {text}")
            def clear(self): print("LCD_DUMMY: clear()")
            def cursor_pos(self, pos): print(f"LCD_DUMMY: cursor_pos({pos})")
        lcd = DummyLCD()
def lcd_display_merged(line1, line2="", clear_first=True):
    if not lcd_ready and not isinstance(lcd, DummyLCD): return
    try:
        if clear_first: lcd.clear()
        lcd.cursor_pos = (0, 0); lcd.write_string(str(line1)[:16])
        if line2: lcd.cursor_pos = (1, 0); lcd.write_string(str(line2)[:16])
    except Exception as e: print(f"[ERROR] LCD display error: {e}")

# --- Camera Setup --- (Same as before)
def setup_cameras():
    global picam2_entry, camera_entry_ready, picam2_exit, camera_exit_ready
    cam_config_main = {"size": (IMG_WIDTH, IMG_HEIGHT), "format": "RGB888"}
    try:
        print("Initializing Entry Camera (cam_num=0)...")
        picam2_entry = Picamera2(camera_num=0)
        picam2_entry.configure(picam2_entry.create_preview_configuration(main=cam_config_main))
        picam2_entry.start(); time.sleep(2.0); camera_entry_ready = True; print("Entry Camera Initialized.")
    except Exception as e: print(f"[ERROR] Failed to initialize Entry Camera: {e}"); camera_entry_ready = False
    try:
        print("Initializing Exit Camera (cam_num=1)...")
        picam2_exit = Picamera2(camera_num=1)
        picam2_exit.configure(picam2_exit.create_preview_configuration(main=cam_config_main))
        picam2_exit.start(); time.sleep(2.0); camera_exit_ready = True; print("Exit Camera Initialized.")
    except Exception as e: print(f"[ERROR] Failed to initialize Exit Camera: {e}"); camera_exit_ready = False

# --- GPIO Setup --- (Same as before)
def setup_gpio():
    GPIO.setwarnings(False); GPIO.setmode(GPIO.BCM)
    all_us_sensors = []
    if US_ENTRY_SENSOR and "trig" in US_ENTRY_SENSOR: all_us_sensors.append(US_ENTRY_SENSOR)
    if US_EXIT_SENSOR and "trig" in US_EXIT_SENSOR: all_us_sensors.append(US_EXIT_SENSOR)
    if US_SENSORS: all_us_sensors.extend(US_SENSORS)

    for us_dev_info in all_us_sensors:
        GPIO.setup(us_dev_info["trig"], GPIO.OUT); GPIO.setup(us_dev_info["echo"], GPIO.IN)
        GPIO.output(us_dev_info["trig"], False)
    GPIO.setup(SERVO_ENTRY_PIN, GPIO.OUT); GPIO.setup(SERVO_EXIT_PIN, GPIO.OUT)
    GPIO.setup(BUZZ_PIN, GPIO.OUT, initial=GPIO.LOW); print("GPIO Initialized.")

# --- Servo Control --- (Same, with SERVODBG emphasis)
def setup_servos():
    global servo_entry_pwm, servo_exit_pwm
    print("SERVODBG: Initializing Servos...")
    try:
        servo_entry_pwm = GPIO.PWM(SERVO_ENTRY_PIN, SERVO_FREQUENCY); servo_entry_pwm.start(0)
        servo_exit_pwm = GPIO.PWM(SERVO_EXIT_PIN, SERVO_FREQUENCY); servo_exit_pwm.start(0)
        print("SERVODBG: PWM started for servos.");
        set_servo_angle_parking(servo_entry_pwm, SERVO_CLOSED_ANGLE, "Entry Initial")
        set_servo_angle_parking(servo_exit_pwm, SERVO_CLOSED_ANGLE, "Exit Initial")
        print("Servos Initialized to Closed.")
    except Exception as e: print(f"[ERROR] Failed to initialize servos: {e}")
def set_servo_angle_parking(servo_pwm, angle, gate_name_debug=""):
    if servo_pwm is None: print(f"SERVODBG [WARN]: Servo PWM for {gate_name_debug} is None."); return
    duty = max(2.0, min(12.0, (angle / 18.0) + 2.0))
    print(f"SERVODBG: {gate_name_debug} - Angle: {angle} deg, Duty: {duty:.2f}%")
    servo_pwm.ChangeDutyCycle(duty); time.sleep(SERVO_MOVE_DELAY)
    servo_pwm.ChangeDutyCycle(0); print(f"SERVODBG: {gate_name_debug} - PWM signal stopped.")
def open_gate_parking(servo_pwm_obj, gate_name, associated_plate=""):
    print(f"SERVODBG: OPEN gate: {gate_name}"); line1 = associated_plate[:8].ljust(8) if associated_plate else f"{gate_name} Gate"
    lcd_display_merged(line1, "Opening...", clear_first=not((gate_name=="Entry" and entry_anpr_processing_active)or(gate_name=="Exit" and exit_anpr_processing_active)))
    set_servo_angle_parking(servo_pwm_obj, SERVO_OPEN_ANGLE, gate_name); print(f"{gate_name} gate OPENED.")
def close_gate_parking(servo_pwm_obj, gate_name):
    print(f"SERVODBG: CLOSE gate: {gate_name}"); lcd_display_merged(f"{gate_name} Gate", "Closing...")
    set_servo_angle_parking(servo_pwm_obj, SERVO_CLOSED_ANGLE, gate_name); print(f"{gate_name} gate CLOSED.")

# --- Ultrasonic Sensor Function --- (Same stable version)
def measure_distance(trig_pin, echo_pin, sensor_name="US"):
    try:
        GPIO.output(trig_pin, False); time.sleep(0.02)
        GPIO.output(trig_pin, True); time.sleep(0.00001); GPIO.output(trig_pin, False)
        t_start = time.time(); t_echo_start = t_start
        while GPIO.input(echo_pin) == 0:
            t_echo_start = time.time()
            if t_echo_start - t_start > 0.05: return float('inf') # Timeout
        t_echo_end = time.time()
        while GPIO.input(echo_pin) == 1:
            t_echo_end = time.time()
            if t_echo_end - t_echo_start > 0.05: return float('inf') # Timeout
        dist = ((t_echo_end - t_echo_start) * 34300) / 2
        return dist if dist >= 0 else float('inf')
    except RuntimeError: return float('inf')
    except Exception: return float('inf')

# --- ANPR Generic Functions --- (Same as before)
def capture_image_anpr_generic(picam_instance, camera_name="Cam"):
    if not (picam_instance and hasattr(picam_instance, 'started') and picam_instance.started):
        print(f"[ERROR-ANPR-{camera_name}] Camera not ready."); return None
    try: return cv2.cvtColor(picam_instance.capture_array("main"), cv2.COLOR_RGB2BGR)
    except Exception as e: print(f"[ERROR-ANPR-{camera_name}] Capture fail: {e}"); return None
def plate_extraction_anpr_generic(image_color, debug_path_prefix="anpr"): # Uses global TUNING constants
    # ... (full logic from previous answer, ensure it uses constants like MIN_PLATE_AREA, etc.)
    if image_color is None or image_color.size == 0: return None
    ts = int(time.time()); path = DEBUG_IMG_PATH_ENTRY if "entry" in debug_path_prefix else DEBUG_IMG_PATH_EXIT
    gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    if DEBUG_MODE: cv2.imwrite(os.path.join(path, f"{ts}_{debug_path_prefix}_00_gray.png"), gray)
    blur = cv2.GaussianBlur(gray, (5,5), 0) # Or bilateralFilter
    if DEBUG_MODE: cv2.imwrite(os.path.join(path, f"{ts}_{debug_path_prefix}_00b_blur.png"), blur)
    edges = cv2.Canny(blur, CANNY_LOW_THRESH, CANNY_HIGH_THRESH)
    if DEBUG_MODE: cv2.imwrite(os.path.join(path, f"{ts}_{debug_path_prefix}_01_edges.png"), edges)
    cnts, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    contour_found = None
    for c in cnts:
        peri = cv2.arcLength(c, True); approx = cv2.approxPolyDP(c, CONTOUR_APPROX_FACTOR * peri, True)
        if len(approx) == 4:
            (x,y,w,h) = cv2.boundingRect(approx)
            if h==0: continue
            ar = w/float(h); area = cv2.contourArea(approx)
            if MIN_PLATE_AREA < area and MIN_ASPECT_RATIO < ar < MAX_ASPECT_RATIO:
                contour_found = approx; 
                if DEBUG_MODE: 
                    img_sel = image_color.copy(); cv2.drawContours(img_sel, [approx],-1,(0,255,0),2)
                    cv2.imwrite(os.path.join(path,f"{ts}_{debug_path_prefix}_02_sel.png"),img_sel)
                break
    if contour_found is None: return None
    # Perspective transform (condensed for brevity, ensure full logic from previous answer is here)
    rect = np.zeros((4,2),dtype="float32"); s=contour_found.sum(axis=2)
    rect[0]=contour_found[np.argmin(s)]; rect[2]=contour_found[np.argmax(s)]
    diff=np.diff(contour_found,axis=2); rect[1]=contour_found[np.argmin(diff)]; rect[3]=contour_found[np.argmax(diff)]
    (tl,tr,br,bl)=rect
    wA=np.sqrt(((br[0]-bl[0])**2)+((br[1]-bl[1])**2)); wB=np.sqrt(((tr[0]-tl[0])**2)+((tr[1]-tl[1])**2)); maxW=max(int(wA),int(wB))
    hA=np.sqrt(((tr[0]-br[0])**2)+((tr[1]-br[1])**2)); hB=np.sqrt(((tl[0]-bl[0])**2)+((tl[1]-bl[1])**2)); maxH=max(int(hA),int(hB))
    if maxW<=0 or maxH<=0: return None
    dst=np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]],dtype="float32")
    M=cv2.getPerspectiveTransform(rect,dst); warped=cv2.warpPerspective(gray,M,(maxW,maxH))
    if DEBUG_MODE: cv2.imwrite(os.path.join(path, f"{ts}_{debug_path_prefix}_03_warped.png"), warped)
    return warped
def ocr_processing_anpr_generic(plate_image_gray, debug_path_prefix="anpr"): # Uses global TUNING constants
    # ... (full logic from previous answer, ensure it uses TUNING constants and sets contour detection time)
    global entry_anpr_last_plate_contour_detection_time, exit_anpr_last_plate_contour_detection_time
    if plate_image_gray is None: return ""
    path = DEBUG_IMG_PATH_ENTRY if "entry" in debug_path_prefix else DEBUG_IMG_PATH_EXIT
    t_ocr = time.time()
    if "entry" in debug_path_prefix: entry_anpr_last_plate_contour_detection_time = t_ocr
    else: exit_anpr_last_plate_contour_detection_time = t_ocr
    h,w = plate_image_gray.shape[:2]; ar=w/float(h) if h>0 else 1.0
    res_w = int(OCR_RESIZE_HEIGHT*ar)
    if res_w > 10 and OCR_RESIZE_HEIGHT > 10: resized = cv2.resize(plate_image_gray,(res_w,OCR_RESIZE_HEIGHT),interpolation=cv2.INTER_LANCZOS4)
    else: resized = plate_image_gray
    if DEBUG_MODE: cv2.imwrite(os.path.join(path,f"{int(t_ocr)}_{debug_path_prefix}_04a_res.png"),resized)
    if THRESHOLD_METHOD=='ADAPTIVE': binary = cv2.adaptiveThreshold(resized,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,ADAPT_THRESH_BLOCK_SIZE,ADAPT_THRESH_C)
    elif THRESHOLD_METHOD=='OTSU': _,binary=cv2.threshold(cv2.GaussianBlur(resized,(5,5),0),0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    else: _,binary=cv2.threshold(resized,127,255,cv2.THRESH_BINARY_INV)
    if DEBUG_MODE: cv2.imwrite(os.path.join(path,f"{int(t_ocr)}_{debug_path_prefix}_04b_bin.png"),binary)
    cfg=f'--oem {TESS_OEM} --psm {TESS_PSM} -l {TESS_LANG}' + (f' -c tessedit_char_whitelist={TESS_WHITELIST}' if TESS_WHITELIST else '')
    try:
        raw=pytesseract.image_to_string(binary,config=cfg)
        clean=''.join(filter(str.isalnum,raw)).upper()
        if DEBUG_MODE: print(f"  OCR ({debug_path_prefix}): Raw='{raw.strip()}', Clean='{clean}'")
        if len(clean) < MIN_PLATE_LENGTH: return ""
        if EXPECTED_PLATE_PATTERN and not re.fullmatch(EXPECTED_PLATE_PATTERN, clean): return ""
        # Add advanced sanity checks if needed
        return clean
    except Exception as e: print(f"[ERROR-OCR] {debug_path_prefix}: {e}"); return ""
def check_database_anpr(plate_text): # Same
    if not plate_text: return False
    try:
        with open('Database.txt', 'r') as f: db_plates = {line.strip().upper() for line in f if line.strip()}
        return plate_text in db_plates
    except FileNotFoundError: return False # Assume not in DB if file missing after initial check
    except Exception: return False

# --- Parking System Logic (WITH DEBOUNCING for Slot Sensors) ---
def update_parking_spots_status():
    global occupied_spots_status, previous_occupied_spots_status, available_spots_count
    global last_us_poll_time, slot_consistent_occupied_count, slot_consistent_empty_count

    current_time_us_poll = time.time()
    if current_time_us_poll - last_us_poll_time < US_POLLING_INTERVAL:
        return

    if not US_SENSORS: # No parking bay sensors defined
        available_spots_count = TOTAL_PARKING_SPOTS # Or some other logic if no bays
        last_us_poll_time = current_time_us_poll
        return

    confirmed_occupied_count = 0
    for i, sensor_info in enumerate(US_SENSORS):
        dist = measure_distance(sensor_info["trig"], sensor_info["echo"], sensor_info["name"])
        raw_reading_is_occupied = (dist < CAR_PRESENT_THRESHOLD_SLOT_CM)

        if raw_reading_is_occupied:
            slot_consistent_occupied_count[i] += 1
            slot_consistent_empty_count[i] = 0
        else:
            slot_consistent_empty_count[i] += 1
            slot_consistent_occupied_count[i] = 0

        # Update actual status only if consistency threshold is met
        old_status_for_slot = occupied_spots_status[i]
        new_status_for_slot = old_status_for_slot

        if slot_consistent_occupied_count[i] >= CONSISTENCY_THRESHOLD_SLOT:
            new_status_for_slot = True
        elif slot_consistent_empty_count[i] >= CONSISTENCY_THRESHOLD_SLOT:
            new_status_for_slot = False
        # Else, status remains unchanged until consistency is met

        if new_status_for_slot != old_status_for_slot:
            occupied_spots_status[i] = new_status_for_slot
            print(f"PARKING (Debounced): Slot {sensor_info['name']} is now {'OCCUPIED' if new_status_for_slot else 'EMPTY'} (Dist: {dist:.1f}cm)")
        
        if occupied_spots_status[i]: # Count based on debounced status
            confirmed_occupied_count += 1
        
        if DEBUG_MODE:
            print(f"  Slot {sensor_info['name']}: D={dist:.0f}, RawOcc={raw_reading_is_occupied}, OccCnt={slot_consistent_occupied_count[i]}, EmpCnt={slot_consistent_empty_count[i]}, Final={occupied_spots_status[i]}")
        
        # time.sleep(0.03) # Small delay between each sensor if needed for cross-talk reduction

    available_spots_count = max(0, TOTAL_PARKING_SPOTS - confirmed_occupied_count)

    if occupied_spots_status != previous_occupied_spots_status: # If any debounced status changed
        previous_occupied_spots_status = list(occupied_spots_status)

    if DEBUG_MODE: print(f"PARKING UPDATE: DebouncedOcc={confirmed_occupied_count}, Avail={available_spots_count}, LoggedUnreg={len(active_unregistered_parkers)}")
    last_us_poll_time = current_time_us_poll

def display_parking_main_status_lcd(): # Same
    global available_spots_count
    if entry_anpr_processing_active or exit_anpr_processing_active or entry_gate_busy or exit_gate_busy: return
    spots_disp = available_spots_count
    line1 = "Parking Empty" if spots_disp >= TOTAL_PARKING_SPOTS and TOTAL_PARKING_SPOTS > 0 else ("Parking Full!" if spots_disp <= 0 else "Spots Available")
    line2 = f"{spots_disp} Free" if spots_disp > 0 else ("No Spots Free" if spots_disp <=0 else "")
    lcd_display_merged(line1, line2, clear_first=True)

# --- Logging, Beeper --- (Same as before)
def setup_log_file():
    if not os.path.exists(LOG_FILE_PATH):
        try: with open(LOG_FILE_PATH, 'w') as f: f.write("Timestamp,PlateNumber,Action,Details\n"); print(f"Log file: {LOG_FILE_PATH}")
        except IOError as e: print(f"[ERROR] Log file create: {e}")
def log_parking_event(plate, action, details=""):
    try: with open(LOG_FILE_PATH, 'a') as f: f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S},{plate},{action},{details}\n")
    except IOError as e: print(f"[ERROR] Log write: {e}")
def beep(duration=0.1, count=1, delay=0.1):
    try:
        for _ in range(count): GPIO.output(BUZZ_PIN,1); time.sleep(duration); GPIO.output(BUZZ_PIN,0); time.sleep(delay if count>1 else 0)
    except RuntimeError: pass # Ignore if GPIO cleaned up

# --- ANPR Entry/Exit Sequences --- (Same overall logic as before, ensure busy flags handled)
def run_anpr_entry_sequence():
    global entry_anpr_last_processed_plate, entry_anpr_last_process_time, available_spots_count, active_unregistered_parkers, entry_gate_busy
    print("SERVODBG: ENTRY SEQUENCE START"); entry_gate_busy = True
    if not camera_entry_ready: lcd_display_merged("Entry Cam Error", "Svc Down", 1);time.sleep(2); entry_gate_busy=False; return False
    lcd_display_merged("Car at Entry", "Scanning...", 1); print("ANPR Entry: Reading plate...")
    frame = capture_image_anpr_generic(picam2_entry, "EntryCam")
    if not frame: lcd_display_merged("Entry Cam Error","Capture Fail",1);time.sleep(2);entry_gate_busy=False;return False
    plate_img = plate_extraction_anpr_generic(frame,"entry")
    if not plate_img:lcd_display_merged("No Plate Found","Reposition",1);time.sleep(2);entry_gate_busy=False;return False
    plate_txt = ocr_processing_anpr_generic(plate_img,"entry")
    if not plate_txt:lcd_display_merged("Plate Found","OCR Failed",1);time.sleep(2);entry_gate_busy=False;return False
    t_now=time.time()
    if plate_txt==entry_anpr_last_processed_plate and t_now-entry_anpr_last_process_time<PROCESS_COOLDOWN_ANPR:
        print(f"ANPR Entry: Cooldown for '{plate_txt}'."); entry_gate_busy=False; return False
    entry_anpr_last_processed_plate=plate_txt; entry_anpr_last_process_time=t_now
    print(f"ANPR Entry: '{plate_txt}'"); lcd_display_merged(f"{plate_txt[:8]} Checking","Wait...",1)
    is_reg = check_database_anpr(plate_txt)
    # Capacity check: physical spots available AND total system occupancy (logged cars) < capacity
    # Simplified: using physical spots (available_spots_count)
    can_enter = (available_spots_count > 0)
    gate_opened=False
    if is_reg:
        if can_enter: print("[REG] ✔️ Granted!");lcd_display_merged(f"{plate_txt[:8]} Granted","",1);beep(0.3);log_parking_event(plate_txt,"ENTRY_REG_OK"); open_gate_parking(servo_entry_pwm,"Entry",plate_txt); gate_opened=True
        else: print("[REG] ❌ FULL!");lcd_display_merged(f"{plate_txt[:8]} ParkingFull","",1);beep(0.1,3);log_parking_event(plate_txt,"ENTRY_REG_FULL");time.sleep(1.5)
    else: # Unregistered
        if can_enter:
            print(f"[UNREG] '{plate_txt}' Temp. Access");lcd_display_merged(f"{plate_txt[:8]} TempAccess","",1);beep(0.15,2)
            active_unregistered_parkers[plate_txt]=t_now; save_active_parkers(); log_parking_event(plate_txt,"ENTRY_UNREG_OK",f"ET:{t_now}")
            open_gate_parking(servo_entry_pwm,"Entry",plate_txt); gate_opened=True
        else: print(f"[UNREG] '{plate_txt}' ❌ FULL!");lcd_display_merged(f"{plate_txt[:8]} ParkingFull","",1);beep(0.1,3);log_parking_event(plate_txt,"ENTRY_UNREG_FULL");time.sleep(1.5)
    if gate_opened: time.sleep(GATE_OPEN_DURATION_PARKING); close_gate_parking(servo_entry_pwm,"Entry"); print("SERVODBG: ENTRY SEQ END (Gate Cycled)"); entry_gate_busy=False; return True
    print("SERVODBG: ENTRY SEQ END (No Gate Action)");entry_gate_busy=False; return False

def run_anpr_exit_sequence():
    global exit_anpr_last_processed_plate, exit_anpr_last_process_time, active_unregistered_parkers, exit_gate_busy
    print("SERVODBG: EXIT SEQUENCE START"); exit_gate_busy=True
    if not camera_exit_ready: # Fallback if no exit cam, but pay-as-you-go needs it.
        print("ANPR Exit: Exit Cam not ready. Allowing generic exit."); lcd_display_merged("Exit Cam Error","Generic Exit",1);time.sleep(2)
        log_parking_event("NO_EXIT_CAM","EXIT_GENERIC_NO_CAM"); open_gate_parking(servo_exit_pwm,"Exit"); time.sleep(GATE_OPEN_DURATION_PARKING); close_gate_parking(servo_exit_pwm,"Exit");
        exit_gate_busy=False; return True # Or False to be stricter
    lcd_display_merged("Car at Exit","Scanning...",1); print("ANPR Exit: Reading plate...")
    frame = capture_image_anpr_generic(picam2_exit, "ExitCam")
    if not frame: lcd_display_merged("Exit Cam Error","Capture Fail",1);time.sleep(2);exit_gate_busy=False;return False
    plate_img = plate_extraction_anpr_generic(frame,"exit")
    if not plate_img:lcd_display_merged("No Plate Found","Reposition",1);time.sleep(2);exit_gate_busy=False;return False
    plate_txt = ocr_processing_anpr_generic(plate_img,"exit")
    if not plate_txt:lcd_display_merged("Plate Found","OCR Failed",1);time.sleep(2);exit_gate_busy=False;return False # Critical: no exit if OCR fails
    t_now=time.time()
    if plate_txt==exit_anpr_last_processed_plate and t_now-exit_anpr_last_process_time<PROCESS_COOLDOWN_ANPR:
        print(f"ANPR Exit: Cooldown for '{plate_txt}'."); exit_gate_busy=False; return False
    exit_anpr_last_processed_plate=plate_txt; exit_anpr_last_process_time=t_now
    print(f"ANPR Exit: '{plate_txt}'"); lcd_display_merged(f"{plate_txt[:8]} Processing","Wait...",1)
    is_reg_db = check_database_anpr(plate_txt); is_unreg_parker = plate_txt in active_unregistered_parkers
    gate_opened=False
    if is_unreg_parker:
        entry_t = active_unregistered_parkers[plate_txt]; parked_secs = t_now - entry_t; parked_hrs = parked_secs / 3600.0
        billed_hrs = math.ceil(parked_hrs) if parked_hrs*60 > GRACE_PERIOD_MINUTES else 0
        if billed_hrs == 0 and parked_hrs*60 > GRACE_PERIOD_MINUTES: billed_hrs = 1 # Min 1 hr charge if over grace & not 0
        bill = billed_hrs * HOURLY_RATE
        if bill > 0:
            print(f"[UNREG] '{plate_txt}' Exit. Bill:Rs.{bill}");lcd_display_merged(f"Bill:Rs.{bill}",f"{plate_txt[:8]} Pay",1);beep(0.08,5,0.08);log_parking_event(plate_txt,"EXIT_UNREG_BILLED",f"Bill:{bill},Hrs:{billed_hrs}")
            print("SIMULATING PAYMENT (5s)..."); time.sleep(5)
        else: print(f"[UNREG] '{plate_txt}' Exit (Grace).");lcd_display_merged(f"{plate_txt[:8]} Exiting","Grace Period",1);beep(0.2,2);log_parking_event(plate_txt,"EXIT_UNREG_GRACE",f"Dur:{parked_hrs:.2f}h")
        open_gate_parking(servo_exit_pwm,"Exit",plate_txt); gate_opened=True; del active_unregistered_parkers[plate_txt]; save_active_parkers()
    elif is_reg_db:
        print(f"[REG] '{plate_txt}' Exit Granted.");lcd_display_merged(f"{plate_txt[:8]} ExitOK","",1);beep(0.3);log_parking_event(plate_txt,"EXIT_REG_OK"); open_gate_parking(servo_exit_pwm,"Exit",plate_txt); gate_opened=True
    else: print(f"[UNKNOWN] '{plate_txt}' at Exit!");lcd_display_merged(f"{plate_txt[:8]} Unknown!","",1);beep(0.1,5,0.05);log_parking_event(plate_txt,"EXIT_UNKNOWN_DENIED");time.sleep(2)
    if gate_opened: time.sleep(GATE_OPEN_DURATION_PARKING); close_gate_parking(servo_exit_pwm,"Exit");print("SERVODBG: EXIT SEQ END (Gate Cycled)"); exit_gate_busy=False; return True
    print("SERVODBG: EXIT SEQ END (No Gate Action)");exit_gate_busy=False; return False

# --- MERGED Main Logic ---
def merged_main_loop():
    global entry_gate_busy, exit_gate_busy, entry_anpr_processing_active, exit_anpr_processing_active
    global entry_anpr_last_processed_plate, entry_anpr_last_plate_contour_detection_time
    global exit_anpr_last_processed_plate, exit_anpr_last_plate_contour_detection_time

    t_loop = time.time()
    if t_loop-entry_anpr_last_plate_contour_detection_time > RESET_TIMEOUT_ANPR and entry_anpr_last_processed_plate:
        print(f"ANPR Entry: Reset lock '{entry_anpr_last_processed_plate}'."); entry_anpr_last_processed_plate=""
    if t_loop-exit_anpr_last_plate_contour_detection_time > RESET_TIMEOUT_ANPR and exit_anpr_last_processed_plate:
        print(f"ANPR Exit: Reset lock '{exit_anpr_last_processed_plate}'."); exit_anpr_last_processed_plate=""
    entry_anpr_last_plate_contour_detection_time = t_loop # Always update to prevent re-resetting immediately
    exit_anpr_last_plate_contour_detection_time = t_loop  # Same for exit

    entry_dist = measure_distance(US_ENTRY_SENSOR["trig"], US_ENTRY_SENSOR["echo"], "Entry") if US_ENTRY_SENSOR else float('inf')
    entry_detected = (entry_dist < CAR_PRESENT_THRESHOLD_CM)
    exit_dist = measure_distance(US_EXIT_SENSOR["trig"], US_EXIT_SENSOR["echo"], "Exit") if US_EXIT_SENSOR else float('inf')
    exit_detected = (exit_dist < CAR_PRESENT_THRESHOLD_CM)
    if DEBUG_MODE and (entry_dist!=float('inf') or exit_dist!=float('inf')):print(f"Sensors: E:{entry_dist:.0f} D:{entry_detected}, X:{exit_dist:.0f} D:{exit_detected} SPOTS_US:{available_spots_count}   ",end='\r')

    # Only process new events if no major gate/ANPR operation is ongoing
    if not (entry_gate_busy or entry_anpr_processing_active or exit_gate_busy or exit_anpr_processing_active):
        if entry_detected:
            print(f"\nMAIN: Entry detected ({entry_dist:.0f}cm).")
            entry_anpr_processing_active = True # Set flag before calling
            run_anpr_entry_sequence()
            entry_anpr_processing_active = False # Clear flag after
        elif exit_detected: # Process only if entry was not detected this tick
            print(f"\nMAIN: Exit detected ({exit_dist:.0f}cm).")
            exit_anpr_processing_active = True
            run_anpr_exit_sequence()
            exit_anpr_processing_active = False
    
    update_parking_spots_status() # Poll parking bay sensors
    if not (entry_gate_busy or exit_gate_busy or entry_anpr_processing_active or exit_anpr_processing_active):
        display_parking_main_status_lcd()

# --- Main Execution --- (Error handling and setup largely same)
if __name__ == "__main__":
    try:
        print("System Starting..."); create_debug_dir(DEBUG_IMG_PATH_ENTRY); create_debug_dir(DEBUG_IMG_PATH_EXIT)
        try: print(f"Tesseract: {pytesseract.get_tesseract_version()}");
        except Exception as e: print(f"[FATAL] Tesseract not found/error: {e}. `sudo apt install tesseract-ocr`"); exit(1)
        if not os.path.exists('Database.txt'): print("[WARN] Database.txt missing, creating."); open('Database.txt','w').close()
        setup_log_file(); load_active_parkers(); setup_gpio(); setup_lcd(); setup_cameras(); setup_servos()

        if not (camera_entry_ready or camera_exit_ready): print("[CRIT] Camera(s) failed.") # Check specific readiness if needed

        t_main_init = time.time()
        entry_anpr_last_plate_contour_detection_time = t_main_init; entry_anpr_last_process_time = t_main_init
        exit_anpr_last_plate_contour_detection_time = t_main_init; exit_anpr_last_process_time = t_main_init
        last_us_poll_time = 0 # Force first poll immediately in update_parking_spots_status
        
        update_parking_spots_status() # Initial status
        lcd_display_merged("System Ready", f"{available_spots_count} Spots Free", 1)
        print("\nSmart Parking System Ready. Tuning params printed if DEBUG_MODE.")
        if DEBUG_MODE: 
            print(f"ANPR PARAMS: CannyL/H:{CANNY_LOW_THRESH}/{CANNY_HIGH_THRESH}, Area:{MIN_PLATE_AREA}, AR:{MIN_ASPECT_RATIO}-{MAX_ASPECT_RATIO}")
            print(f"OCR PARAMS: ResizeH:{OCR_RESIZE_HEIGHT},Thresh:{THRESHOLD_METHOD},Block/C:{ADAPT_THRESH_BLOCK_SIZE}/{ADAPT_THRESH_C},PSM:{TESS_PSM}")
        print("--- System Live ---")

        while True:
            merged_main_loop()
            time.sleep(MAIN_LOOP_POLLING_INTERVAL)
    except KeyboardInterrupt: print("\nCtrl+C. Shutting down...")
    except Exception as e: print(f"\n[FATAL MAIN ERROR]: {e}"); traceback.print_exc()
    finally:
        print("Cleaning up..."); save_active_parkers()
        for cam in [picam2_entry, picam2_exit]:
            if cam and hasattr(cam,'started') and cam.started:
                try: cam.stop_preview(); cam.stop(); print(f"Cam {cam.camera_num if hasattr(cam,'camera_num') else '?'} stopped.")
                except: pass
        if lcd_ready and lcd and not isinstance(lcd,DummyLCD) and hasattr(lcd,'clear'):
            try: lcd_display_merged("System Offline","Goodbye!",1);time.sleep(1);lcd.clear()
            except: pass
        for pwm in [servo_entry_pwm,servo_exit_pwm]:
            if pwm: try: pwm.stop(); print("Servo PWM stopped.")
            except: pass
        GPIO.cleanup(); print("GPIO Cleaned. Exit.")
# PYTHON SCRIPT END
