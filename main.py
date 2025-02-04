import cv2
import mediapipe as mp
import numpy as np
from Quartz import CGEventPost, kCGHIDEventTap, CGEventCreateMouseEvent, CGEventCreateScrollWheelEvent, CGEventCreateKeyboardEvent
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Screen dimensions (adjust based on your display)
screen_width, screen_height = 1920, 1080

# Variables for smoothing finger tracking
prev_finger_x, prev_finger_y = 0, 0
smooth_factor = 0.5

# Thresholds for gestures and click detection
CLICK_THRESHOLD = 0.03       # Distance threshold for a click
SWIPE_THRESHOLD = 0.4        # Distance threshold for hand wave gestures
PINCH_THRESHOLD = 0.05       # Distance threshold for pinch gesture
SWIPE_DURATION = 0.5         # Time duration for swipe gestures
WAVE_START_TIME = None       # Start time for hand wave gesture

# Function to move the mouse cursor
def move_mouse(x, y):
    """
    Moves the mouse cursor to the specified screen coordinates.
    """
    event = CGEventCreateMouseEvent(None, 5, (x, y), 0)  # 5 corresponds to kCGEventMouseMoved
    CGEventPost(kCGHIDEventTap, event)

# Function to simulate mouse click
def mouse_click():
    """
    Simulates a left mouse button click.
    """
    event_down = CGEventCreateMouseEvent(None, 1, (0, 0), 0)  # 1 corresponds to kCGEventLeftMouseDown
    CGEventPost(kCGHIDEventTap, event_down)
    event_up = CGEventCreateMouseEvent(None, 2, (0, 0), 0)  # 2 corresponds to kCGEventLeftMouseUp
    CGEventPost(kCGHIDEventTap, event_up)

# Function to simulate right mouse click
def mouse_right_click():
    """
    Simulates a right mouse button click.
    """
    event_down = CGEventCreateMouseEvent(None, 1, (0, 0), 2)  # Right mouse button down
    CGEventPost(kCGHIDEventTap, event_down)
    event_up = CGEventCreateMouseEvent(None, 2, (0, 0), 2)  # Right mouse button up
    CGEventPost(kCGHIDEventTap, event_up)

# Function to simulate scrolling
def scroll(direction):
    """
    Simulates scrolling up or down.
    :param direction: 'up' or 'down' to scroll in that direction.
    """
    scroll_event = CGEventCreateScrollWheelEvent(None, 0, 1, 1 if direction == "up" else -1)
    CGEventPost(kCGHIDEventTap, scroll_event)

# Function to switch desktops using keyboard shortcuts
def switch_desktop(direction):
    """
    Switches the desktop using a keyboard shortcut.
    :param direction: 'left' or 'right' to switch desktops.
    """
    key_code_left = 123  # Left arrow key
    key_code_right = 124  # Right arrow key
    key_code_control = 59  # Control key

    # Press and hold the Control key
    control_down = CGEventCreateKeyboardEvent(None, key_code_control, True)
    CGEventPost(kCGHIDEventTap, control_down)

    # Press the arrow key
    if direction == "left":
        key_event = CGEventCreateKeyboardEvent(None, key_code_left, True)
    elif direction == "right":
        key_event = CGEventCreateKeyboardEvent(None, key_code_right, True)
    CGEventPost(kCGHIDEventTap, key_event)

    # Release the keys
    CGEventCreateKeyboardEvent(None, key_code_left if direction == "left" else key_code_right, False)
    CGEventCreateKeyboardEvent(None, key_code_control, False)

# Main loop for finger tracking and gesture detection
cap = cv2.VideoCapture(0)  # Use default webcam
prev_wrist_position = None
wave_start_time = None
wave_direction = None

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip the image horizontally for a mirrored view
    image = cv2.flip(image, 1)

    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image for hand detection
    results = hands.process(rgb_image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the wrist position (a stable reference point for hand wave gestures)
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

            # Map the wrist position to screen coordinates
            wrist_x = int(wrist.x * screen_width)
            wrist_y = int(wrist.y * screen_height)

            # Track the index finger for mouse movement
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Calculate the distance between the index finger and thumb
            click_distance = np.sqrt(
                (index_finger_tip.x - thumb_tip.x) ** 2 +
                (index_finger_tip.y - thumb_tip.y) ** 2
            )

            # Map the index finger position to screen coordinates
            finger_x = int(index_finger_tip.x * screen_width)
            finger_y = int(index_finger_tip.y * screen_height)

            # Smooth the finger movement
            finger_x = prev_finger_x + (finger_x - prev_finger_x) * smooth_factor
            finger_y = prev_finger_y + (finger_y - prev_finger_y) * smooth_factor
            prev_finger_x, prev_finger_y = finger_x, finger_y

            # Move the mouse cursor
            move_mouse(finger_x, finger_y)

            # Simulate a mouse click if the distance is below the threshold
            if click_distance < CLICK_THRESHOLD:
                mouse_click()

            # Gesture detection: Pinch gesture for right-click
            if click_distance < PINCH_THRESHOLD:
                mouse_right_click()

            # Gesture detection: Hand wave for desktop switching
            if prev_wrist_position is not None:
                current_wrist_position = np.array([wrist.x, wrist.y])
                previous_wrist_position = np.array(prev_wrist_position)

                # Calculate the displacement vector
                displacement = current_wrist_position - previous_wrist_position
                displacement_magnitude = np.linalg.norm(displacement)

                # Check if the displacement exceeds the wave threshold
                if abs(displacement[0]) > SWIPE_THRESHOLD:
                    if wave_start_time is None:
                        wave_start_time = time.time()  # Start timing the wave
                        wave_direction = "left" if displacement[0] < 0 else "right"
                    elif time.time() - wave_start_time < SWIPE_DURATION:
                        # If the wave continues in the same direction, switch desktops
                        if (wave_direction == "left" and displacement[0] < 0) or \
                           (wave_direction == "right" and displacement[0] > 0):
                            switch_desktop(wave_direction)
                            wave_start_time = None  # Reset the timer
                            wave_direction = None
                else:
                    wave_start_time = None  # Reset the timer if no significant movement

            # Update the previous wrist position
            prev_wrist_position = [wrist.x, wrist.y]

            # Draw landmarks on the image
            mp.solutions.drawing_utils.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    # Display the image with landmarks
    cv2.imshow('Finger-Only Mouse Controller', image)

    # Exit on 'q' key press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()