import cv2
import numpy as np
import mediapipe as mp
import tflite_runtime.interpreter as tflite
from collections import deque
import time
import sys

# Parameters
frame_rate = 30  # Frames per second
buffer_time = 5  # Seconds for the initial clip
buffer_size = frame_rate * buffer_time
confidence_threshold = 0.5  # 50% confidence threshold
distraction_time_threshold = 3  # Seconds to start recording after distraction
distraction_frame_threshold = frame_rate * distraction_time_threshold
post_distraction_time_threshold = 5  # Seconds to stop recording after no distraction
post_distraction_frame_threshold = frame_rate * post_distraction_time_threshold
max_recording_time = 30  # Max recording time in seconds
max_recording_frames = frame_rate * max_recording_time
consecutive_ticks_required = 4  # Number of consecutive ticks required to confirm label
consecutive_check_ticks = 3  # Number of consecutive ticks required to change status
no_hands_ticks_threshold = 3  # Number of consecutive ticks with no hands detected to stop recording

# Paths to model and labels
model_path = 'Model/model.tflite'
labels_path = 'Model/labels.txt'

# Initialize buffers and counters
frame_buffer = deque(maxlen=buffer_size)
recording_buffer = deque()
distraction_counter = 0
non_distraction_counter = 0
is_recording = False
has_saved_pre_distraction_clip = False
recording_start_time = None

# Track consecutive ticks
tick_buffer = deque(maxlen=consecutive_ticks_required)  # Track the last few ticks
consecutive_distracted_ticks = 0
consecutive_focused_ticks = 0
no_hands_counter = 0  # Counter for consecutive ticks with no hands detected

# Load the TFLite model
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the labels
with open(labels_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# Initialize video capture
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change codec for MP4 format

def save_buffer(buffer, file_name):
    if not buffer:
        print(f"No frames to save for {file_name}")
        return
    
    height, width, _ = buffer[0].shape
    out = cv2.VideoWriter(file_name, fourcc, frame_rate, (width, height))
    for frame in buffer:
        out.write(frame)
    out.release()
    print(f"Saved video: {file_name}")

last_tick_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    # Check if it's time for the next tick
    if current_time - last_tick_time >= 1:
        last_tick_time = current_time

        # Add the frame to the buffer
        frame_buffer.append(frame)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with Mediapipe
        result = hands.process(rgb_frame)

        hands_detected = False
        hand_info = {}

        # Check if hands are detected
        if result.multi_hand_landmarks:
            hands_detected = True
            no_hands_counter = 0  # Reset no hands counter
            
            for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                # Get bounding box for the hand
                x_min, x_max = min([landmark.x for landmark in hand_landmarks.landmark]), max([landmark.x for landmark in hand_landmarks.landmark])
                y_min, y_max = min([landmark.y for landmark in hand_landmarks.landmark]), max([landmark.y for landmark in hand_landmarks.landmark])
                
                # Convert normalized coordinates to pixel values
                height, width, _ = frame.shape
                x_min, x_max = int(x_min * width), int(x_max * width)
                y_min, y_max = int(y_min * height), int(y_max * height)
                
                # Crop the frame to the hand region
                hand_roi = frame[y_min:y_max, x_min:x_max]

                if hand_roi.size > 0:
                    # Convert the image to grayscale
                    hand_roi = cv2.cvtColor(hand_roi, cv2.COLOR_RGB2GRAY)

                    # Resize the image to match model input shape
                    expected_height = input_details[0]['shape'][1]
                    expected_width = input_details[0]['shape'][2]
                    input_data = cv2.resize(hand_roi, (expected_width, expected_height))
                    
                    # Expand dimensions to match the input shape (batch size, height, width, channels)
                    input_data = np.expand_dims(input_data, axis=0)
                    input_data = np.expand_dims(input_data, axis=-1)  # Add channel dimension

                    # Normalize the input data
                    input_data = input_data / 255.0

                    # Convert to FLOAT32
                    input_data = input_data.astype(np.float32)

                    # Run inference
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(output_details[0]['index'])

                    # Determine if the hand is distracted
                    distraction_confidence = output_data[0][0]  # Confidence for the "distracted" class (index 0)
                    focused_confidence = output_data[0][1]  # Confidence for the "not distracted" class (index 1)

                    hand_info[f'Hand {idx + 1}'] = {
                        'distracted_confidence': distraction_confidence,
                        'focused_confidence': focused_confidence
                    }

                    # Draw the bounding box
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Determine final status based on hand info
        overall_distracted = False
        output_lines = []

        if hands_detected:
            for hand, info in hand_info.items():
                distracted_percentage = info['distracted_confidence'] * 100
                focused_percentage = info['focused_confidence'] * 100

                # Append individual hand results
                output_lines.append(f"{hand} Distracted: {distracted_percentage:.2f}% Focused: {focused_percentage:.2f}%")

                # Collect tick for each hand
                hand_status = 'distracted' if distracted_percentage > confidence_threshold else 'focused'
                tick_buffer.append(hand_status)

            # Check if recording should start
            if len(tick_buffer) == consecutive_ticks_required:
                if tick_buffer.count('distracted') >= consecutive_ticks_required and tick_buffer.count('focused') <= 1:
                    if not is_recording:
                        is_recording = True
                        recording_start_time = time.time()
                        has_saved_pre_distraction_clip = False
                        print("Started recording.")

            # Check if recording should stop
            if len(tick_buffer) == consecutive_ticks_required:
                if tick_buffer.count('focused') >= 3 and is_recording:
                    print("Stopping recording due to focus.")
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    if not has_saved_pre_distraction_clip:
                        save_buffer(list(frame_buffer), f'pre_distraction_{timestamp}.mp4')
                        has_saved_pre_distraction_clip = True
                    save_buffer(list(recording_buffer), f'distraction_{timestamp}.mp4')
                    frame_buffer.clear()
                    recording_buffer.clear()
                    is_recording = False
                    distraction_counter = 0
                    non_distraction_counter = 0

            # Count consecutive ticks for distraction or focus
            if tick_buffer:
                current_status = tick_buffer[-1]
                if current_status == 'distracted':
                    consecutive_distracted_ticks += 1
                    consecutive_focused_ticks = 0
                else:
                    consecutive_focused_ticks += 1
                    consecutive_distracted_ticks = 0

                # Check if the status should change
                if consecutive_distracted_ticks >= consecutive_check_ticks:
                    overall_distracted = True
                elif consecutive_focused_ticks >= consecutive_check_ticks:
                    overall_distracted = False

                # Reset counters if needed
                if len(tick_buffer) >= consecutive_check_ticks:
                    if overall_distracted and consecutive_focused_ticks >= consecutive_check_ticks:
                        overall_distracted = False
                    elif not overall_distracted and consecutive_distracted_ticks >= consecutive_check_ticks:
                        overall_distracted = True
        else:
            # Increment no hands counter if no hands are detected
            no_hands_counter += 1

            # Check if recording should stop due to no hands detected
            if no_hands_counter >= no_hands_ticks_threshold and is_recording:
                print("Stopping recording due to no hands detected.")
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                if not has_saved_pre_distraction_clip:
                    save_buffer(list(frame_buffer), f'pre_distraction_{timestamp}.mp4')
                    has_saved_pre_distraction_clip = True
                save_buffer(list(recording_buffer), f'distraction_{timestamp}.mp4')
                frame_buffer.clear()
                recording_buffer.clear()
                is_recording = False
                distraction_counter = 0
                non_distraction_counter = 0
                no_hands_counter = 0  # Reset the no hands counter

        # Mark overall distraction status
        overall_status = 'distracted' if overall_distracted else 'not distracted'

        # Set recording status
        recording_status = 'Yes' if is_recording else 'No'

        # Print results
        print(f"Overall Status: {overall_status}")
        print(f"Recording: {recording_status}")
        for line in output_lines:
            print(line)

        # Show the frame
        cv2.imshow('Hand Tracking', frame)

        # Save frames if recording
        if is_recording:
            recording_buffer.append(frame)

    # Break the loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
