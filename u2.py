import cv2
import mediapipe as mp

# Initialize OpenCV VideoCapture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Load custom component icons with alpha channel
resistor_icon = cv2.imread("resistor.png", cv2.IMREAD_UNCHANGED)
voltage_source_icon = cv2.imread("battery.png", cv2.IMREAD_UNCHANGED)

# Check if icons are loaded successfully
if resistor_icon is None or voltage_source_icon is None:
    print("Error: Failed to load custom component icons.")
    exit()

# Resize the icons to match the ROI on the frame
icon_size = (100, 100)
resistor_icon = cv2.resize(resistor_icon, icon_size)
voltage_source_icon = cv2.resize(voltage_source_icon, icon_size)

# Convert icons to RGB format
resistor_icon_rgb = cv2.cvtColor(resistor_icon, cv2.COLOR_BGRA2RGB)
voltage_source_icon_rgb = cv2.cvtColor(voltage_source_icon, cv2.COLOR_BGRA2RGB)

# Initial positions of icons on the right side of the panel
resistor_position = (500, 50)
voltage_source_position = (500, 150)

# Initialize variables for dragging functionality
dragging_resistor = False
dragging_voltage_source = False

# Create the dot positions for the panel
dot_positions = [(250, 300), (350,300), (450, 300), (550, 300),
                 (250, 400), (350, 400), (450, 400), (550, 400)]
dot_radius = 10

# Initialize resistance value
resistance_value = 0

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame")
            break

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Display resistor and voltage source icons at their respective positions
        frame[resistor_position[1]:resistor_position[1] + icon_size[1],
              resistor_position[0]:resistor_position[0] + icon_size[0]] = resistor_icon_rgb

        frame[voltage_source_position[1]:voltage_source_position[1] + icon_size[1],
              voltage_source_position[0]:voltage_source_position[0] + icon_size[0]] = voltage_source_icon_rgb

        # Detect hand landmarks
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Check if hand is over the resistor icon
                if (resistor_position[0] < hand_landmarks.landmark[8].x * frame.shape[1] < resistor_position[0] + icon_size[0] and
                        resistor_position[1] < hand_landmarks.landmark[8].y * frame.shape[0] < resistor_position[1] + icon_size[1]):
                    dragging_resistor = True
                    resistor_position = (int(hand_landmarks.landmark[8].x * frame.shape[1] - icon_size[0] / 2),
                                         int(hand_landmarks.landmark[8].y * frame.shape[0] - icon_size[1] / 2))
                else:
                    dragging_resistor = False

                # Check if hand is over the voltage source icon
                if (voltage_source_position[0] < hand_landmarks.landmark[8].x * frame.shape[1] < voltage_source_position[0] + icon_size[0] and
                        voltage_source_position[1] < hand_landmarks.landmark[8].y * frame.shape[0] < voltage_source_position[1] + icon_size[1]):
                    dragging_voltage_source = True
                    voltage_source_position = (int(hand_landmarks.landmark[8].x * frame.shape[1] - icon_size[0] / 2),
                                               int(hand_landmarks.landmark[8].y * frame.shape[0] - icon_size[1] / 2))
                else:
                    dragging_voltage_source = False

        # Update resistance value based on the resistor icon position
        if dragging_resistor:
            # Change the resistance value based on the vertical position of the resistor icon
            resistance_value = int((resistor_position[1] - 50) / 50 * 1000)  # Adjust the scaling factor as needed

        # Display the dot panel
        for dot_position in dot_positions:
            cv2.circle(frame, dot_position, dot_radius, (0, 255, 0), -1)

        # Display resistance value
        cv2.putText(frame, f"Resistance: {resistance_value} Ohms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Hand Tracking with Resistance Value', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Release resources
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
