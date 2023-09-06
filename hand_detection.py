# Import the OpenCV library.
import cv2

# Import the MediaPipe library.
import mediapipe as mp

# Open the camera. 0 represents the default camera device of the computer.
cap = cv2.VideoCapture(0)

# Create an object to use the MediaPipe Hands solution.
mphands = mp.solutions.hands

# Import the drawing utility functions from MediaPipe.
draw = mp.solutions.drawing_utils

# Initialize the Hands object.
hand = mphands.Hands()

# Start an infinite loop to continuously capture images from the camera.
while True:

    # Capture an image from the camera and return whether it was successful (success) and the image itself (img).
    success, img = cap.read()

    # Convert the image to the RGB color space (MediaPipe works with RGB images).
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image using the MediaPipe Hands for hand detection.
    result = hand.process(imgRGB)

    # If at least one hand is detected:
    if result.multi_hand_landmarks:

        # For each detected hand:
        for handlms in result.multi_hand_landmarks:

            # Draw landmarks and connections on the image.
            draw.draw_landmarks(img, handlms, mphands.HAND_CONNECTIONS)

    # Display the image in a window.
    cv2.imshow("img", img)

    # Wait for the user to press the 'q' key to exit the loop.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera (stop using the camera).
cap.release()

# Close all OpenCV windows.
cv2.destroyAllWindows()
