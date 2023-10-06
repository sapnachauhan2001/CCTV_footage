import cv2
import os

# Define the path to save the collected frames
output_directory = 'collected_frames'

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Path to the CCTV footage video file
video_path = 'cctv_footage.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Initialize variables
frame_count = 0

# Loop through the video frames
while True:
    ret, frame = cap.read()

    # Break the loop if we have reached the end of the video
    if not ret:
        break

    # Save the frame as an image
    frame_filename = os.path.join(output_directory, f'frame_{frame_count:04d}.jpg')
    cv2.imwrite(frame_filename, frame)

    frame_count += 1

# Release the video capture object
cap.release()

print(f"Collected {frame_count} frames.")
