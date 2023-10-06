import os
import cv2

# Define the path to the collected frames
input_directory = 'collected_frames'

# Create an output directory for preprocessed frames
output_directory = 'preprocessed_frames'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Loop through the collected frames
for filename in os.listdir(input_directory):
    if filename.endswith('.jpg'):
        frame_path = os.path.join(input_directory, filename)

        # Load the frame
        frame = cv2.imread(frame_path)

        # Perform preprocessing (e.g., resizing, normalization, etc.)
        # You can add your preprocessing steps here

        # Save the preprocessed frame
        output_path = os.path.join(output_directory, filename)
        cv2.imwrite(output_path, frame)

print("Data preprocessing completed.")
