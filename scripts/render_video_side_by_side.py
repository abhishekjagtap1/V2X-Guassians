import cv2
import os

def create_side_by_side_video(folder_path, output_video_file, frame_rate=10):
    # List all image files in the folder
    image_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')])
    half_index = int(len(image_files) / 2)

    # Ensure we have at least 50 images
    assert len(image_files) == 156, "There should be exactly 50 images in the folder"

    # Get the first image to determine the size
    first_image = cv2.imread(image_files[0])
    height, width, _ = first_image.shape

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_file, fourcc, frame_rate, (width * 2, height))

    for i in range(half_index):
        # Read images from both cameras
        img1 = cv2.imread(image_files[i])
        img2 = cv2.imread(image_files[i + half_index])

        # Concatenate images side by side
        side_by_side = cv2.hconcat([img1, img2])

        # Write the frame to the video file
        out.write(side_by_side)

    # Release the VideoWriter object
    out.release()
    print(f"Video saved as {output_video_file}")

# Example usage
folder_path = "/home/uchihadj/ECCV_workshop/4DGaussians/output/ECCV_2025/less_dynamic_scene/train/ours_14000/renders"
output_video_file = "/home/uchihadj/ECCV_workshop/Only_video_results/Reconstructude_video.mp4"
create_side_by_side_video(folder_path, output_video_file)
"""
import cv2
import os

def create_video_from_frames(folder_path, output_path, fps=10):
    # Get list of all files in the folder and sort them
    frames = sorted([os.path.join(folder_path, frame) for frame in os.listdir(folder_path) if frame.endswith(('.png', '.jpg', '.jpeg'))])

    # Check if there are any frames
    if not frames:
        raise ValueError("No frames found in the specified folder.")

    # Read the first frame to get the frame size
    frame = cv2.imread(frames[0])
    height, width, layers = frame.shape
    size = (width, height)

    # Initialize the video writer
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for frame_path in frames:
        frame = cv2.imread(frame_path)
        out.write(frame)

    # Release the video writer
    out.release()

# Folder containing the frames
frames_folder = "/home/uchihadj/ECCV_workshop/4DGaussians/output/ECCV_2025/less_dynamic_scene/train/ours_14000/only_novel_view"
# Output video file path
output_video = "/home/uchihadj/ECCV_workshop/Only_video_results/Novel_view_showing_collboration.mp4"

# Create the video from frames
create_video_from_frames(frames_folder, output_video)
"""
