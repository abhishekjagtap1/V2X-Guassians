import cv2
import os

def create_side_by_side_video(folder_path, output_video_file, frame_rate=30):
    # List all image files in the folder
    image_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')])

    # Ensure we have at least 50 images
    assert len(image_files) == 50, "There should be exactly 50 images in the folder"

    # Get the first image to determine the size
    first_image = cv2.imread(image_files[0])
    height, width, _ = first_image.shape

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_file, fourcc, frame_rate, (width * 2, height))

    for i in range(25):
        # Read images from both cameras
        img1 = cv2.imread(image_files[i])
        img2 = cv2.imread(image_files[i + 25])

        # Concatenate images side by side
        side_by_side = cv2.hconcat([img1, img2])

        # Write the frame to the video file
        out.write(side_by_side)

    # Release the VideoWriter object
    out.release()
    print(f"Video saved as {output_video_file}")

# Example usage
folder_path = "/home/uchihadj/CVPR_2025/4DGaussians/output/multipleview/dynamic_scene_lidar/train/ours_3000/renders"
output_video_file = "side_by_side_video.mp4"
create_side_by_side_video(folder_path, output_video_file)
