import cv2
import os

def create_side_by_side_video(folder_path, output_video_file, frame_rate=30):
    # List all image files in the folder
    image_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')])

    # Ensure we have at least 50 images
    assert len(image_files) == 40, "There should be exactly 40 images in the folder"

    # Get the first image to determine the size
    first_image = cv2.imread(image_files[0])
    height, width, _ = first_image.shape

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')#
    out = cv2.VideoWriter(output_video_file, fourcc, frame_rate, (width * 2, height))

    for i in range(20):
        # Read images from both cameras
        img1 = cv2.imread(image_files[i])
        img2 = cv2.imread(image_files[i + 20])

        # Concatenate images side by side
        side_by_side = cv2.hconcat([img1, img2])

        # Write the frame to the video file
        out.write(side_by_side)

    # Release the VideoWriter object
    out.release()
    print(f"Video saved as {output_video_file}")

# Example usage
"""

#folder_path = "/home/uchihadj/ECCV_workshop/4DGaussians/output/ECCV_2025/novel_view_rendering/Novel_view_random_rotation_translation/good_left_right_0.2/ours_5000/renders" #Left__right_novel_view_rendering
folder_path = "/home/uchihadj/ECCV_workshop/4DGaussians/output/ECCV_2025/novel_view_rendering/Novel_view_random_rotation_translation/novel_views_up_down_0.1/ours_5000/renders" #up_down
#folder_path ="/home/uchihadj/ECCV_workshop/4DGaussians/output/ECCV_2025/novel_view_rendering/Novel_view_random_rotation_translation/good_left_right_0.2/ours_5000/gt" #ground_truth
output_video_file = "/home/uchihadj/ECCV_workshop/4DGaussians/output/Render_results_dynamic/novel_views_up_down_0/side_by_side_video.mp4"
"""
#folder_path = "/home/uchihadj/ECCV_workshop/4DGaussians/output/ECCV_2025/Second_part_trained_and_rendered_novel_views/train/ours_14000/renders" #Left__right_novel_view_rendering
folder_path = "/home/uchihadj/ECCV_workshop/4DGaussians/output/ECCV_2025/First_part_trained_rendered_novel_views/train/ours_10000/renders" #Pakka lo
#folder_path ="/home/uchihadj/ECCV_workshop/4DGaussians/output/High_rendering/renders" #ground_truth
output_video_file = "/home/uchihadj/ECCV_workshop/4DGaussians/output/ECCV_2025/First_part_trained_rendered_novel_views/train/ours_10000/side_by_side_zoom_video.mp4"
#output_video_file = "/home/uchihadj/ECCV_workshop/4DGaussians/output/ECCV_2025/Second_part_trained_and_rendered_novel_views/train/ours_14000/side_by_side_zoom_video.mp4"

#folder_path = "/home/uchihadj/ECCV_workshop/4DGaussians/output/ECCV_2025/Second_part_trained_and_rendered_novel_views/train/ours_14000/renders" #Left__right_novel_view_rendering
folder_path = "/home/uchihadj/ECCV_workshop/4DGaussians/output/ECCV_2025/dynamic_new_scene_please_work/train/ours_7000/gt" #Pakka lo
#folder_path ="/home/uchihadj/ECCV_workshop/4DGaussians/output/High_rendering/renders" #ground_truth
output_video_file = "/home/uchihadj/ECCV_workshop/4DGaussians/output/ECCV_2025/dynamic_new_scene_please_work/train/ours_7000/gt/side_by_side_zoom_video.mp4"
#output_video_file = "/home/uchihadj/ECCV_workshop/4DGaussians/output/ECCV_2025/Second_part_trained_and_rendered_novel_views/train/ours_14000/side_by_side_zoom_video.mp4"
create_side_by_side_video(folder_path, output_video_file)
