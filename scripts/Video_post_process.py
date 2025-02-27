from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, clips_array

def create_combined_video(gt_path, pred_path, output_path):
    # Load the video clips
    gt_clip = VideoFileClip(gt_path)
    pred_clip = VideoFileClip(pred_path)

    # Create text clips
    gt_text = TextClip("GT", fontsize=70, color='white', bg_color='black')
    pred_text = TextClip("Novel View Synthesis", fontsize=70, color='white', bg_color='black')

    # Set the duration of the text clips to match the videos
    gt_text = gt_text.set_duration(gt_clip.duration).set_position(('center', 'top')).set_opacity(0.6)
    pred_text = pred_text.set_duration(pred_clip.duration).set_position(('center', 'top')).set_opacity(0.6)

    # Overlay the text clips on the video clips
    gt_clip = CompositeVideoClip([gt_clip, gt_text])
    pred_clip = CompositeVideoClip([pred_clip, pred_text])

    # Combine the clips vertically
    combined_clip = clips_array([[gt_clip], [pred_clip]])

    # Write the output video
    combined_clip.write_videofile(output_path, codec='libx264', fps=24)

# Paths to the input videos and the output video
gt_video_path = "/home/uchihadj/ECCV_workshop/Only_video_results/GT_Less_dynmaic_scene_novel_view.mp4"
pred_video_path = "/home/uchihadj/ECCV_workshop/Only_video_results/Less_dynmaic_scene_novel_view.mp4"
output_video_path = "/home/uchihadj/ECCV_workshop/Only_video_results/Novel_View_synthesis_using_Collaboration.mp4"

# Create the combined video
create_combined_video(gt_video_path, pred_video_path, output_video_path)
