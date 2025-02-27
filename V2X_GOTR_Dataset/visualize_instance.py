import json
import os
from PIL import Image
import matplotlib.pyplot as plt


def extract_and_visualize_images(json_file, image_dir, output_dir):
    """
    Extract, stitch, and visualize images for a given JSON file.

    Args:
        json_file (str): Path to the JSON file.
        image_dir (str): Directory containing the images.
        output_dir (str): Directory to save visualizations.
    """
    # Load the JSON file
    with open(json_file, "r") as file:
        data = json.load(file)

    # Check if "frames" exists
    if "frames" not in data["openlabel"] or not data["openlabel"]["frames"]:
        print(f"Warning: No frames found in {json_file}. Skipping.")
        return

    # Extract the first key from frames dynamically
    frame_key = next(iter(data["openlabel"]["frames"]))  # Get the first frame key dynamically
    if "frame_properties" not in data["openlabel"]["frames"][frame_key]:
        print(f"Warning: No frame properties found in {json_file}. Skipping.")
        return

    frame = data["openlabel"]["frames"][frame_key]["frame_properties"]
    image_file_names = frame["image_file_names"]

    # Set variables cam1, cam2, and cam3 based on conditions
    cam1 = next((img for img in image_file_names if img.endswith("south2_8mm.jpg")), None)
    cam2 = next((img for img in image_file_names if "vehicle_camera" in img), None)
    cam3 = next((img for img in image_file_names if img.endswith("south1_8mm.jpg")), None)

    if not cam1 or not cam2 or not cam3:
        print(f"Warning: Skipping {json_file}. Missing required images.")
        return

    print(f"Processing {json_file}: cam1={cam1}, cam2={cam2}, cam3={cam3}")

    # Load images
    image_paths = {
        "cam1": os.path.join(image_dir, 'cam1', cam1),
        "cam2": os.path.join(image_dir, 'cam2', cam2),
        "cam3": os.path.join(image_dir, 'cam3', cam3),
    }

    loaded_images = {}
    for cam, path in image_paths.items():
        if os.path.exists(path):
            loaded_images[cam] = Image.open(path)
        else:
            print(f"Error: Image file not found - {path}")
            return

    # Stitch images horizontally
    stitched_image = stitch_images_horizontally([loaded_images["cam1"], loaded_images["cam2"], loaded_images["cam3"]])
    stitched_image.thumbnail((1500, 500))
    # Save the stitched image
    json_filename = os.path.basename(json_file).replace(".json", "")
    stitched_output_path = os.path.join(output_dir, f"{json_filename}_stitched.png")
    stitched_image.save(stitched_output_path)
    print(f"Saved stitched image: {stitched_output_path}")

    # # Visualize stitched image
    # plt.figure(figsize=(10, 5))
    # plt.imshow(stitched_image)
    # plt.title(f"Stitched Visualization for {json_filename}")
    # plt.axis("off")

    # # Save visualization
    # visualization_path = os.path.join(output_dir, f"{json_filename}_visualization.png")
    # plt.savefig(visualization_path)
    # print(f"Saved visualization: {visualization_path}")
    # plt.close()


def stitch_images_horizontally(images):
    """
    Stitch a list of images horizontally.

    Args:
        images (list): List of PIL.Image objects.

    Returns:
        PIL.Image: Stitched image.
    """
    widths, heights = zip(*(img.size for img in images))

    # Calculate total width and maximum height
    total_width = sum(widths)
    max_height = max(heights)

    # Create a blank canvas
    stitched_image = Image.new("RGB", (total_width, max_height))

    # Paste images side by side
    x_offset = 0
    for img in images:
        stitched_image.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    return stitched_image


def process_json_files(json_dir, image_dir, output_dir):
    """
    Process all JSON files in a directory.

    Args:
        json_dir (str): Directory containing JSON files.
        image_dir (str): Directory containing the images.
        output_dir (str): Directory to save visualizations.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json_files = [os.path.join(json_dir, file) for file in os.listdir(json_dir) if file.endswith(".json")]

    for json_file in json_files:
        extract_and_visualize_images(json_file, image_dir, output_dir)


# Example usage
json_dir = "/home/workstation/Documents/V2X_scene_reconstruction_tumtraf_dataset/meta_data"  # Replace with the path to your JSON files
image_dir = "/home/workstation/Downloads/combined_dataset"  # Replace with the directory containing your images
output_dir = "output2"  # Directory to save visualizations

process_json_files(json_dir, image_dir, output_dir)