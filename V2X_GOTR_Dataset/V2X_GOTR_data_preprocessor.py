import argparse
import os
import shutil
import json


def create_core_meta_data(dataset_path):
    """
    Process the TUMTRAF dataset located at the given path for V2X gaussian splatting.
    """

    # Step 1: Verify dataset path
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return
    print(f"Processing dataset at: {dataset_path}")

    # Step 2: Create 'meta_data' folder in the current working directory
    meta_data_folder = "meta_data"
    if not os.path.exists(meta_data_folder):
        os.makedirs(meta_data_folder)
        print(f"Created folder: {meta_data_folder}")
    else:
        print(f"Folder already exists: {meta_data_folder}")

    # Step 3:  Store combined metadata from the TUMTRAF dataset.
    splits = ['train', 'val', 'test']

    for split in splits:
        split_folder = os.path.join(dataset_path, split,
                                    'labels_point_clouds/s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered')
        if not os.path.exists(split_folder):
            print(f"Warning: Split folder does not exist: {split_folder}")
            continue

        print(f"Processing split: {split}")
        for file_name in os.listdir(split_folder):
            src_file = os.path.join(split_folder, file_name)
            dst_file = os.path.join(meta_data_folder, file_name)

            # Ensure the source is a file before copying
            if os.path.isfile(src_file):
                shutil.copy(src_file, dst_file)
                print(f"Copied: {src_file} -> {dst_file}")
            else:
                print(f"Skipped (not a file): {src_file}")


def setup_gaussian_splatting_dataset(main_json, meta_data_dir, output_base_dir, dataset_path, scene_size):
    """
    Set up a dataset folder based on scene descriptions and metadata.
    """

    # Load the main JSON file containing scenes
    with open(main_json, "r") as file:
        scene_data = json.load(file)

    # Ensure output base directory exists
    os.makedirs(output_base_dir, exist_ok=True)

    # Process each scene described in the main JSON
    for scene in scene_data["scenes"]:
        scene_name = scene.get("scene_name")
        start_frame = scene.get("start_frame")

        if not scene_name or not start_frame:
            print(f"Warning: Missing 'scene_name' or 'start_frame' for a scene. Skipping...")
            continue

        # Create scene folder structure
        scene_folder = os.path.join(output_base_dir, scene_name)
        os.makedirs(scene_folder, exist_ok=True)
        os.makedirs(os.path.join(scene_folder, "meta_data"), exist_ok=True)
        for subfolder in ["cam1", "cam2", "cam3", "lidar"]:
            os.makedirs(os.path.join(scene_folder, subfolder), exist_ok=True)

        print(f"Processing scene '{scene_name}' starting from frame '{start_frame}'...")

        # Copy metadata JSON files in ascending order starting from start_frame
        all_json_files = sorted(f for f in os.listdir(meta_data_dir) if f.endswith(".json"))
        matching_index = next((i for i, f in enumerate(all_json_files) if start_frame in f), -1)

        if matching_index == -1:
            print(f"Error: Start frame '{start_frame}' not found in metadata directory. Skipping scene...")
            continue

        selected_json_files = all_json_files[matching_index:matching_index + scene_size]

        for json_file in selected_json_files:
            src_path = os.path.join(meta_data_dir, json_file)
            dst_path = os.path.join(scene_folder, "meta_data", json_file)
            shutil.copy2(src_path, dst_path)

        # Path to the folder containing JSON files
        json_folder_path = os.path.join(scene_folder, "meta_data")
        json_files = [f for f in os.listdir(json_folder_path) if f.endswith(".json")]

        # Loop through each JSON file in the folder
        for json_file in json_files:
            json_file_path = os.path.join(json_folder_path, json_file)
            print(f"Processing file: {json_file}")

            # Output dictionary to store extracted filenames
            extracted_files = {"cam1": None, "cam2": None, "cam3": None, "lidar": None}

            try:
                # Open and parse the JSON file
                with open(json_file_path, "r") as file:
                    data = json.load(file)

                # Iterate over frames in the JSON data (frame keys can vary)
                frames = data["openlabel"]["frames"]
                for frame_key, frame_data in frames.items():
                    # Assuming the frame data has the key "frame_properties"
                    frame_properties = frame_data.get("frame_properties", {})

                    # Extract specific image files
                    image_files = frame_properties.get("image_file_names", [])
                    for img_file in image_files:
                        if "south1_8mm" in img_file:
                            extracted_files["cam3"] = img_file
                        elif "south2_8mm" in img_file:
                            extracted_files["cam1"] = img_file
                        elif "vehicle_camera_basler_16mm" in img_file:
                            extracted_files["cam2"] = img_file

                    # Extract specific point cloud file
                    point_cloud_files = frame_properties.get("point_cloud_file_names", [])
                    for pc_file in point_cloud_files:
                        if "s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered" in pc_file:
                            extracted_files["lidar"] = pc_file

                # Process the extracted files and copy to the respective directories

                for cam_key, cam_filename in extracted_files.items():
                    if cam_filename:
                        # Flag to indicate if the file is found
                        file_found = False

                        # Check the file in each of the folders: train, val, test
                        for folder in ["train", "val", "test"]:
                            if cam_key == "lidar":
                                sub_dataset_folder = "s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered"
                                file_path = os.path.join(dataset_path, folder, "point_clouds", sub_dataset_folder,
                                                         cam_filename)
                                dst_dir = os.path.join(scene_folder, "lidar")
                            else:
                                if cam_key == 'cam1':
                                    sub_dataset_folder = 's110_camera_basler_south2_8mm'
                                elif cam_key == 'cam2':
                                    sub_dataset_folder = 'vehicle_camera_basler_16mm'
                                elif cam_key == 'cam3':
                                    sub_dataset_folder = 's110_camera_basler_south1_8mm'

                                file_path = os.path.join(dataset_path, folder, "images", sub_dataset_folder,
                                                         cam_filename)
                                dst_dir = os.path.join(scene_folder, cam_key)  # cam1, cam2, or cam3

                            # Verify the file exists before copying
                            if os.path.exists(file_path):
                                os.makedirs(dst_dir, exist_ok=True)
                                shutil.copy(file_path, dst_dir)
                                print(f"Copied {cam_key} file {cam_filename} from folder {folder} to {dst_dir}")

                                # Set file_found flag to True and break from the folder loop
                                file_found = True
                                break  # Exit the loop as we found the file in one folder
                            else:
                                print(f"File not found in {folder}: {file_path}")

                        # If the file was not found in any folder, log the error
                        if not file_found:
                            print(
                                f"Error: File {cam_filename} not found in any folder (train, val, test) for {cam_key}")


            except KeyError as e:
                print(f"Error: Missing key in file {json_file} -> {e}")
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON format in file {json_file}")
            except Exception as e:
                print(f"Error: {e}")


def main():
    """
    Main function to parse arguments and preprocess the dataset.
    """
    parser = argparse.ArgumentParser(description="TUMTRAF Dataset Preprocessor for V2X Gaussian Splatting.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the TUMTRAF dataset.")

    args = parser.parse_args()
    dataset_path = args.dataset
    create_core_meta_data(dataset_path)

    main_json = "V2X_GOTR_Dataset/tumtraf_scene_selection.json"  # Path to the scene description file
    meta_data_dir = "V2X_GOTR_Dataset/meta_data"  # Path to the folder with 1000 JSON files
    output_base_dir = "data"  # Output directory for the organized dataset

    # Run the scene setup function
    setup_gaussian_splatting_dataset(main_json, meta_data_dir, output_base_dir, dataset_path, 50)


if __name__ == "__main__":
    main()
