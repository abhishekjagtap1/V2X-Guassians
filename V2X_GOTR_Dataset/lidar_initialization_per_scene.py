import open3d as o3d
import os
import argparse

def concatenate_pcd_files(input_folder, output_file):
    """Concatenate all PCD files from a folder and save the combined point cloud"""
    # Create an empty PointCloud object
    combined_pcd = o3d.geometry.PointCloud()

    # Iterate over each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.pcd'):
            # Load the point cloud file
            pcd = o3d.io.read_point_cloud(os.path.join(input_folder, filename))
            # Concatenate the point cloud to the combined point cloud
            combined_pcd += pcd

    # Save the combined point cloud
    o3d.io.write_point_cloud(output_file, combined_pcd)
    print(f"Combined point cloud saved as {output_file}")
    return combined_pcd

def voxel_downsample_point_cloud(pcd, voxel_size):
    """Downsample the point cloud using voxel filtering"""
    return pcd.voxel_down_sample(voxel_size=voxel_size)

def visualize_point_cloud(pcd):
    """Visualize the point cloud"""
    o3d.visualization.draw_geometries([pcd])

def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Concatenate lidar frames and downsample the point cloud")
    parser.add_argument("--lidar_input_path", type=str, required=True, help="Path to the folder containing lidar .pcd files")
    parser.add_argument("--combined_output_path", type=str, required=True, help="Path to save the combined point cloud")
    parser.add_argument("--voxel_size", type=float, default=0.05, help="Voxel size for downsampling")
    parser.add_argument("--downsampled_output_path", type=str, required=True, help="Path to save the downsampled point cloud")

    args = parser.parse_args()

    # Step 1: Combine all lidar frames
    print("Combining lidar frames...")
    combined_pcd = concatenate_pcd_files(args.lidar_input_path, args.combined_output_path)

    # Step 2: Voxel downsample the combined point cloud
    print(f"Voxel downsampling with voxel size {args.voxel_size}...")
    downsampled_pcd = voxel_downsample_point_cloud(combined_pcd, args.voxel_size)

    # Step 3: Save the downsampled point cloud
    print(f"Saving downsampled point cloud to {args.downsampled_output_path}...")
    o3d.io.write_point_cloud(args.downsampled_output_path, downsampled_pcd)

    # Step 4: Optionally visualize the downsampled point cloud
    visualize_point_cloud(downsampled_pcd)

if __name__ == "__main__":
    main()
