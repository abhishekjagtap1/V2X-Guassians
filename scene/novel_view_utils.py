import concurrent.futures
import gc
import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation as R






def add_noise_to_pose(pose, translation_noise_std=0.001, rotation_noise_std=0.001):
    noisy_pose = pose.copy()

    # 1. Add noise to the translation vector (last column, first three rows)
    translation_noise = np.random.normal(0, translation_noise_std, size=(3,))
    noisy_pose[:3, 3] += translation_noise

    # 2. Add noise to the rotation matrix
    # Generate random small rotation angles
    angles = np.random.normal(0, rotation_noise_std, size=(3,))  # roll, pitch, yaw
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])

    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])

    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])

    # Combine the small rotations
    noise_rotation = Rz @ Ry @ Rx
    noisy_pose[:3, :3] = noisy_pose[:3, :3] @ noise_rotation  # Apply rotation noise

    return noisy_pose




def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    poses = poses @ blender2opencv
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[
    :3
    ] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = np.concatenate(
        [poses, last_row], 1
    )  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    #     poses_centered = poses_centered  @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.0])

    for theta in np.linspace(0.0, 2.0 * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
            * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses


def process_video(video_data_save, video_path, img_wh, downsample, transform):
    """
    Load video_path data to video_data_save tensor.
    """
    video_frames = cv2.VideoCapture(video_path)
    count = 0
    video_images_path = video_path.split('.')[0]
    image_path = os.path.join(video_images_path, "images")

    if not os.path.exists(image_path):
        os.makedirs(image_path)
        while video_frames.isOpened():
            ret, video_frame = video_frames.read()
            if ret:
                video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
                video_frame = Image.fromarray(video_frame)
                if downsample != 1.0:
                    img = video_frame.resize(img_wh, Image.LANCZOS)
                img.save(os.path.join(image_path, "%04d.png" % count))

                img = transform(img)
                video_data_save[count] = img.permute(1, 2, 0)
                count += 1
            else:
                break

    else:
        images_path = os.listdir(image_path)
        images_path.sort()

        for path in images_path:
            img = Image.open(os.path.join(image_path, path))
            if downsample != 1.0:
                img = img.resize(img_wh, Image.LANCZOS)
                img = transform(img)
                video_data_save[count] = img.permute(1, 2, 0)
                count += 1

    video_frames.release()
    print(f"Video {video_path} processed.")
    return None


# define a function to process all videos
def process_videos(videos, skip_index, img_wh, downsample, transform, num_workers=1):
    """
    A multi-threaded function to load all videos fastly and memory-efficiently.
    To save memory, we pre-allocate a tensor to store all the images and spawn multi-threads to load the images into this tensor.
    """
    all_imgs = torch.zeros(len(videos) - 1, 300, img_wh[-1], img_wh[-2], 3)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # start a thread for each video
        current_index = 0
        futures = []
        for index, video_path in enumerate(videos):
            # skip the video with skip_index (eval video)
            if index == skip_index:
                continue
            else:
                future = executor.submit(
                    process_video,
                    all_imgs[current_index],
                    video_path,
                    img_wh,
                    downsample,
                    transform,
                )
                futures.append(future)
                current_index += 1
    return all_imgs


def generate_novel_poses_infra(extrinsic_matrix, num_poses=30, translation_variation=0.1, rotation_variation=0.1):
    def perturb_matrix(matrix, translation_variation, rotation_variation):
        # Generate random small translations
        translation_perturbation = np.random.uniform(-translation_variation, translation_variation, size=(3,))

        # Generate random small rotations
        angle_x = np.random.uniform(-rotation_variation, rotation_variation)
        angle_y = np.random.uniform(-rotation_variation, rotation_variation)
        angle_z = np.random.uniform(-rotation_variation, rotation_variation)

        # Construct rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)]
        ])

        Ry = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)]
        ])

        Rz = np.array([
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1]
        ])

        # Combine rotations
        rotation_perturbation = Rz @ Ry @ Rx

        # Apply perturbations to the extrinsic matrix
        perturbed_matrix = np.copy(matrix)
        perturbed_matrix[:3, :3] = rotation_perturbation @ perturbed_matrix[:3, :3]
        perturbed_matrix[:3, 3] += translation_perturbation

        return perturbed_matrix

    novel_poses = [perturb_matrix(extrinsic_matrix, translation_variation, rotation_variation) for _ in
                   range(num_poses)]

    return novel_poses


def generate_translated_poses(extrinsic_matrix, translations):
    poses = []
    for translation in translations:
        perturbed_matrix = np.copy(extrinsic_matrix)
        perturbed_matrix[:3, 3] += translation
        poses.append(perturbed_matrix)
    return poses


def get_spiral(c2ws_all, near_fars, rads_scale=1.0, N_views=120):
    """
    Generate a set of poses using NeRF's spiral camera trajectory as validation poses.
    """
    # center pose
    c2w = average_poses(c2ws_all)

    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    dt = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / ((1.0 - dt) / close_depth + dt / inf_depth)

    # Get radii for spiral path
    zdelta = near_fars.min() * 0.2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    render_poses = render_path_spiral(
        c2w, up, rads, focal, zdelta, zrate=0.5, N=N_views
    )
    return np.stack(render_poses)


def generate_spiral_poses_vehicle(extrinsic_matrix, num_poses=20, translation_step=0.05, rotation_step=2):
    poses = []
    for i in range(num_poses):
        angle = i * rotation_step * (np.pi / 180)  # Convert degrees to radians
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        translation_vector = np.array([
            [1, 0, 0, translation_step * i],
            [0, 1, 0, translation_step * i],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        perturbed_matrix = np.matmul(rotation_matrix, extrinsic_matrix)
        perturbed_matrix = np.matmul(translation_vector, perturbed_matrix)

        poses.append(perturbed_matrix)
    return poses


"""
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"Here we go again", i want to go to ECCV2025, Render novel poses for visualization -June 10 2024
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

"""


def generate_zoom_extrinsics(initial_extrinsic, steps=30, zoom_distance=1.0, vehicle=bool):
    # Extract the rotation matrix and translation vector
    rotation_matrix = initial_extrinsic[:3, :3]
    translation_vector = initial_extrinsic[:3, 3]

    # Get the camera viewing direction (negative z-axis in camera space)
    # viewing_direction = rotation_matrix[:, 2]  # Third column of the rotation matrix
    if vehicle:
        viewing_direction = rotation_matrix[:, 1]
    else:
        viewing_direction = rotation_matrix[:, 1]

    # Create a list to hold the new extrinsic matrices
    extrinsics_list = []

    # Generate new extrinsic matrices by moving the camera along the viewing direction
    for i in range(steps):
        # Calculate the new translation vector
        new_translation_vector = translation_vector + (i / steps) * zoom_distance * viewing_direction

        # Construct the new extrinsic matrix
        new_extrinsic = np.eye(4, dtype=np.float32)
        new_extrinsic[:3, :3] = rotation_matrix
        new_extrinsic[:3, 3] = new_translation_vector

        # Append the new extrinsic matrix to the list
        extrinsics_list.append(new_extrinsic)

    return extrinsics_list


def reorder_extrinsic_matrices(image_poses):
    # Split the list into the first 30 and the remaining elements

    first_30 = image_poses[:30]
    remaining = image_poses[30:]

    # first_30 = image_poses[:20]
    # remaining = image_poses[20:]

    # Reverse the first 30 elements
    first_30_reversed = first_30[::-1]
    # remaining = remaining[::-1]

    # Concatenate the reversed first 30 elements with the remaining elements
    reordered_poses = first_30_reversed + remaining
    # reordered_poses = first_30_reversed + remaining_reversed -> Not good

    return reordered_poses


def compute_intermediate_matrices(E1, E2, num_intermediates=20):
    """
    Compute intermediate extrinsic matrices between E1 and E2.

    Parameters:
    E1 (numpy.ndarray): The first extrinsic matrix (4x4).
    E2 (numpy.ndarray): The second extrinsic matrix (4x4).
    num_intermediates (int): The number of intermediate matrices to compute (default is 20).

    Returns:
    list: A list of intermediate extrinsic matrices.
    """
    intermediate_matrices = []
    for i in range(1, num_intermediates + 1):
        t = i / (num_intermediates + 1.0)
        E_i = E1 + t * (E2 - E1)
        intermediate_matrices.append(E_i)
    return intermediate_matrices


# Example usage
def compute_intermediate_matrices_novel(E1, E2, num_intermediates=8):
    """
    Compute intermediate extrinsic matrices between E1 and E2.

    Parameters:
    E1 (numpy.ndarray): The first extrinsic matrix (4x4).
    E2 (numpy.ndarray): The second extrinsic matrix (4x4).
    num_intermediates (int): The number of intermediate matrices to compute (default is 8).

    Returns:
    list: A list of intermediate extrinsic matrices including E1 and E2.
    """
    intermediate_matrices = [E1]
    for i in range(1, num_intermediates + 1):
        t = i / (num_intermediates + 1.0)
        E_i = E1 + t * (E2 - E1)
        intermediate_matrices.append(E_i)
    intermediate_matrices.append(E2)
    return intermediate_matrices


import numpy as np
from scipy.spatial.transform import Rotation as R


def slerp(q1, q2, alpha):
    """ Spherical linear interpolation (slerp) between two quaternions """
    dot = np.dot(q1, q2)

    # Clamp dot product to stay within the domain of acos
    dot = np.clip(dot, -1.0, 1.0)

    # If the dot product is very close to 1, use linear interpolation
    if dot > 0.9995:
        result = (1.0 - alpha) * q1 + alpha * q2
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * alpha
    sin_theta = np.sin(theta)

    s1 = np.sin(theta_0 - theta) / sin_theta_0
    s2 = sin_theta / sin_theta_0

    return s1 * q1 + s2 * q2


def generate_intermediate_posesshortest(E1, E2, n_poses=12):
    # Extract rotation matrices and translation vectors
    R1 = E1[:3, :3]
    t1 = E1[:3, 3]

    R2 = E2[:3, :3]
    t2 = E2[:3, 3]

    # Convert rotation matrices to quaternions
    q1 = R.from_matrix(R1).as_quat()
    q2 = R.from_matrix(R2).as_quat()

    poses = []

    for i in range(n_poses):
        alpha = i / (n_poses - 1)  # interpolation factor
        # Spherical linear interpolation (slerp) for quaternions
        q_interpolated = slerp(q1, q2, alpha)
        # Linear interpolation for translation vectors
        t_interpolated = (1 - alpha) * t1 + alpha * t2
        # Convert the quaternion back to a rotation matrix
        R_interpolated = R.from_quat(q_interpolated).as_matrix()
        # Form the transformation matrix
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R_interpolated
        pose[:3, 3] = t_interpolated
        poses.append(pose)

    return poses


def shift_view(extrinsic_matrix, angle_deg=10, axis='y'):
    """
    Shift the camera view by rotating around the specified axis (X, Y, or Z),
    while keeping the camera's position fixed.

    Parameters:
    - extrinsic_matrix: The original 4x4 extrinsic matrix.
    - angle_deg: The angle (in degrees) to rotate the view (default is 10°).
    - axis: The axis to rotate around ('x', 'y', or 'z') (default is 'y' for left/right shift).

    Returns:
    - new_extrinsic: The new extrinsic matrix after rotating the view.
    """
    # Extract the rotation matrix and translation vector
    rotation_matrix = extrinsic_matrix[:3, :3]
    translation_vector = extrinsic_matrix[:3, 3]

    # Define a rotation matrix around the specified axis
    def rotation_around_axis(angle_deg, axis):
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        if axis == 'y':
            # Rotation around the Y-axis (up-down shift or left-right)
            return np.array([[cos_a, 0, sin_a],
                             [0, 1, 0],
                             [-sin_a, 0, cos_a]], dtype=np.float32)
        elif axis == 'x':
            # Rotation around the X-axis (tilt up-down)
            return np.array([[1, 0, 0],
                             [0, cos_a, -sin_a],
                             [0, sin_a, cos_a]], dtype=np.float32)
        elif axis == 'z':
            # Rotation around the Z-axis (roll right or left)
            return np.array([[cos_a, -sin_a, 0],
                             [sin_a, cos_a, 0],
                             [0, 0, 1]], dtype=np.float32)
        else:
            raise ValueError(f"Invalid axis: {axis}. Choose from 'x', 'y', or 'z'.")

    # Generate the rotation matrix based on the chosen axis and angle
    rotation_matrix_shift = rotation_around_axis(angle_deg, axis)

    # Apply the rotation to the original rotation matrix
    new_rotation_matrix = np.dot(rotation_matrix_shift, rotation_matrix)

    # Create the new extrinsic matrix
    new_extrinsic = np.eye(4, dtype=np.float32)
    new_extrinsic[:3, :3] = new_rotation_matrix  # Set the new rotation
    new_extrinsic[:3, 3] = translation_vector  # Keep the translation the same

    return new_extrinsic


import numpy as np
from scipy.spatial.transform import Rotation as R


def slerp(q1, q2, t):
    """
    Perform spherical linear interpolation (slerp) between two quaternions.

    Parameters:
        q1 (np.ndarray): Quaternion at t=0 (as a 4-element array).
        q2 (np.ndarray): Quaternion at t=1 (as a 4-element array).
        t (float): Interpolation factor between 0 and 1.

    Returns:
        np.ndarray: Interpolated quaternion.
    """
    # Compute the cosine of the angle between the two quaternions
    dot_product = np.dot(q1, q2)

    # If the dot product is negative, reverse one quaternion to take the shorter path
    if dot_product < 0.0:
        q1 = -q1
        dot_product = -dot_product

    # Clamp dot product to avoid numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate the angle between the quaternions
    theta_0 = np.arccos(dot_product)
    sin_theta_0 = np.sin(theta_0)

    # If the angle is very small, use linear interpolation
    if sin_theta_0 < 1e-6:
        return (1.0 - t) * q1 + t * q2

    # Perform slerp interpolation
    theta = theta_0 * t
    sin_theta = np.sin(theta)

    s1 = np.sin((1.0 - t) * theta_0) / sin_theta_0
    s2 = sin_theta / sin_theta_0

    return s1 * q1 + s2 * q2


def generate_novel_views_slerp(extrinsic_1, extrinsic_2, num_views=50):
    """
    Generate novel views by interpolating between two extrinsic matrices.

    Parameters:
        extrinsic_1 (np.ndarray): First extrinsic matrix (4x4).
        extrinsic_2 (np.ndarray): Second extrinsic matrix (4x4).
        num_views (int): Number of novel views to generate (default is 50).

    Returns:
        list of np.ndarray: List of interpolated extrinsic matrices (4x4 each).
    """
    # Extract rotation and translation components
    R1 = extrinsic_1[:3, :3]
    t1 = extrinsic_1[:3, 3]
    R2 = extrinsic_2[:3, :3]
    t2 = extrinsic_2[:3, 3]

    # Convert rotation matrices to quaternions
    q1 = R.from_matrix(R1).as_quat()
    q2 = R.from_matrix(R2).as_quat()

    # Interpolation factors between 0 and 1
    interpolation_factors = np.linspace(0, 1, num_views)

    # Initialize list for novel views
    novel_views = []

    # Interpolate each view
    for t in interpolation_factors:
        # Slerp interpolation for the quaternion
        q_interpolated = slerp(q1, q2, t)
        R_interpolated = R.from_quat(q_interpolated).as_matrix()

        # Linear interpolation for translation
        t_interpolated = (1 - t) * t1 + t * t2

        # Create the extrinsic matrix for the interpolated pose
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = R_interpolated
        extrinsic_matrix[:3, 3] = t_interpolated

        # Append to list of novel views
        novel_views.append(extrinsic_matrix)

    return novel_views


"""
Nerf Spiral
"""


def normalize(v):
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Compute an average pose from a set of poses.
    """
    center = poses[:, :3, 3].mean(0)
    rotation_matrix = R.from_matrix(poses[:, :3, :3]).mean().as_matrix()
    avg_pose = np.eye(4)
    avg_pose[:3, :3] = rotation_matrix
    avg_pose[:3, 3] = center
    return avg_pose


def render_path_spiral_new(c2w, up, rads, focal, zdelta, zrate, N=120):
    """
    Generate a list of camera poses following a spiral path.
    """
    render_poses = []
    for theta in np.linspace(0, 2 * np.pi * zrate, N):
        c = np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * 0.5)]) * rads
        z = normalize(c2w[:3, 2]) * zdelta
        pos = c + z + c2w[:3, 3]

        look_at = c2w[:3, 3]  # Look towards the center
        forward = normalize(look_at - pos)
        right = normalize(np.cross(up, forward))
        up_adjusted = np.cross(forward, right)

        # Build the 4x4 transformation matrix for this pose
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up_adjusted
        pose[:3, 2] = forward
        pose[:3, 3] = pos
        render_poses.append(pose)

    return render_poses


def get_spiral_poses_new(extrinsic_1, extrinsic_2, near_fars, rads_scale=1.0, N_views=120):
    """
    Generate a set of poses using a spiral camera trajectory.
    """
    # Stack the two extrinsic matrices to get an array of poses
    c2ws_all = np.stack([extrinsic_1, extrinsic_2])

    # Get average pose and compute the 'up' vector
    c2w = average_poses(c2ws_all)
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find the focal depth
    dt = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / ((1.0 - dt) / close_depth + dt / inf_depth)

    # Calculate radii and depth delta for spiral path
    zdelta = near_fars.min() * 0.2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale

    # Generate the spiral path
    render_poses = render_path_spiral_new(c2w, up, rads, focal, zdelta, zrate=0.5, N=N_views)

    return np.stack(render_poses)
