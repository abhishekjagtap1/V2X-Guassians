import concurrent.futures
from scene.novel_view_utils import *
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
from scene.novel_view_utils import add_noise_to_pose




class V2XGOTR_Dataset(Dataset):
    def __init__(
            self,
            datadir,
            split="train",
            downsample=1.0,
            is_stack=True,
            N_vis=-1,
            time_scale=1.0,
            scene_bbox_min=[-1.0, -1.0, -1.0],
            scene_bbox_max=[1.0, 1.0, 1.0],
            bd_factor=0.75,
            eval_step=1,
            eval_index=0,
    ):
        self.img_wh = (
            int(1920 / downsample),
            int(1200 / downsample),
        )
        self.root_dir = datadir
        self.split = split
        self.downsample = 1
        self.is_stack = is_stack
        self.N_vis = N_vis
        self.time_scale = time_scale
        self.scene_bbox = torch.tensor([scene_bbox_min, scene_bbox_max])

        self.world_bound_scale = 1.1
        self.bd_factor = bd_factor
        self.eval_step = eval_step
        self.eval_index = eval_index
        self.blender2opencv = np.eye(4)
        self.transform = T.ToTensor()

        self.near = 1.0
        self.far = 100
        self.near_far = [self.near, self.far] #Dont use a rigid one
        self.white_bg = False
        self.ndc_ray = True
        self.depth_data = False

        self.load_meta(datadir)
        print(f"meta data loaded, total image:{len(self)}")

    def load_meta(self, datadir):
        """
        Load meta data from the dataset preprocessed V2X Gaussian On The Road Dataset
        """


        self.focal = [2726.550048828125, 2726.550048828125] # Not neccesary to use (Note: this is not used)
        self.image_paths, self.image_poses, self.image_times, self.depth_paths = self.load_images_path(datadir,
                                                                                                       self.split)

        """
        Note: Render Novel val poses is still under construction please dont use the below val poses for training
        """

    def get_val_pose(self):
        render_poses = self.val_poses
        render_times = torch.linspace(0.0, 1.0, render_poses.shape[0]) * 2.0 - 1.0
        return render_poses, self.time_scale * render_times

    def load_images_path(self, datadir, split):
        image_paths = []
        image_poses = []
        image_times = []
        depth_paths = []
        N_cams = 0
        N_time = 0
        countss = 300

        for root, dirs, files in os.walk(datadir):
            for dir in dirs:
                if dir == "cam1":  # South 2
                    N_cams += 1
                    image_folders = os.path.join(root, dir)
                    image_files = sorted(os.listdir(image_folders))
                    this_count = 0

                    for img_file in image_files:
                        if this_count >= countss: break
                        img_index = image_files.index(img_file)
                        images_path = os.path.join(image_folders, img_file)
                        if self.depth_data:
                            depth_folder = os.path.join(root, "cam01_depth")
                            depth_files = sorted(os.listdir(depth_folder))
                            depth_files = depth_files[:20]
                            assert len(depth_files) == len(image_files)
                            for depth_file in depth_files:
                                depth_path = os.path.join(depth_folder, depth_file)

                        intrinsic_south_2 = np.asarray([[1315.158203125, 0.0, 962.7348338975571],
                                                        [0.0, 1362.7757568359375, 580.6482296623581],
                                                        [0.0, 0.0, 1.0]], dtype=np.float32)
                        extrinsic_south_2 = np.asarray([[0.6353517, -0.24219051, 0.7332613, -0.03734626],
                                                        [-0.7720766, -0.217673, 0.5970893, 2.5209506],
                                                        [0.01500183, -0.9454958, -0.32528937, 0.543223],
                                                        [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
                        south_2_proj = np.asarray([[1546.63215008, -436.92407115, -295.58362676, 1319.79271737],
                                                   [93.20805656, 47.90351592, -1482.13403199, 687.84781276],
                                                   [0.73326062, 0.59708904, -0.32528854, -1.30114325]],
                                                  dtype=np.float32)

                        """
                        Use this only for rendering novel poses
                        """

                        # novel_poses = generate_novel_poses_infra(extrinsic_south_2, num_poses= 30, translation_variation=0.05, rotation_variation=0.1)
                        # novel_poses = get_spiral(poses, self.near_fars, N_views=N_views)

                        # for i, pose in enumerate(novel_poses):
                        #   print(f"Pose {i + 1}:\n{pose}\n")

                        """
                        Use this for up and down poses


                        translation_variations_up = [[0, i * 0.09, 0] for i in range(1, 11)]
                        translation_variations_down = [[0, -i * 0.05, 0] for i in range(1, 11)]

                        novel_poses_up = generate_translated_poses(extrinsic_south_2, translation_variations_up)
                        novel_poses_down = generate_translated_poses(extrinsic_south_2, translation_variations_down)

                        novel_poses = novel_poses_up + novel_poses_down
                        """

                        """
                        Use this for left and right

                        translation_variations_left = [[-i * 0.1, 0, 0] for i in range(1, 11)]
                        translation_variations_right = [[i * 0.1, 0, 0] for i in range(1, 11)]

                        novel_poses_left = generate_translated_poses(extrinsic_south_2, translation_variations_left)
                        novel_poses_right = generate_translated_poses(extrinsic_south_2, translation_variations_right)

                        novel_poses = novel_poses_left + novel_poses_right
                        """

                        """
                        dumb hack
                        """

                        # Generate 30 extrinsic matrices with a zoom distance of 1.0 units
                        zoom_extrinsics = generate_zoom_extrinsics(extrinsic_south_2, steps=len(image_files),
                                                                   zoom_distance=3.0, vehicle=False)
                        # zoom_extrinsics = generate_zoom_extrinsics(extrinsic_south_2, steps=20, zoom_distance=4.0, vehicle=False)
                        update_extrinsics = zoom_extrinsics[img_index]
                        # print("I am using zoomed ones")

                        """
                        Lets try new one 
                        """
                        E2 = np.array([[-9.9610770e-01, 2.4993204e-02, -8.4530041e-02, 1.5214223e+01],
                                       [8.0638155e-02, -1.2893400e-01, -9.8836863e-01, 1.2008096e+00],
                                       [-3.5601299e-02, -9.9133861e-01, 1.2641662e-01, 5.5706013e+01],
                                       [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]], dtype=np.float32)

                        E1 = np.array([[0.6353517, -0.24219051, 0.7332613, -0.03734626],
                                       [-0.7720766, -0.217673, 0.5970893, 2.5209506],
                                       [0.01500183, -0.9454958, -0.32528937, 0.543223],
                                       [0., 0., 0., 1.]], dtype=np.float32)

                        E3 = np.array([[9.8894250e-01, -1.1535851e-02, -1.4785174e-01, -2.7661972e+01],
                                       [1.4816706e-01, 1.1927225e-01, 9.8174328e-01, -1.7739400e+00],
                                       [6.3093919e-03, -9.9279499e-01, 1.1966250e-01, 3.2480175e+01],
                                       [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]],
                                      dtype=np.float32)

                        """
                        Intermediate poses for south2 and vehicle for Less dynmaic scene
                        """
                        E_dynamic = np.array([
                            [1.9375929e-01, 1.5485874e-01, 9.6874940e-01, -5.4834819e-01],
                            [-9.8104781e-01, 3.2331135e-02, 1.9105035e-01, -5.0577650e+00],
                            [-1.7348743e-03, -9.8740792e-01, 1.5818821e-01, 9.9938498e+00],
                            [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]
                        ], dtype=np.float32)

                        intrinsic_vehicle = np.asarray([[2726.55, 0.0, 685.235],
                                                        [0.0, 2676.64, 262.745],
                                                        [0.0, 0.0, 1.0]], dtype=np.float32)
                        transformation_matrix_base_to_camera_south_1 = np.array([
                            [0.891382638626301, 0.37756862104528707, -0.07884507325924934, 25.921784677055939],
                            [0.2980421080238165, -0.6831891949380544, -0.6660273169946723, 13.668310799382738],
                            [-0.24839844089507856, 0.5907739097931769, -0.7525203649548087, 18.630430017833277],
                            [0, 0, 0, 1]], dtype=float)
                        transformation_matrix_lidar_to_base_south_1 = np.array([
                            [0.247006, -0.955779, -0.15961, -16.8017],
                            [0.912112, 0.173713, 0.371316, 4.66979],
                            [-0.327169, -0.237299, 0.914685, 6.4602],
                            [0.0, 0.0, 0.0, 1.0], ], dtype=float)

                        extrinsic_matrix_lidar_to_camera_south_1 = np.matmul(
                            transformation_matrix_base_to_camera_south_1,
                            transformation_matrix_lidar_to_base_south_1)
                        camera_to_lidar_extrinsics_south_1 = np.linalg.inv(extrinsic_matrix_lidar_to_camera_south_1)

                        intermediate_matrices = compute_intermediate_matrices(E1, camera_to_lidar_extrinsics_south_1,
                                                                              num_intermediates=len(image_files))
                        # intermediate_matrices = generate_intermediate_posesshortest(E1, E_dynamic, n_poses=len(image_files))

                        # intermediate_matrices = compute_intermediate_matrices_novel(E2, E1, num_intermediates=20)

                        new_extrinsic_up = shift_view(extrinsic_south_2, angle_deg=-img_index, axis='z')
                        near_fars = np.array([0.01, 100.0])
                        novel_views = generate_novel_views_slerp(extrinsic_south_2, camera_to_lidar_extrinsics_south_1,
                                                                 num_views=len(image_files))
                        update_inter = intermediate_matrices[img_index]

                        if img_index < 30:
                            intrinsic_matrix_for_rendering = intrinsic_south_2
                        else:
                            intrinsic_matrix_for_rendering = intrinsic_vehicle

                        noisy_pose = add_noise_to_pose(extrinsic_south_2, translation_noise_std=0.05,
                                                       rotation_noise_std=0.01)

                        image_pose_dict = {
                            'intrinsic_matrix': intrinsic_south_2,
                            # intrinsic_matrix_for_rendering, #intrinsic_vehicle, #
                            'extrinsic_matrix': extrinsic_south_2,
                            # update_inter, #new_extrinsic_up, #update_inter, #noisy_pose, #extrinsic_south_2, # novel_views[img_index], #novel_views[img_index], #new_extrinsic_up, #update_inter, #extrinsic_south_2, #, # update_extrinsics, # #novel_poses[img_index], # #extrinsic_south_2, #
                            "projection_matrix": south_2_proj,
                            "cam_id": 1
                        }
                        N_time = len(image_files)
                        image_paths.append(os.path.join(images_path))
                        if self.depth_data:
                            depth_paths.append(os.path.join(depth_path))

                        image_times.append(float(img_index / len(image_files)))
                        image_poses.append(image_pose_dict)

                        this_count += 1

                if dir == "cam2":
                    N_cams += 1
                    image_folders = os.path.join(root, dir)
                    image_files = sorted(os.listdir(image_folders))
                    metadata_folders = os.path.join(root, "meta_data")
                    meta_files = sorted(os.listdir(metadata_folders))
                    this_count = 0

                    for img_file, meta_file in zip(image_files, meta_files):
                        images_path = os.path.join(image_folders, img_file)
                        img_index = image_files.index(img_file)
                        metadata_path = os.path.join(metadata_folders, meta_file)
                        intrinsic_vehicle = np.asarray([[2726.55, 0.0, 685.235],
                                                        [0.0, 2676.64, 262.745],
                                                        [0.0, 0.0, 1.0]], dtype=np.float32)
                        intrinsic_south_2 = np.asarray([[1315.158203125, 0.0, 962.7348338975571],
                                                        [0.0, 1362.7757568359375, 580.6482296623581],
                                                        [0.0, 0.0, 1.0]], dtype=np.float32)
                        vehicle_cam_to_lidar = np.asarray([[0.12672871, 0.12377692, 0.9841849, 0.14573078],  # TBD
                                                           [-0.9912245, -0.02180046, 0.13037732, 0.19717109],
                                                           [0.03759337, -0.99207014, 0.11992808, -0.02214238],
                                                           [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
                        if self.depth_data:
                            depth_folder = os.path.join(root, "cam02_depth")
                            depth_files = sorted(os.listdir(depth_folder))
                            depth_files = depth_files[:20]
                            assert len(depth_files) == len(image_files)
                            for depth_file in depth_files:
                                depth_path = os.path.join(depth_folder, depth_file)

                        import json
                        with open(metadata_path, 'r') as file:
                            calib_data = json.load(file)
                            frames = calib_data.get("openlabel", {}).get("frames", {})

                            for frame_key, frame_data in frames.items():
                                transforms = frame_data.get("frame_properties", {}).get("transforms", {})

                                for transform_key, transform_data in transforms.items():
                                    matrix = transform_data.get("transform_src_to_dst", {}).get("matrix4x4")

                        vehicle_to_infra_transformation_matrix = np.array(matrix)

                        extrinsic_matrix_vehicle = np.matmul(np.linalg.inv(vehicle_cam_to_lidar),
                                                             np.linalg.inv(vehicle_to_infra_transformation_matrix))

                        # import numpy as np

                        # Shared intrinsic matrix
                        intrinsic_south_2 = np.array([
                            [1315.158203125, 0.0, 962.7348338975571],
                            [0.0, 1362.7757568359375, 580.6482296623581],
                            [0.0, 0.0, 1.0]
                        ], dtype=np.float32)

                        # Third camera's intrinsic matrix
                        intrinsic_vehicle = np.array([
                            [2726.55, 0.0, 685.235],
                            [0.0, 2676.64, 262.745],
                            [0.0, 0.0, 1.0]
                        ], dtype=np.float32)

                        # Compute T_intrinsic
                        T_intrinsic = np.dot(intrinsic_south_2, np.linalg.inv(intrinsic_vehicle))

                        # Assuming extrinsic_matrix_vehicle is a 4x4 matrix
                        # extrinsic_matrix_vehicle = np.matmul(np.linalg.inv(vehicle_cam_to_lidar), np.linalg.inv(vehicle_to_infra_transformation_matrix))

                        # Extract rotation (R) and translation (t)
                        R_vehicle = extrinsic_matrix_vehicle[:3, :3]  # 3x3 rotation
                        t_vehicle = extrinsic_matrix_vehicle[:3, 3:]  # 3x1 translation

                        # Adjust the rotation using T_intrinsic
                        R_vehicle_adjusted = np.dot(T_intrinsic, R_vehicle)

                        # Combine adjusted rotation and original translation to form a 3x4 matrix
                        extrinsic_matrix_vehicle_adjusted = np.hstack((R_vehicle_adjusted, t_vehicle))

                        # Convert to 4x4 homogeneous matrix
                        extrinsic_matrix_vehicle_adjusted_homogeneous = np.vstack(
                            (extrinsic_matrix_vehicle_adjusted, [0, 0, 0, 1]))

                        infra_to_vehicle = np.linalg.inv(extrinsic_matrix_vehicle)
                        transformation_matrix_s110_lidar_ouster_south_to_s110_base = np.array([
                            [0.21479485, -0.9761028, 0.03296187, -15.87257873],
                            [0.97627128, 0.21553835, 0.02091894, 2.30019086],
                            [-0.02752358, 0.02768645, 0.99923767, 7.48077521],
                            [0.00000000, 0.00000000, 0.00000000, 1.00000000]
                        ])
                        shata = np.matmul(transformation_matrix_s110_lidar_ouster_south_to_s110_base, infra_to_vehicle)
                        vehicle_proj = np.matmul(np.hstack((intrinsic_vehicle, np.zeros((3, 1), dtype=np.float32))),
                                                 extrinsic_matrix_vehicle)
                        # Vehicle transformation needs to be transposed inorder to maintaion consistentcy
                        # Use extrinsic_v  for training
                        extrinsic_v = np.eye(4).astype(np.float32)
                        extrinsic_v[:3, :3] = extrinsic_matrix_vehicle[:3, :3].transpose()
                        extrinsic_v[:3, 3] = extrinsic_matrix_vehicle[:3, 3]

                        # extrinsic_v = np.eye(4).astype(np.float32)
                        # extrinsic_v[:3, :3] = shata[:3, :3].transpose()
                        # extrinsic_v[:3, 3] = shata[:3, 3]

                        if img_index == 0:
                            save_first_pose = extrinsic_v

                        """
                        Use for up and down movments for vehicle
                        """
                        # print(len(image_files))

                        # translation_variations_up = [[0, i * 0.9, 0] for i in range(1, 21)]
                        # translation_variations_down = [[0, -i * 0.05, 0] for i in range(1, 16)]

                        # novel_poses_up = generate_translated_poses(extrinsic_v, translation_variations_up)
                        # novel_poses_down = generate_translated_poses(extrinsic_v, translation_variations_down)

                        # novel_poses_vehicle = novel_poses_up #+ novel_poses_down

                        zoom_extrinsics_ve = generate_zoom_extrinsics(save_first_pose, steps=len(image_files),
                                                                      zoom_distance=2.0, vehicle=True)
                        # zoom_extrinsics_ve = generate_zoom_extrinsics(save_first_pose, steps=20, zoom_distance=1.0, vehicle=True)
                        update_extrinsics_ve = zoom_extrinsics_ve[img_index]

                        """
                        Use for left and right movments for vehicle


                        translation_variations_left = [[-i * 0.1, 0, 0] for i in range(1, 31)]
                        translation_variations_right = [[i * 0.1, 0, 0] for i in range(1, 31)]

                        #novel_poses_left = generate_translated_poses(extrinsic_v, translation_variations_left)
                        #novel_poses_right = generate_translated_poses(extrinsic_v, translation_variations_right)

                        novel_poses_vehicle = novel_poses_right #+novel_poses_left #
                        """

                        # novel_poses_vehicle = generate_novel_poses_infra(extrinsic_v, translation_variation=0.05, rotation_variation=0.05)

                        """
                        Lets try new one shortest imtermediate novel poses
                        """
                        E1 = np.array([[-9.9610770e-01, 2.4993204e-02, -8.4530041e-02, 1.5214223e+01],
                                       [8.0638155e-02, -1.2893400e-01, -9.8836863e-01, 1.2008096e+00],
                                       [-3.5601299e-02, -9.9133861e-01, 1.2641662e-01, 5.5706013e+01],
                                       [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]], dtype=np.float32)

                        E2 = np.array([[0.6353517, -0.24219051, 0.7332613, -0.03734626],
                                       [-0.7720766, -0.217673, 0.5970893, 2.5209506],
                                       [0.01500183, -0.9454958, -0.32528937, 0.543223],
                                       [0., 0., 0., 1.]], dtype=np.float32)
                        # neww nexz_scene_20 ge e1
                        E1 = np.array([[-9.9896169e-01, 7.1184416e-03, -4.5003839e-02, 1.3543032e+01],
                                       [4.3314915e-02, -1.5806700e-01, -9.8647743e-01, -2.8491244e+00],
                                       [-1.4135795e-02, -9.8740315e-01, 1.5759440e-01, 2.6300457e+01],
                                       [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]], dtype=np.float32)
                        # New E1
                        E1 = np.array([[-9.9939364e-01, 3.4756299e-02, 2.2151275e-03, 1.3439852e+01],
                                       [-8.0836415e-03, -1.6962631e-01, -9.8547488e-01, -4.6583529e+00],
                                       [-3.3875708e-02, -9.8489583e-01, 1.6980423e-01, 1.5624603e+01],
                                       [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]], dtype=np.float32)
                        E3 = np.array([[9.8894250e-01, -1.1535851e-02, -1.4785174e-01, -2.7661972e+01],
                                       [1.4816706e-01, 1.1927225e-01, 9.8174328e-01, -1.7739400e+00],
                                       [6.3093919e-03, -9.9279499e-01, 1.1966250e-01, 3.2480175e+01],
                                       [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]],
                                      dtype=np.float32)

                        """
                        Intermediate poses for south2 and vehicle for Less dynmaic scene
                        """
                        E_dynamic = np.array([
                            [1.9375929e-01, 1.5485874e-01, 9.6874940e-01, -5.4834819e-01],
                            [-9.8104781e-01, 3.2331135e-02, 1.9105035e-01, -5.0577650e+00],
                            [-1.7348743e-03, -9.8740792e-01, 1.5818821e-01, 9.9938498e+00],
                            [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]
                        ], dtype=np.float32)

                        intrinsic_south_2 = np.asarray([[1315.158203125, 0.0, 962.7348338975571],
                                                        [0.0, 1362.7757568359375, 580.6482296623581],
                                                        [0.0, 0.0, 1.0]], dtype=np.float32)

                        intermediate_matrices = compute_intermediate_matrices(E_dynamic, E2,
                                                                              num_intermediates=len(image_files))
                        # intermediate_matrices = compute_intermediate_matrices_novel(E1, E2, num_intermediates=20)
                        update_inter_veh = intermediate_matrices[img_index]

                        if img_index < 3:
                            intrinsic_matrix_for_rendering = intrinsic_vehicle
                        else:
                            intrinsic_matrix_for_rendering = intrinsic_south_2

                        noisy_pose_vehicle = add_noise_to_pose(extrinsic_v, translation_noise_std=0.05,
                                                               rotation_noise_std=0.01)

                        image_pose_dict = {
                            'intrinsic_matrix': intrinsic_vehicle,
                            # intrinsic_south_2, #intrinsic_vehicle, ##intrinsic_matrix_for_rendering, ## #intrinsic_south_2, # #intrinsic_vehicle, #
                            'extrinsic_matrix': extrinsic_v,
                            # extrinsic_v, #update_inter_veh, #extrinsic_v, #update_extrinsics_ve, #  , novel_poses_vehicle[img_index], # #
                            "projection_matrix": vehicle_proj,
                            "cam_id": 2
                        }
                        N_time = len(image_files)
                        image_paths.append(os.path.join(images_path))
                        if self.depth_data:
                            depth_paths.append(os.path.join(depth_path))
                        image_times.append(float(img_index / len(image_files)))
                        image_poses.append(image_pose_dict)

                        this_count += 1

                if dir == "cam3":  # South 1
                    N_cams += 1
                    image_folders = os.path.join(root, dir)
                    image_files = sorted(os.listdir(image_folders))
                    this_count = 0
                    for img_file in image_files:
                        if this_count >= countss: break
                        img_index = image_files.index(img_file)
                        images_path = os.path.join(image_folders, img_file)
                        if self.depth_data:
                            depth_folder = os.path.join(root, "cam01_depth")
                            depth_files = sorted(os.listdir(depth_folder))
                            depth_files = depth_files[:20]
                            assert len(depth_files) == len(image_files)
                            for depth_file in depth_files:
                                depth_path = os.path.join(depth_folder, depth_file)

                        intrinsic_south_1 = np.asarray([[1400.3096617691212, 0.0, 967.7899705163408],
                                                        [0.0, 1403.041082755918, 581.7195041357244],
                                                        [0.0, 0.0, 1.0]], dtype=np.float32)
                        intrinsic_south_2 = np.asarray([[1315.158203125, 0.0, 962.7348338975571],
                                                        [0.0, 1362.7757568359375, 580.6482296623581],
                                                        [0.0, 0.0, 1.0]], dtype=np.float32)

                        extrinsic_south_1 = np.asarray([[0.41204962, -0.45377758, 0.7901276, 2.158825],
                                                        [-0.9107832, -0.23010845, 0.34281868, -15.5765505],
                                                        [0.02625162, -0.86089253, -0.5081085, 0.08758777],
                                                        [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

                        extrinsic_south_1_calib = np.asarray([[0.95302056, - 0.30261307, 0.01330958, 1.77326515],
                                                              [-0.12917788, - 0.44577866, - 0.88577337, 7.60903957],
                                                              [0.27397972, 0.84244093, - 0.46392715, 4.04778098],
                                                              [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
                        transformation_matrix_base_to_camera_south_1 = np.array([
                            [0.891382638626301, 0.37756862104528707, -0.07884507325924934, 25.921784677055939],
                            [0.2980421080238165, -0.6831891949380544, -0.6660273169946723, 13.668310799382738],
                            [-0.24839844089507856, 0.5907739097931769, -0.7525203649548087, 18.630430017833277],
                            [0, 0, 0, 1]], dtype=float)
                        transformation_matrix_lidar_to_base_south_1 = np.array([
                            [0.247006, -0.955779, -0.15961, -16.8017],
                            [0.912112, 0.173713, 0.371316, 4.66979],
                            [-0.327169, -0.237299, 0.914685, 6.4602],
                            [0.0, 0.0, 0.0, 1.0], ], dtype=float)

                        extrinsic_matrix_lidar_to_camera_south_1 = np.matmul(
                            transformation_matrix_base_to_camera_south_1,
                            transformation_matrix_lidar_to_base_south_1)
                        camera_to_lidar_extrinsics_south_1 = np.linalg.inv(extrinsic_matrix_lidar_to_camera_south_1)
                        new_extrinsic_up = shift_view(camera_to_lidar_extrinsics_south_1, angle_deg=img_index, axis='z')

                        south_1_proj = np.asarray(
                            [[1279.275240545117, -862.9254609474538, -443.6558546306608, -16164.33175985643],
                             [-57.00793327192514, -67.92432779187584, -1461.785310749125, -806.9258947569469],
                             [0.7901272773742676, 0.3428181111812592, -0.508108913898468, 3.678680419921875]],
                            dtype=np.float32)
                        # novel_views = generate_novel_views_slerp(camera_to_lidar_extrinsics_south_1, extrinsic_south_2,
                        #                                        num_views=len(image_files))
                        noisy_pose_cam3 = add_noise_to_pose(camera_to_lidar_extrinsics_south_1,
                                                            translation_noise_std=0.05, rotation_noise_std=0.01)
                        image_pose_dict = {
                            'intrinsic_matrix': intrinsic_south_2,
                            'extrinsic_matrix': camera_to_lidar_extrinsics_south_1,
                            # noisy_pose_cam3, #camera_to_lidar_extrinsics_south_1, #novel_views[img_index], #new_extrinsic_up, #camera_to_lidar_extrinsics_south_1, #extrinsic_south_1, #c
                            "projection_matrix": south_1_proj,
                            "cam_id": 3
                        }
                        N_time = len(image_files)
                        image_paths.append(os.path.join(images_path))
                        if self.depth_data:
                            depth_paths.append(os.path.join(depth_path))

                        image_times.append(float(img_index / len(image_files)))
                        image_poses.append(image_pose_dict)

                        this_count += 1

                if dir == "cam08":  # north 1 - this is not used througout the dataset but can be extended to new agents 
                    N_cams += 1
                    image_folders = os.path.join(root, dir)
                    image_files = sorted(os.listdir(image_folders))
                    this_count = 0
                    # image_files = image_files[:20] #hardcode to take only first 20 samples
                    for img_file in image_files:
                        if this_count >= countss: break
                        img_index = image_files.index(img_file)
                        images_path = os.path.join(image_folders, img_file)
                        if self.depth_data:
                            depth_folder = os.path.join(root, "cam01_depth")
                            depth_files = sorted(os.listdir(depth_folder))
                            depth_files = depth_files[:20]
                            assert len(depth_files) == len(image_files)
                            for depth_file in depth_files:
                                depth_path = os.path.join(depth_folder, depth_file)

                        intrinsic_south_1 = np.asarray([[1400.3096617691212, 0.0, 967.7899705163408],
                                                        [0.0, 1403.041082755918, 581.7195041357244],
                                                        [0.0, 0.0, 1.0]], dtype=np.float32)
                        northintrinsics = np.asarray([[1315.158203125, 0.0, 962.7348338975571],
                                                      [0.0, 1362.7757568359375, 580.6482296623581],
                                                      [0.0, 0.0, 1.0]], dtype=np.float32)
                        """
                        Using extrinsics matrix from TUMTRAF Repo
                        """

                        extrinsic_south_1 = np.asarray([[0.41204962, -0.45377758, 0.7901276, 2.158825],
                                                        [-0.9107832, -0.23010845, 0.34281868, -15.5765505],
                                                        [0.02625162, -0.86089253, -0.5081085, 0.08758777],
                                                        [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

                        extrinsic_south_1_calib = np.asarray([[0.95302056, - 0.30261307, 0.01330958, 1.77326515],
                                                              [-0.12917788, - 0.44577866, - 0.88577337, 7.60903957],
                                                              [0.27397972, 0.84244093, - 0.46392715, 4.04778098],
                                                              [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
                        transformation_matrix_base_to_camera_south_1 = np.array([
                            [0.891382638626301, 0.37756862104528707, -0.07884507325924934, 25.921784677055939],
                            [0.2980421080238165, -0.6831891949380544, -0.6660273169946723, 13.668310799382738],
                            [-0.24839844089507856, 0.5907739097931769, -0.7525203649548087, 18.630430017833277],
                            [0, 0, 0, 1]], dtype=float)
                        transformation_matrix_lidar_to_base_south_1 = np.array([
                            [0.247006, -0.955779, -0.15961, -16.8017],
                            [0.912112, 0.173713, 0.371316, 4.66979],
                            [-0.327169, -0.237299, 0.914685, 6.4602],
                            [0.0, 0.0, 0.0, 1.0], ], dtype=float)

                        north2infralidar = np.asarray([[-0.56460226, -0.4583457, 0.6863989, 0.64204305],
                                                       [-0.8248329, 0.34314296, -0.4493365, -16.182753],
                                                       [-0.02958117, -0.81986094, -0.57179797, 1.6824605],
                                                       [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
                        shata = np.linalg.inv(north2infralidar)

                        rotation_matrix = np.array([
                            [-0.29298163, 0.95605249, 0.01119853],
                            [0.87735771, 0.26417368, 0.40056923],
                            [0.38000685, 0.12718454, -0.91619806]
                        ])

                        # Define the new translation vector
                        translation_matrix = np.array([
                            [236.433],
                            [312.179],
                            [-987.74]
                        ])

                        # Combine rotation matrix and translation vector to create the extrinsic matrix
                        extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))
                        extrinsic_matrix_4x4 = np.vstack((extrinsic_matrix, [0, 0, 0, 1]))

                        extrinsic_matrix_lidar_to_camera_north_1 = np.matmul(
                            extrinsic_matrix_4x4,
                            transformation_matrix_lidar_to_base_south_1)
                        camera_to_lidar_extrinsics_north_1 = np.linalg.inv(extrinsic_matrix_lidar_to_camera_north_1)

                        south_1_proj = np.asarray(
                            [[1279.275240545117, -862.9254609474538, -443.6558546306608, -16164.33175985643],
                             [-57.00793327192514, -67.92432779187584, -1461.785310749125, -806.9258947569469],
                             [0.7901272773742676, 0.3428181111812592, -0.508108913898468, 3.678680419921875]],
                            dtype=np.float32)
                        infralidar2n1image = np.asarray(
                            [[-185.2891049687059, -1504.063395597006, -525.9215327879701, -23336.12843138125],
                             [-240.2665682659353, 220.6722195428702, -1567.287260600104, 6362.243306159624],
                             [0.6863989233970642, -0.4493367969989777, -0.5717979669570923, -6.750176429748535]],
                            dtype=np.float32)

                        # Given projection matrix
                        infralidar2n1image = np.array([
                            [-185.2891049687059, -1504.063395597006, -525.9215327879701, -23336.12843138125],
                            [-240.2665682659353, 220.6722195428702, -1567.287260600104, 6362.243306159624],
                            [0.6863989233970642, -0.4493367969989777, -0.5717979669570923, -6.750176429748535]
                        ], dtype=np.float32)

                        # Given intrinsic matrix
                        northintrinsics = np.array([
                            [1315.158203125, 0.0, 962.7348338975571],
                            [0.0, 1362.7757568359375, 580.6482296623581],
                            [0.0, 0.0, 1.0]
                        ], dtype=np.float32)

                        # Compute the inverse of the intrinsic matrix
                        K_inv = np.linalg.inv(northintrinsics)

                        # Extract the extrinsic matrix [R | t] by multiplying K^-1 with the projection matrix
                        extrinsic_matrix_north = K_inv @ infralidar2n1image
                        extrinsic_matrix_4x4_north = np.vstack((extrinsic_matrix_north, [0, 0, 0, 1]))
                        print(extrinsic_matrix_4x4_north)

                        # Extract the rotation matrix R and the translation vector t
                        R = extrinsic_matrix[:, :3]
                        t = extrinsic_matrix[:, 3]

                        noisy_pose = add_noise_to_pose(extrinsic_v, translation_noise_std=0.05, rotation_noise_std=0.01)

                        image_pose_dict = {
                            'intrinsic_matrix': northintrinsics,
                            'extrinsic_matrix': north2infralidar,  # north2infralidar, #camera_to_lidar_extrinsics_north_1, #
                            "projection_matrix": infralidar2n1image
                        }
                        N_time = len(image_files)
                        image_paths.append(os.path.join(images_path))
                        if self.depth_data:
                            depth_paths.append(os.path.join(depth_path))

                        image_times.append(float(img_index / len(image_files)))
                        image_poses.append(image_pose_dict)

                        this_count += 1

                # image_poses = reorder_extrinsic_matrices(image_poses)
        return image_paths, image_poses, image_times, depth_paths  # Use this for training gaussian spalatting on TUMTRAF image_poses,

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        img = img.resize(self.img_wh, Image.LANCZOS)
        if self.depth_data:
            depth_image = Image.open(self.depth_paths[index])
            depth_tensor = self.transform(depth_image).float()
            depth = (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min())
            # np.array(cv2.imread(self.depth_paths[index], cv2.IMREAD_UNCHANGED), dtype=np.float32))
        # depth = np.load(depth_path).astype(np.float32)
        else:
            depth = None

        img = self.transform(img)

        return img, self.image_poses[index], self.image_times[index]  # , depth

    def load_pose(self, index):
        pose = self.image_poses[index]  #
        extrinsic_matrix = pose['extrinsic_matrix']
        R = extrinsic_matrix[:3, :3]  # np.transpose(extrinsic_matrix[:3, :3])
        T = extrinsic_matrix[:3, 3]
        return R, T  # self.image_poses[index]

