from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov


class V2XGOTR(Dataset):
    def __init__(
        self,
        dataset,
        args,
        dataset_type
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type=dataset_type


    def __getitem__(self, index):

        if self.dataset_type != "PanopticSports":

             try:
                image, w2c, time = self.dataset[index]
                #store = self.dataset[index]
                extrinsic_matrix = w2c['extrinsic_matrix']
                cam_id = w2c['cam_id']
                R = extrinsic_matrix[:3, :3]  # .transpose()
                T = extrinsic_matrix[:3, 3]
                #R,T = w2c
                FovX = focal2fov(self.dataset.focal[0], image.shape[2]) * 1.9
                FovY = focal2fov(self.dataset.focal[0], image.shape[1]) * 1.9
                #print("Training Parameters Fovx, y", FovX, FovY )
                mask=None
                F = w2c['intrinsic_matrix']
                #print("Focal values that is being used to train properly", F[0, 0], F[1, 1])
                depth = None

             except:
                #print("Wrong loop for loading Data")
                caminfo = self.dataset[index]
                #print("DATA Index ", len(self.dataset))
                image = caminfo.image
                R = caminfo.R
                T = caminfo.T
                FovX = caminfo.FovX * 1.9
                FovY = caminfo.FovY * 1.9
                #print("Rendering Parameters Fovx, y", FovX, FovY)
                time = caminfo.time
                depth = None #caminfo.depth #
    
                mask = caminfo.mask
                F = np.eye(4)#
                cam_id = w2c['cam_id']
                #flow = caminfo.flow
                #F=np.eye(4)
                #F [:3, :3] =
             return Camera(colmap_id=cam_id,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
                              image_name=f"{index}",uid=cam_id,data_device=torch.device("cuda"),time=time,
                              mask=mask, F=F, depth=depth)

        else:
            print("Warning")
            return self.dataset[index]
    def __len__(self):
        
        return len(self.dataset)
