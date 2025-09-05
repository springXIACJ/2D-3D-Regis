#| export
from pathlib import Path
from typing import Optional, Union
import scipy
import h5py
import numpy as np
import torch
from beartype import beartype
from diffdrr.drr import DRR
import SimpleITK as sitk
from .calibration import RigidTransform, perspective_projection

@beartype
class FemurFluoroDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filename_origin,
        filename_mask,
        focal_len,delx,x0,y0
    ):
        (self.volume, self.spacing) = load_ct_dataset(filename_origin,filename_mask)
        self.focal_len=focal_len
        self.x0=x0
        self.y0=y0
        self.delx=delx
         # Get the isocenter pose (AP viewing angle at volume isocenter)
        isocenter_rot = torch.tensor([[-torch.pi / 2, 0.0, -torch.pi / 2]])
        isocenter_xyz = torch.tensor(self.volume.shape) * self.spacing / 2
        isocenter_xyz = isocenter_xyz.unsqueeze(0)
        self.isocenter_pose = RigidTransform(
            
            isocenter_rot, isocenter_xyz, "euler_angles", "ZYX"
        )
        
        
        
def resample(image, spacing, new_spacing=[1,1,1]):
    # .mhd image order : z, y, x
    if not isinstance(spacing, np.ndarray):
        spacing = np.array(spacing)
    if not isinstance(new_spacing, np.ndarray):
        new_spacing = np.array(new_spacing)
    # spacing = spacing[::-1]
    # new_spacing = new_spacing[::-1]
    
    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return image, new_spacing
        
def load_ct_dataset(filename_origin,filename_mask):
    
    ct=sitk.ReadImage(filename_origin)
    mask=sitk.ReadImage(filename_mask)
    ct_np=sitk.GetArrayFromImage(ct).astype(np.float32)
    mask_np=sitk.GetArrayFromImage(mask).astype(np.float32)
    ct_mask_np=ct_np.copy()
    ct_mask_np[:,:,:][mask_np[:,:,:]==0]=-1024
    volume_old = np.swapaxes(ct_mask_np, 0, 2)[::-1].copy()
    spacing=np.array(ct.GetSpacing())[:].flatten()
    ct_mask_volume,reshape_spacing=resample(volume_old,spacing)
    return (ct_mask_volume,reshape_spacing)

from torchvision.transforms import Compose, Lambda, Normalize, Resize
class Transforms:
    def __init__(
        self,
        size: int,  # Dimension to resize image
        eps: float = 1e-6,
    ):
        """Transform X-rays and DRRs before inputting to CNN."""
        self.transforms = Compose(
            [
                Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + eps)),
                Resize((size, size), antialias=True),
                #Normalize(mean=0.3080, std=0.1494), #为什么需要这个参数的Normalize
                Normalize(0,1),
            ]
        )

    def __call__(self, x):
        return self.transforms(x)
    


from .calibration import RigidTransform, convert
#测试在two-views条件下的训练效果
@beartype
def get_twoviews_random_offset(batch_size:int,device):
    r1 = torch.distributions.Normal(0, 0.2).sample((batch_size,))
    r2 = torch.distributions.Normal(0, 0.1).sample((batch_size,))
    r3 = torch.distributions.Normal(-0.15, 0.25).sample((batch_size,))
    t1 = torch.distributions.Normal(-100, 30).sample((batch_size,))
    t2 = torch.distributions.Normal(250, 90).sample((batch_size,))
    t3 = torch.distributions.Normal(-100, 30).sample((batch_size,))
    
    log_R_vee = torch.stack([r1, r2, r3], dim=1).to(device)
    log_t_vee = torch.stack([t1, t2, t3], dim=1).to(device)

    #只改变r3
    r31=torch.tensor([1.2]*batch_size)+r3
    log_R1_vee = torch.stack([r1, r2, r31], dim=1).to(device)
    return convert(
        [log_R_vee, log_t_vee],
        "se3_log_map",
        "se3_exp_map",
    ), convert(
        [log_R1_vee, log_t_vee],
        "se3_log_map",
        "se3_exp_map",
    )
@beartype
def get_random_offset(batch_size: int, device) -> RigidTransform:
    r1 = torch.distributions.Normal(0, 1.6).sample((batch_size,))
    r2 = torch.distributions.Normal(0, 1.6).sample((batch_size,))
    r3 = torch.distributions.Normal(0, 1.6).sample((batch_size,))
    t1 = torch.distributions.Normal(-200, 100).sample((batch_size,))
    t2 = torch.distributions.Normal(0, 150).sample((batch_size,))
    t3 = torch.distributions.Normal(0, 180).sample((batch_size,))
    log_R_vee = torch.stack([r1, r2, r3], dim=1).to(device)
    log_t_vee = torch.stack([t1, t2, t3], dim=1).to(device)
    return  RigidTransform(log_R_vee, log_t_vee, "euler_angles", "ZYX")
#def get_random_offset(batch_size: int, device) -> RigidTransform:
    # r1 = torch.distributions.Normal(0, 0.2).sample((batch_size,))
    # r2 = torch.distributions.Normal(0, 0.1).sample((batch_size,))
    # r3 = torch.distributions.Normal(0, 0.25).sample((batch_size,))
    # t1 = torch.distributions.Normal(10, 70).sample((batch_size,))
    # t2 = torch.distributions.Normal(250, 90).sample((batch_size,))
    # t3 = torch.distributions.Normal(5, 50).sample((batch_size,))
    # r1 = torch.distributions.Normal(0, 0.2).sample((batch_size,))
    # r2 = torch.distributions.Normal(0, 0.1).sample((batch_size,))
    # r3 = torch.distributions.Normal(0, 0.1).sample((batch_size,))
    # t1 = torch.distributions.Normal(10, 70).sample((batch_size,))
    # t2 = torch.distributions.Normal(10, 90).sample((batch_size,))
    # t3 = torch.distributions.Normal(20, 50).sample((batch_size,))
    
    # r1 = torch.distributions.Normal(0, 0.2).sample((batch_size,))
    # r2 = torch.distributions.Normal(0, 0.1).sample((batch_size,))
    # r3 = torch.distributions.Normal(0, 0.25).sample((batch_size,))
    # t1 = torch.distributions.Normal(-50, 50).sample((batch_size,))
    # t2 = torch.distributions.Normal(250, 20).sample((batch_size,))
    # t3 = torch.distributions.Normal(-50, 50).sample((batch_size,))

    # r1 = torch.distributions.Normal(0, 0.2).sample((batch_size,))
    # r2 = torch.distributions.Normal(0, 0.1).sample((batch_size,))
    # r3 = torch.distributions.Normal(0, 0.25).sample((batch_size,))
    # t1 = torch.distributions.Normal(-100, 50).sample((batch_size,))
    # t2 = torch.distributions.Normal(250, 90).sample((batch_size,))
    # t3 = torch.distributions.Normal(-100, 50).sample((batch_size,))
    
    # r1 = torch.distributions.Normal(0, 0.2).sample((batch_size,))
    # r2 = torch.distributions.Normal(0, 0.2).sample((batch_size,))
    # r3 = torch.distributions.Normal(0, 0.2).sample((batch_size,))
    # t1 = torch.distributions.Normal(0, 90).sample((batch_size,))
    # t2 = torch.distributions.Normal(0, 90).sample((batch_size,))
    # t3 = torch.distributions.Normal(0, 90).sample((batch_size,))
    # log_R_vee = torch.stack([r1, r2, r3], dim=1).to(device)
    # log_t_vee = torch.stack([t1, t2, t3], dim=1).to(device)
    # return convert(
    #     [log_R_vee, log_t_vee],
    #     "se3_log_map",
    #     "se3_exp_map",
    # )