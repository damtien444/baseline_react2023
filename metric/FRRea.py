from torchmetrics.image.fid import FrechetInceptionDistance as FID
import os
from torchvision.io import read_image
import torch


def compute_fid(output_dir, device):
    
    fid_dir = output_dir + "/fid"
    real_dir = fid_dir + "/real"
    fake_dir = fid_dir + "/fake"
    
    assert os.path.exists(real_dir), "real_dir does not exist"
    assert os.path.exists(fake_dir), "fake_dir does not exist"
    assert len(os.listdir(real_dir)) == len(os.listdir(fake_dir)), "real and fake dir have different number of files"
    
    reals = []
    fakes = []
    
    for file in os.listdir(real_dir):
        
        real = read_image(real_dir + "/" + file)
        fake = read_image(fake_dir + "/" + file)
        
        reals.append(real)
        fakes.append(fake)
        
    real_tensor = torch.stack(reals)
    fake_tensor = torch.stack(fakes)
    
    real_tensor = real_tensor.to(device)
    fake_tensor = fake_tensor.to(device)
    
    fid = FID(normalize=True)
    fid = fid.to(device)
    
    fid.update(fake_tensor, real=False)
    fid.update(real_tensor, real=True)

    val = fid.compute()
    return val