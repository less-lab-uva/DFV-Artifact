import numpy as np
from tqdm import tqdm
import pytorch_ssim


def ssim(vae, images):
    decoded_images,_,_ = vae(images.view(images.shape[0],784))
    ssim_list = list()
    for i in tqdm(range(images.shape[0])):
        ssim_list.append(pytorch_ssim.ssim(images[i].view(1,1,28,28), decoded_images[i].view(1,1,28,28)).item())
    ssim_list = np.array(ssim_list)
    return ssim_list, ssim_list.mean()


def ssim_100(vae, images):
    all_ssim_list = list()
    for r in range(100):

        decoded_images,_,_ = vae(images.view(images.shape[0],784))
        ssim_list = list()
        for i in range(images.shape[0]):
            ssim_list.append(pytorch_ssim.ssim(images[i].view(1,1,28,28), decoded_images[i].view(1,1,28,28)).item())
        all_ssim_list.append(ssim_list)

    all_ssim_list = np.array(all_ssim_list)
    return all_ssim_list, all_ssim_list.mean()