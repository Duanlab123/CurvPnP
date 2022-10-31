import os.path
import logging

import numpy as np
from collections import OrderedDict

import torch

from utils import utils_logger
from utils import utils_image as util

"""
# --------------------------------------------
|--model_zoo               # model_zoo
   |--estnoise_color
   |--denoise_color
|--testsets                # testsets
   |--Kodak24              # testset_name
   |--Urban100
   |--PIPAL
   |--CC
   |--PolyU
# --------------------------------------------
"""

def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    noise_level_img = 50               # default: 0, noise level for LR image
    modele_name = 'estnoise_color'     # 'estnoise_color'
    modeld_name = 'denoise_color'      # 'denoise_color'
    testset_name = 'Kodak24'           # test set,  'Kodak24' | 'Urban100' | 'PIPAL' | 'CC' | 'PolyU'
    border = 0                         # shave border to calculate PSNR and SSIM

    # --------------------------------
    # load kernel
    # --------------------------------

    task_current = 'denoise'             # 'denoise' for denoising
    n_channels = 3 if 'color' in  modeld_name else 1  # fixed
    model_zoo = 'model_zoo'              # fixed
    testsets = 'testsets'                # fixed
    results = 'results'                  # fixed
    result_name = testset_name + '_' + task_current + '_' + modeld_name
    modele_path = os.path.join(model_zoo, modele_name+'.pth')
    modeld_path = os.path.join(model_zoo, modeld_name+'.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = os.path.join(testsets, testset_name) # L_path, for Low-quality images
    E_path = os.path.join(results, result_name)   # E_path, for Estimated images
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    # ----------------------------------------
    # load model
    # ----------------------------------------
    
    from models.network_estnunet import EstNUNet as enet
    modele = enet(in_nc=n_channels, out_nc=n_channels, nc=[96, 192, 384, 768], nb=2)
    modele.load_state_dict(torch.load(modele_path), strict=True)
    modele.eval()
    for _, v in modele.named_parameters():
        v.requires_grad = False
    modele = modele.to(device)

    from models.network_dnunet import DNUNet as dnet
    modeld = dnet(in_nc=n_channels, out_nc=n_channels, nc=[96, 192, 384, 768], nb=4)
    modeld.load_state_dict(torch.load(modeld_path), strict=True)
    modeld.eval()
    for _, v in modeld.named_parameters():
        v.requires_grad = False
    modeld = modeld.to(device)

    logger.info('model_name:{}, image sigma:{:.3f}'.format(modeld_name, noise_level_img))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    for idx, img in enumerate(L_paths):

        # --------------------------------
        # img_L
        # --------------------------------

        img_name, ext = os.path.splitext(os.path.basename(img))
        img_H = util.imread_uint(img, n_channels=n_channels)
        img_H = util.modcrop(img_H, 8)  # modcrop
        img_L = util.uint2single(img_H)
        
        np.random.seed(seed=0)  # for reproducibility
        img_L += np.random.normal(0, noise_level_img/255., img_L.shape) # add AWGN
        img_L = util.single2tensor4(img_L).to(device)

        # --------------------------------
        # img_E
        # --------------------------------
        
        noise_level, noise_dis, img_C = modele(img_L)
        img_E = modeld(img_L, noise_level, img_C)

        img_E = util.tensor2uint(img_E)
        util.imsave(img_E, os.path.join(E_path, img_name+ext))

        # --------------------------------
        # PSNR and SSIM
        # --------------------------------
        
        if n_channels == 1:
            img_H = img_H.squeeze()
        psnr = util.calculate_psnr(img_E, img_H, border=border)
        ssim = util.calculate_ssim(img_E, img_H, border=border)
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)
        logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(img_name+ext, psnr, ssim))

    # --------------------------------
    # Average PSNR and SSIM
    # --------------------------------
    
    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    logger.info('Average PSNR/SSIM(RGB) - {} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, ave_psnr, ave_ssim))

if __name__ == '__main__':

    main()
