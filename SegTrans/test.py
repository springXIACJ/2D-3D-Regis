#  Copyright (c) 2023. by Yi GU <gu.yi.gu4@naist.ac.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from network.hrnet import get_hrnet
from utils.typing import TypePathLike
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics import Dice
from dataset import TestDataset
from utils import ConfigureHelper
import pathlib
import toml
import argparse
import pandas as pd
import numpy as np
from PIL import Image


class DataModule:

    def __init__(
            self, image_size, perspect,
            data_root: TypePathLike,
            test_batch_size: int | None = None,
            preload=True, n_worker=ConfigureHelper.max_n_worker, debug=False):
        self._n_worker = n_worker
        data_root = pathlib.Path(data_root + perspect)

        if test_batch_size is None:
            self._test_batch_size = 1
        else:
            self._test_batch_size = test_batch_size

        self._test_dataset = TestDataset(
            data_root=data_root, preload=preload, image_size=image_size, n_worker=self._n_worker, debug=debug)
        self.test_data_loader = DataLoader(
            dataset=self._test_dataset, batch_size=self._test_batch_size,
            shuffle=False, num_workers=self._n_worker, pin_memory=True)

class ModelTester:

    def __init__(
            self, perspect, data_module: DataModule, device: torch.device, output_dir: TypePathLike,
            pretrain_load_dir: TypePathLike, pretrain_load_prefix: str = 'ckp', model="hrnet48",):
        self.perspect = perspect
        self._data_module = data_module
        self._output_dir = pathlib.Path(output_dir)
        self._device = device
        #self._pretrain_load_dir = pathlib.Path(pretrain_load_dir + perspect)
        self._pretrain_load_dir = pathlib.Path(pretrain_load_dir)
        self._pretrain_load_prefix = pretrain_load_prefix

        self._nc = 1
        self._netG = get_hrnet(model, self._nc, self._nc).to(self._device)
        total_params = sum(p.numel() for p in self._netG.parameters())
        print(f"{self._netG.__class__} has {total_params * 1.e-6:.2f} M params.")

        self.load_model(self._pretrain_load_dir, self._pretrain_load_prefix)

    @torch.no_grad()
    def test(self):

        result = []
        calc_psnr = PeakSignalNoiseRatio(reduction=None, dim=(2, 3), data_range=1.)
        calc_ssim = StructuralSimilarityIndexMeasure(reduction=None, data_range=1.)
        calc_dice = Dice(average='micro')
        iterator = self._data_module.test_data_loader
        n_iter = len(iterator)
        for data in tqdm(iterator, desc="Testing", mininterval=30, total=n_iter, maxinterval=60):
            batch_image_id = data["image_id"]
            batch_mask = data["mask"].to(self._device)
            batch_pred_pet_image = self._netG(data["image"].to(self._device))

            # denormalize -> (0, 1)
            normed_batch_mask = batch_mask.clamp(0., 1.)
            normed_batch_pred_mask = batch_pred_pet_image .clamp(0., 1.)
            mask_binary = torch.where(normed_batch_mask < 0.005, torch.tensor([0]), torch.tensor([1]))
            pred_mask_binary = torch.where(normed_batch_pred_mask < 0.005, torch.tensor([0]), torch.tensor([1]))
            dice = calc_dice(mask_binary,pred_mask_binary)
            psnr = calc_psnr(normed_batch_pred_mask, normed_batch_mask)
            ssim = calc_ssim(normed_batch_pred_mask, normed_batch_mask)

            n_batch = batch_mask.shape[0]
            for i in range(n_batch):
                image_id = batch_image_id[i]
                data = {"image_id": image_id, "dice": dice.item(),"psnr": psnr.item(), "ssim": ssim.item()}
                result.append(data)
                img = (normed_batch_pred_mask * 255.).to(torch.uint8)
                img = torch.permute(img, (0, 2, 3, 1))  # (B, H, W, C)
                img = img[i].numpy()  # (H, W, C)
                img = np.squeeze(img)
                image = Image.fromarray(img, mode='L')
                image.save(self._output_dir / 'predicted'/ (image_id + '.jpg'))

        result = pd.DataFrame(result)
        result.sort_values(by='image_id', inplace=True)
        result.reset_index(drop=True, inplace=True)
        result.to_csv(self._output_dir / "test_result.csv")

    def load_model(self, load_dir: TypePathLike, prefix="ckp") -> None:
        load_dir = pathlib.Path(load_dir)
        load_path = load_dir / f"{prefix}_netG.pt"
        self._netG.load_state_dict(torch.load(load_path, map_location='cpu'))
        print(f"Model netG weights loaded from {load_path}.")


def main():
    assert torch.cuda.is_available(), "GPU is not available."
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config_path", type=str, default='config/test.toml')
    parser.add_argument("--perspect", type=str, default='ML_Tibial')
    parser.add_argument("--device",default='cpu')
    parser.add_argument("--n_worker", type=int, default=ConfigureHelper.max_n_worker)
    opt = parser.parse_args()

    config_path = pathlib.Path(opt.config_path)
    device = torch.device(opt.device)
    n_worker = opt.n_worker

    assert config_path.exists(), config_path
    with config_path.open(mode='r') as f:
        config = toml.load(f)
    print(f"Config loaded from {config_path}.")
    print(config)

    output_dir = pathlib.Path(config["output_data_root"]+opt.perspect)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = config["image_size"]
    data_moudle_config = config["data_module_config"]
    tester_config = config["tester_config"]

    print("Configuring data module.")
    data_module = DataModule(image_size=image_size, perspect=opt.perspect, **data_moudle_config)
    print("Configuring tester.")
    tester = ModelTester(perspect=opt.perspect, output_dir=output_dir, data_module=data_module, device=device, **tester_config)

    print("Start testing.")
    tester.test()


if __name__ == '__main__':
    main()