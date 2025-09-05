#  Copyright (c) 2023. by Yi GU <gu.yi.gu4@naist.ac.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from torch.nn import DataParallel
import torch
from network.discriminator import MultiscaleDiscriminator
from network.loss.feature_matching_loss import FeatureMatchingLoss
from torch.optim import AdamW
from utils import EpochLossLogger
from network.loss.gradient_correlation_loss import GradientCorrelationLoss
from network.loss.ncc_loss import NormalizedCrossCorrelation
from network.hrnet import get_hrnet
from network.loss.gan_loss import GANLoss
from typing import Any
from utils.typing import TypePathLike
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from dataset import TrainingDataset, TestDataset, VisualizationDataset
from utils import ConfigureHelper
import pathlib
#import tomllib
import toml
import argparse


class DataModule:

    def __init__(
            self, perspect, image_size, train_data_root: TypePathLike, test_data_root: TypePathLike,
            training_batch_size: int | None = None, test_batch_size: int | None = None,
            preload=True, n_worker=ConfigureHelper.max_n_worker, debug=False):
        self._n_worker = n_worker
        train_data_root = pathlib.Path(train_data_root + perspect)
        test_data_root = pathlib.Path(test_data_root + perspect)

        if training_batch_size is None:
            self._training_batch_size = 1
        else:
            self._training_batch_size = training_batch_size

        if test_batch_size is None:
            self._test_batch_size = self._training_batch_size
        else:
            self._test_batch_size = test_batch_size

        self._training_dataset = TrainingDataset(
            data_root=train_data_root, preload=preload, image_size=image_size, n_worker=self._n_worker, debug=debug)
        self.training_data_loader = DataLoader(
            dataset=self._training_dataset, batch_size=self._training_batch_size,
            shuffle=True, num_workers=self._n_worker, pin_memory=True)

        self._test_dataset = TestDataset(
            data_root=test_data_root, preload=preload, image_size=image_size, n_worker=self._n_worker, debug=debug)
        self.test_data_loader = DataLoader(
            dataset=self._test_dataset, batch_size=self._test_batch_size,
            shuffle=False, num_workers=self._n_worker, pin_memory=True)

        self._visualization_dataset = VisualizationDataset(test_dataset=self._test_dataset)
        self.visualization_data_loader = DataLoader(
            dataset=self._visualization_dataset, batch_size=1,
            shuffle=False, num_workers=self._n_worker, pin_memory=True)


class Trainer:
    def __init__(
            self, data_module: DataModule, image_size, device: torch.device, output_dir: TypePathLike,
            n_epoch, model="hrnet48", lr=5e-5, weight_decay=1e-4,
            pretrain_load_dir: TypePathLike | None = None, pretrain_load_prefix: str = 'ckp',
            log_visual_every_n_epoch: int | None = None,
            log_model_hist_every_n_epoch: int | None = None,
            test_every_n_epoch: int | None = None,
            save_model_every_n_epoch: int | None = None):
        self._lr = lr
        self._n_epoch = n_epoch
        self._output_dir = pathlib.Path(output_dir)
        self._data_module = data_module
        self._image_size = image_size
        self._nc = 1
        self._log_visual_every_n_epoch = log_visual_every_n_epoch
        self._log_model_hist_every_n_epoch = log_model_hist_every_n_epoch
        self._save_model_every_n_epoch = save_model_every_n_epoch
        if pretrain_load_dir is None:
            self._pretrain_load_dir = pretrain_load_dir
        else:
            self._pretrain_load_dir = pathlib.Path(pretrain_load_dir)
        self._pretrain_load_prefix = pretrain_load_prefix
        self._test_every_n_epoch = test_every_n_epoch

        self._device = device
        self._netG = get_hrnet(model, self._nc, self._nc).to(self._device)
        total_params = sum(p.numel() for p in self._netG.parameters())
        print(f"{self._netG.__class__} has {total_params * 1.e-6:.2f} M params.")

        n_disc = 2
        n_layer = 3
        self._netD = MultiscaleDiscriminator(
            input_nc=self._nc, num_D=n_disc, n_layer=n_layer).to(self._device)
        total_params = sum(p.numel() for p in self._netD.parameters())
        print(f"{self._netD.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")

        if self._pretrain_load_dir is not None:
            self.load_model(self._pretrain_load_dir, self._pretrain_load_prefix)

        self._criterion_l1 = torch.nn.L1Loss(reduction="mean")
        self._criterion_fm = FeatureMatchingLoss(n_disc=n_disc, n_layer=n_layer)
        self._criterion_gan = GANLoss().to(self._device)
        self._criterion_gc = GradientCorrelationLoss(n_channels=self._nc).to(self._device)
        self._criterion_ncc = NormalizedCrossCorrelation(n_channels=self._nc).to(self._device)

        self._G_optimizer = AdamW(params=self._netG.parameters(), lr=self._lr, weight_decay=weight_decay)
        self._D_optimizer = AdamW(params=self._netD.parameters(), lr=self._lr, weight_decay=weight_decay)
        self._G_grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
        self._D_grad_scaler = torch.cuda.amp.GradScaler(enabled=True)

        tb_log_dir = self._output_dir / "tb_log"
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        self._tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
        self._tb_writer.add_graph(self._netG, torch.randn(1, 1, *image_size, device=self._device))

    def train(self):
        for epoch in range(1, self._n_epoch + 1):
            print(f"Training epoch {epoch}. LR={self._lr}.")
            self.train_epoch(epoch)

            if self._log_visual_every_n_epoch is not None:
                if epoch % self._log_visual_every_n_epoch == 0:
                    print(f"Log visualization epoch {epoch}.")
                    self.visual_epoch(epoch)

            if self._log_model_hist_every_n_epoch is not None:
                if epoch % self._log_model_hist_every_n_epoch == 0:
                    print(f"Log model histogram epoch {epoch}.")
                    self.visual_histogram(epoch)

            if self._test_every_n_epoch is not None:
                if epoch % self._test_every_n_epoch == 0:
                    print(f"Testing epoch {epoch}.")

                    self.eval_epoch(epoch)

            if self._save_model_every_n_epoch is not None:
                if epoch % self._save_model_every_n_epoch == 0:
                    self.save_model(self._output_dir, prefix=str(epoch))

        print("Training finished.")
        self.save_model(self._output_dir, prefix=str(self._n_epoch))

    def train_epoch(self, epoch):
        self._tb_writer.add_scalar(f"lr", self._lr, global_step=epoch)

        epoch_loss_logger = EpochLossLogger(device=self._device)
        self.trigger_model(train=True)
        iterator = self._data_module.training_data_loader
        n_iter = len(iterator)
        for i, data in enumerate(tqdm(iterator, desc="Training", mininterval=60, total=n_iter)):
            self.train_batch(data, epoch_loss_logger)

        epoch_loss_log: dict[str, torch.Tensor] = epoch_loss_logger.summary()
        msg = "Loss "
        for k, v in epoch_loss_log.items():
            v = v.item()
            msg += "%s: %.3f " % (k, v)
            self._tb_writer.add_scalar(f"train/{k}", scalar_value=v, global_step=epoch)
        print(msg)

    @torch.no_grad()
    def visual_epoch(self, epoch):
        assert self._data_module.visualization_data_loader.batch_size == 1
        for data in self._data_module.visualization_data_loader:
            image = data["image"].to(self._device)  # (B, C, H, W)
            mask = data["mask"].to(self._device)
            pred_mask = self._netG(image)

            tag_image_dict = {"image": image, "mask": mask, "pred_mask": pred_mask}

            for tag, img in tag_image_dict.items():
                # denormalize -> (0, 1) -> (0, 255)
                img=img.clamp(0., 1.)
                img = (img * 255.).to(torch.uint8)
                img = torch.permute(img, (0, 2, 3, 1))  # (B, H, W, C)
                img = img[0]  # (H, W, C)
                self._tb_writer.add_image(tag=f"e{epoch}_{tag}", img_tensor=img, dataformats="HWC")
            break

    @torch.no_grad()
    def eval_epoch(self, epoch):
        total_count = 0.
        psnr = torch.tensor([0.], device=self._device)
        ssim = torch.tensor([0.], device=self._device)
        calc_psnr = PeakSignalNoiseRatio(reduction=None, dim=(1, 2, 3), data_range=1.)
        calc_ssim = StructuralSimilarityIndexMeasure(reduction=None, data_range=1.)

        iterator = self._data_module.test_data_loader
        n_iter = len(iterator)
        for data in tqdm(iterator, desc="Testing", mininterval=30, total=n_iter, maxinterval=60):
            image = data["image"].to(self._device)  # (B, C, H, W)
            mask = data["mask"].to(self._device)
            pred_mask = self._netG(image)

            # denormalize -> (0, 1)
            mask = mask .clamp(0., 1.)
            pred_mask = pred_mask .clamp(0., 1.)

            psnr += calc_psnr(pred_mask, mask).sum()
            ssim += calc_ssim(pred_mask, mask).sum()
            total_count += mask.shape[0]

        psnr /= total_count
        ssim /= total_count

        metric_eval_dict = {"psnr": psnr, "ssim": ssim}
        msg = f"Test epoch {epoch}"
        for k, v in metric_eval_dict.items():
            v = v.item()
            msg += "%s: %.3f " % (k, v)
            self._tb_writer.add_scalar(f"test/{k}", scalar_value=v, global_step=epoch)
        print(msg)

    @torch.no_grad()
    def visual_histogram(self, epoch):
        for name, param in self._netG.named_parameters():
            self._tb_writer.add_histogram(name, param.cpu().numpy(), epoch)

    def train_batch(self, data: dict, epoch_loss_logger: EpochLossLogger):
        g_loss, d_loss = self._compute_loss(data=data, epoch_loss_logger=epoch_loss_logger)
        self._G_optimizer.zero_grad()
        self._G_grad_scaler.scale(g_loss).backward()
        self._G_grad_scaler.step(self._G_optimizer)
        self._G_grad_scaler.update()

        self._D_optimizer.zero_grad()
        self._D_grad_scaler.scale(d_loss).backward()
        self._D_grad_scaler.step(self._D_optimizer)
        self._D_grad_scaler.update()

    def _compute_loss(self, data: dict[str, Any], epoch_loss_logger: EpochLossLogger):
        g_loss_total = torch.tensor([0.], device=self._device)

        drr = data["mask"]
        fake_drr = self._netG(data["image"].to(self._device))

        drr = drr.to(self._device)
        l1_loss = self._criterion_l1(fake_drr, drr) # MAE 平均绝对误差
        epoch_loss_logger.log("G_L1", l1_loss)
        g_loss_total = g_loss_total + l1_loss * 100.

        ncc_loss = self._criterion_ncc(fake_drr, drr) # 梯度归一化互相关
        epoch_loss_logger.log("G_NCC", ncc_loss)
        g_loss_total = g_loss_total + ncc_loss * 10

        d_pred_fake = self._netD(fake_drr)
        d_pred_real = self._netD(drr)

        fm_loss = self._criterion_fm(pred_fake=d_pred_fake, pred_real=d_pred_real) # 特征匹配(是否能骗过判别器，输出L1）
        epoch_loss_logger.log("G_FM", fm_loss)
        g_loss_total = g_loss_total + fm_loss * 1.

        gan_loss = self._criterion_gan(d_pred_fake, target_is_real=True) # 越小，判别器判别为真,MSE
        epoch_loss_logger.log("G_GAN", gan_loss)
        g_loss_total = g_loss_total + gan_loss * 0.1

        d_loss_total = torch.tensor([0.], device=self._device)
        d_pred_fake = self._netD(fake_drr.detach())
        d_loss_fake = self._criterion_gan(d_pred_fake, target_is_real=False)
        d_loss_real = self._criterion_gan(d_pred_real, target_is_real=True)
        d_loss = (d_loss_fake + d_loss_real) / 2.
        epoch_loss_logger.log("D_GAN", d_loss)
        d_loss_total = d_loss_total + d_loss

        return g_loss_total, d_loss_total

    def load_model(self, load_dir: TypePathLike, prefix="ckp") -> None:
        load_dir = pathlib.Path(load_dir)
        load_path = load_dir / f"{prefix}_netG.pt"
        self._netG.load_state_dict(torch.load(load_path, map_location='cpu'))
        print(f"Model netG weights loaded from {load_path}.")

        load_path = load_dir / f"{prefix}_netD.pt"
        self._netD.load_state_dict(torch.load(load_path, map_location='cpu'))
        print(f"Model netD weights loaded from {load_path}.")

    def save_model(self, save_dir: TypePathLike, prefix="ckp") -> None:
        save_dir = pathlib.Path(save_dir)
        save_path = save_dir / f"{prefix}_netG.pt"
        torch.save(self._netG.state_dict(), save_path)
        print(f"Model netG saved to {save_path}.")

        save_path = save_dir / f"{prefix}_netD.pt"
        torch.save(self._netD.state_dict(), save_path)
        print(f"Model netD saved to {save_path}.")

    def trigger_model(self, train=True):
        self._netG.train(train)
        self._netD.train(train)


def main():
    assert torch.cuda.is_available(), "GPU is not available."
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config_path", type=str, default='config/train.toml')
    parser.add_argument("--perspect", type=str, default='ML_Tibial')
    parser.add_argument("--output_data_root", type=str, default='output/ML_Tibial_nccloss_3_15')
    parser.add_argument("--device",  default='cuda')
    parser.add_argument("--n_worker", type=int, default=ConfigureHelper.max_n_worker)
    opt = parser.parse_args()

    config_path = pathlib.Path(opt.config_path)
    device = torch.device(opt.device)

    n_worker = opt.n_worker

    assert config_path.exists(), config_path
    with config_path.open(mode='r') as f:
        #config = tomllib.load(f)
        config=toml.load(f)
    print(f"Config loaded from {config_path}.")
    print(config)

    output_dir = pathlib.Path(opt.output_data_root)
    output_dir = output_dir.resolve()

    image_size = config["image_size"]
    data_moudle_config = config["data_module_config"]
    trainer_config = config["trainer_config"]

    print("Configuring data module")
    data_module = DataModule(perspect = opt.perspect, image_size=image_size, n_worker=n_worker, **data_moudle_config)
    print("Configuring trainer")
    trainer = Trainer(
        output_dir=output_dir, image_size=image_size, data_module=data_module, device=device, **trainer_config)

    print("Start training.")
    trainer.train()
    pass


if __name__ == '__main__':
    main()