#  Copyright (c) 2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.
import torch

from .protocol import DatasetProtocol
import numpy as np
from torch.utils.data import Dataset
from abc import ABC
from torchvision.datasets.utils import verify_str_arg, download_and_extract_archive
import os
import os.path
import pathlib
from PIL import Image
import pydicom
from utils import ConfigureHelper, ContainerHelper
from utils import ImageHelper
import numpy.typing as npt
from enum import Enum
from utils.typing import TypePathLike, TypeNPDTypeFloat, TypeNPDTypeUnsigned
from utils import MultiProcessingHelper
import skimage


class BaseDataset(Dataset, ABC, DatasetProtocol):

    def __init__(
            self,
            data_root: TypePathLike, split: str, ret_size: int | tuple[int, int],
            preload_dataset: bool, n_preload_worker=ConfigureHelper.max_n_worker, debug=False):
        super(BaseDataset, self).__init__()

        self._split = verify_str_arg(split, "split", ("trainval", "test"))
        self._images_folder = pathlib.Path(data_root) / "rawjpg"
        self._segs_folder = pathlib.Path(data_root) / "seg"
        self._preload_dataset = preload_dataset
        self._preloaded = False
        self._image_load_func = self._load_jpg
        self._seg_load_func = self._load_jpg
        self._ret_size = ContainerHelper.to_tuple(ret_size)

        self._image_ids = self.read_image_ids(self._images_folder)
        self._image_id_idx_dict = {image_id: i for i, image_id in enumerate(self._image_ids)}

        self._images = [str(self._images_folder / f"{image_id}.jpg") for image_id in self._image_ids]
        self._segs = [str(self._segs_folder / f"{image_id}.jpg") for image_id in self._image_ids]

        if debug:
            self._images = self._images[:10]
            self._segs = self._segs[: 10]
            del self._image_id_idx_dict
            print("Running in debug mode.")

        if self._preload_dataset:
            mph = MultiProcessingHelper()
            self._images = mph.run(
                args=[(image,) for image in self._images], func=self._load_jpg, n_worker=n_preload_worker,
                desc=f"Preloading {split} images", mininterval=10, maxinterval=60)
            self._segs = mph.run(
                args=[(seg,) for seg in self._segs], func=self._load_jpg, n_worker=n_preload_worker,
                desc=f"Preloading {split} labels", mininterval=10, maxinterval=60)
            self._image_load_func = self._identity
            self._seg_load_func = self._identity

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> dict[str, npt.NDArray[TypeNPDTypeFloat]]:
        image_id = self._image_ids[idx]
        image = self._image_load_func(self._images[idx])
        seg = self._seg_load_func(self._segs[idx])
        image, seg = self._augment(image=image, seg=seg)

        image = ImageHelper.resize(image, self._ret_size,order=1)/255
        masked_image = ImageHelper.resize(seg, self._ret_size,order=0)/255

        image = image.clip(0., 1.)
        masked_image = masked_image.clip(0., 1.)
        image = np.expand_dims(image, axis=0).astype(np.float32)
        masked_image = np.expand_dims(masked_image, axis=0).astype(np.float32)

        # image = image.transpose((2, 0, 1)).astype(np.float32)
        # masked_image = masked_image.transpose((2, 0, 1)).astype(np.float32)
        ret = {"image_id": image_id, "image": torch.from_numpy(image), "mask": torch.from_numpy(masked_image)}
        return ret

    def get_item_by_image_id(self, image_id: str):
        idx = self._image_id_idx_dict[image_id]
        return self[idx]

    @classmethod
    def _load_dcm(cls, path) -> np.ndarray:
        dicom_data = pydicom.dcmread(path)
        return dicom_data.pixel_array

    @staticmethod
    def _load_jpg(path) -> np.ndarray:
        return np.array(Image.open(path).convert('L'))  # 1, 2, 3 (pet, background, boarder)

    @staticmethod
    def read_image_ids(path) -> list[str]:
        filenames = os.listdir(path)
        image_ids = [os.path.splitext(filename)[0] for filename in filenames]
        return image_ids

    @staticmethod
    def _identity(x):
        return x

    def _augment(self, image: npt.NDArray[TypeNPDTypeFloat], seg: npt.NDArray[TypeNPDTypeUnsigned]) \
            -> tuple[npt.NDArray[TypeNPDTypeFloat], npt.NDArray[TypeNPDTypeUnsigned]]:
        return image, seg

