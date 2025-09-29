# train/data_wrapper.py
import torch
from torch.utils.data import (
    DataLoader,
    IterableDataset,
    default_collate,
)
import webdataset as wd
import io
from PIL import Image
import pytorch_lightning as pl
import ast
from torchvision.transforms import (
    Compose,
    CenterCrop,
    Resize,
    ToTensor,
    Normalize,
    InterpolationMode,
)
from diffusers.image_processor import VaeImageProcessor


class WebDatasetWrapper(IterableDataset):
    def __init__(self, tar_path, decode_fn, shuffle=False, resampled=False):
        super().__init__()
        self.tar_path = tar_path
        self.shuffle = shuffle
        self.resampled = resampled
        self.decode_fn = decode_fn
        self.dataset = None # lazy initialize
        if self.decode_fn is None:
            raise ValueError("decode_fn must be provided")

    def _init_dataset(self):
        dataset = wd.WebDataset(
            self.tar_path,
            nodesplitter=wd.split_by_node,
            workersplitter=wd.split_by_worker,
            resampled=self.resampled,
            handler=wd.handlers.warn_and_continue,
        )
        if self.shuffle:
            dataset = dataset.shuffle(1000)
        self.dataset = dataset

    def __iter__(self):
        if self.dataset is None:
            self._init_dataset()
        for sample in self.dataset:
            image = Image.open(io.BytesIO(sample.get("img")))
            det = self.decode_fn(sample.get("det.pb"))

            text_raw = sample.get("txt")
            if text_raw is None:
                text = ""
            elif isinstance(text_raw, bytes):
                text = text_raw.decode("utf-8").strip()
            else:
                text = str(text_raw).strip()
            yield {"image": image, "det": det, "text": text}


class WebDatasetDataModule(pl.LightningDataModule):
    def __init__(
        self, train_tar, val_tar, batch_size, num_workers=4, collate_fn=default_collate
    ):
        super().__init__()
        self.train_tar = train_tar
        self.val_tar = val_tar
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn

    def train_dataloader(self):
        dataset = WebDatasetWrapper(self.train_tar, decode_detection, shuffle=True)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        if self.val_tar is None:
            return None
        dataset = WebDatasetWrapper(self.val_tar, decode_detection, shuffle=False)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )


def decode_detection(payload: bytes) -> dict:
    text = payload.decode("utf-8").strip()
    if not text:
        return {"labels": [], "scores": [], "boxes": []}
    labels, scores, boxes = [], [], []
    for line in text.split("\n"):
        parts = line.split("|", 2)
        if len(parts) != 3:
            raise ValueError(f"Malformed detection line: {line}")
        label, score_str, box_str = parts
        labels.append(label)
        scores.append(float(score_str))
        boxes.append(ast.literal_eval(box_str))
    return {"labels": labels, "scores": scores, "boxes": boxes}


def _custom_collate_fn(batch):
    images = [item["image"] for item in batch]
    dets = [item["det"] for item in batch]
    texts = [item["text"] for item in batch]
    return {"images": images, "dets": dets, "texts": texts}


def get_datamodule(tar_path, batch_size, num_workers=4):
    return WebDatasetDataModule(
        tar_path,
        None,
        batch_size,
        num_workers=num_workers,
        collate_fn=_custom_collate_fn,
    )

_vae_processor = None
def vae_preprocess(images):
    global _vae_processor
    if _vae_processor is None:
        _vae_processor = VaeImageProcessor(
            do_binarize=False,
            do_convert_grayscale=False,
            do_convert_rgb=False,
            do_normalize=True,
            do_resize=True,
            reducing_gap=None,
            resample="lanczos",
            vae_latent_channels=4,
            vae_scale_factor=8,
        )
    return _vae_processor.preprocess(images)


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def clip_preprocess(images, npx=224):
    transform = _transform(npx)
    if isinstance(images, list):
        return torch.stack(tensors=[transform(image) for image in images], dim=0)
    else:
        return transform(images).unsqueeze(0)
