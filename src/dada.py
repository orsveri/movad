import io
import os
import json
import numpy as np
import torch
import platform
import zipfile
import pandas as pd
from tqdm import tqdm
from natsort import natsorted

import random
from random import randint
from copy import deepcopy
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pytorchvideo import transforms as T

from torch.utils.data import DataLoader
from data_transform import pad_frames, RandomVerticalFlip, RandomHorizontalFlip
from orsveri_src.data_utils import smooth_labels, compute_time_vector
from orsveri_src.sequencing import RegularSequencer, FullSequencer, UnsafeOverlapSequencer


def read_file(path):
    return np.asarray(Image.open(path))


def has_objects(ann):
    return sum([len(labels['objects']) for labels in ann['labels']]) != 0


def gt_cls_target(curtime_batch, toa_batch, tea_batch):
    return (
        (toa_batch >= 0) &
        (curtime_batch >= toa_batch) & (
            (curtime_batch < tea_batch) |
            # case when sub batch end with a positive frame
            (toa_batch == tea_batch)
        )
    )


class ShortSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_samples_per_epoch, shuffle=True, **kwargs):
        """
        A custom Sampler to limit the number of samples per epoch.

        Args:
            dataset (Dataset): The dataset to sample from.
            num_samples_per_epoch (int): Number of samples per epoch.
            shuffle (bool): Whether to shuffle indices each epoch.
            seed (int): Random seed for reproducibility.
        """
        super().__init__(dataset, **kwargs)
        self.dataset = dataset
        self.num_samples_per_epoch = num_samples_per_epoch
        self.total_size = min(len(dataset), num_samples_per_epoch)
        self.shuffle = shuffle
        self.epoch = None
        #self.set_epoch(0)

    def _generate_indices(self):
        """Generate new random indices for the next epoch using PyTorch's global seed state."""
        if self.shuffle:
            # Use PyTorch's global random generator (no need to reset the seed)
            indices = torch.randperm(len(self.dataset)).numpy()  # Randomly shuffled indices
        else:
            indices = np.arange(len(self.dataset))  # Sequential indices if shuffle=False

        self.indices = indices[:self.num_samples_per_epoch]  # Select subset

    def set_epoch(self, epoch):
        """Allows manual setting of epoch (useful for multi-GPU training)."""
        self.epoch = epoch
        self._generate_indices()  # Refresh indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        """Number of samples per epoch."""
        return self.total_size


class AnomalySubBatch(object):
    def __init__(self, dada, index):
        key = dada.keys[index]
        num_frames = dada.metadata[key]['num_frames']
        self.begin, self.end = dada._get_random_subbatch(num_frames)
        # negative case
        if self.end >= dada.metadata[key]['anomaly_start'] and \
                self.begin <= dada.metadata[key]['anomaly_end']:
            self.label = 1
            self.a_start = max(
                0, dada.metadata[key]['anomaly_start'] - self.begin
            )
            self.a_end = min(
                dada.metadata[key]['anomaly_end'] - self.begin,
                self.end - self.begin
            )
        else:
            self.label = -1
            self.a_start = -1
            self.a_end = -1


class Dada(Dataset):
    ego_categories = [str(cat) for cat in list(range(1, 19)) + [61, 62]]

    def __init__(
            self, root_path, phase,
            transforms={'image': None},
            VCL=None,
            vertical_flip_prob=0., horizontal_flip_prob=0.,
            split_file=None):
        self.root_path = root_path
        self.phase = phase  # 'train', 'test', 'play'
        self.transforms = transforms
        self.ofps = 30
        self.tfps = 10
        self.view_step = 1 if VCL is None else int(VCL // 2) * 3
        self.VCL = VCL
        self.ttc_TT = 2.
        self.ttc_TA = 1.
        self.video_ext = ".png"

        if vertical_flip_prob > 0.:
            self.vflipper = RandomVerticalFlip(vertical_flip_prob)

        if horizontal_flip_prob > 0.:
            self.hflipper = RandomHorizontalFlip(horizontal_flip_prob)

        self._read_anno(split_file)
        self._prepare_views()

        print("DADA initialized!")

    def _read_anno(self, split_file_):
        clip_timesteps = []
        clip_binary_labels = []
        clip_cat_labels = []
        clip_ego = []
        clip_toa = []
        clip_ttc = []
        clip_acc = []
        clip_smoothed_labels = []

        errors = []

        split_dict = {
            'train': "DADA2K_my_split/training.txt",
            'test': "DADA2K_my_split/testing.txt",
            'val': "DADA2K_my_split/testing.txt",
            'test_csv': "DADA2K_my_split/testing.txt"
        }

        split_file = split_file_ if split_file_ is not None else os.path.join(self.root_path, split_dict[self.phase])
        with open(split_file, 'r') as file:
            clip_names = [line.rstrip() for line in file]

        df = pd.read_csv(os.path.join(self.root_path, "annotation", "full_anno.csv"))

        for clip in tqdm(clip_names, "Part 1/2. Reading and checking clips"):
            clip_type, clip_subfolder = clip.split("/")
            row = df[(df["video"] == int(clip_subfolder)) & (df["type"] == int(clip_type))]
            info = f"clip: {clip}, type: {clip_type}, subfolder: {clip_subfolder}, rows found: {row}"
            description_csv = row["texts"]
            assert len(row) == 1, f"Multiple results! \n{info}"
            if len(row) != 1:
                errors.append(info)
            row = row.iloc[0]
            with zipfile.ZipFile(os.path.join(self.root_path, "frames", clip, "images.zip"), 'r') as zipf:
                framenames = natsorted([f for f in zipf.namelist() if os.path.splitext(f)[1] == self.video_ext])
            timesteps = natsorted([int(os.path.splitext(f)[0].split("_")[-1]) for f in framenames])
            if_acc_video = int(row["whether an accident occurred (1/0)"])
            st = int(row["abnormal start frame"])
            en = int(row["abnormal end frame"])
            if st > -1 and en > -1:
                binary_labels = [1 if st <= t <= en else 0 for t in timesteps]
            else:
                binary_labels = [0 for t in timesteps]
            cat_labels = [l * clip_type for l in binary_labels]
            if_ego = clip_type in self.ego_categories
            toa = int(row["accident frame"])
            ttc = compute_time_vector(binary_labels, fps=self.ofps, TT=self.ttc_TT, TA=self.ttc_TA)
            smoothed_labels = smooth_labels(labels=torch.Tensor(binary_labels), time_vector=ttc,
                                            before_limit=self.ttc_TT, after_limit=self.ttc_TA)

            clip_timesteps.append(timesteps)
            clip_binary_labels.append(binary_labels)
            clip_cat_labels.append(cat_labels)
            clip_ego.append(if_ego)
            clip_toa.append(toa)
            clip_ttc.append(ttc)
            clip_acc.append(if_acc_video)
            clip_smoothed_labels.append(smoothed_labels)

        for line in errors:
            print(line)
        if len(errors) > 0:
            print(f"\n====\nerrors: {len(errors)}. You can add saving the error list in the code.")
            exit(0)

        assert len(clip_names) == len(clip_timesteps) == len(clip_binary_labels) == len(clip_cat_labels)
        self.clip_names = clip_names
        self.clip_timesteps = clip_timesteps
        self.clip_bin_labels = clip_binary_labels
        self.clip_cat_labels = clip_cat_labels
        self.clip_ego = clip_ego
        self.clip_toa = clip_toa
        self.clip_ttc = clip_ttc
        self.clip_smoothed_labels = clip_smoothed_labels

        print(f"Dataset (not views)")
        all_labels = []
        [all_labels.extend(bl) for bl in clip_binary_labels]
        all_labels = np.array(all_labels)
        unique, counts = np.unique(all_labels, return_counts=True)
        print("unique values and their counts:")
        for u, c in zip(unique, counts):
            print(f"item {u}: {c} times")

    def _prepare_views(self):
        dataset_sequences = []
        if self.VCL is None:
            sequencer = FullSequencer(seq_frequency=self.tfps, input_frequency=self.ofps)
            N = len(self.clip_names)
            for i in tqdm(range(N), desc="Part 2/2. Preparing views"):
                timesteps = self.clip_timesteps[i]
                sequences = sequencer.get_sequences(timesteps_nb=len(timesteps), input_frequency=self.ofps)
                if sequences is None:
                    continue
                dataset_sequences.extend([(i, seq) for seq in sequences])
        else:
            sequencer1 = RegularSequencer(seq_frequency=self.tfps, seq_length=self.VCL, step=self.view_step)
            sequencer2 = UnsafeOverlapSequencer(
                seq_frequency=self.tfps, seq_length=self.VCL, step=self.view_step * 8,
                surrounding_timesteps=(10, 10)
            )
            N = len(self.clip_names)
            for i in tqdm(range(N), desc="Part 2/2. Preparing views"):
                is_unsafe = self.clip_bin_labels[i]
                sequences1 = sequencer1.get_sequences(timesteps_nb=len(is_unsafe), input_frequency=self.ofps)
                sequences2 = sequencer2.get_sequences(is_unsafe=is_unsafe, input_frequency=self.ofps)
                random.shuffle(sequences1)
                random.shuffle(sequences2)
                l1, l2 = len(sequences1), len(sequences2)
                sequences = sequences1[:max(1, int(l1//2))] + sequences2[:max(1, int(l2//10))]
                if sequences in (None, []):
                    print("No sequences for ", self.clip_names[i])
                    continue
                dataset_sequences.extend([(i, seq) for seq in sequences])
        self.dataset_samples = dataset_sequences

        print("Dataset len (samples nb): ", len(self.dataset_samples))
        all_labels = []
        for sample in self.dataset_samples:
            clip_id, seq = sample
            bin_labels = [self.clip_bin_labels[clip_id][j] for j in seq]
            all_labels.extend(bin_labels)
        all_labels = np.array(all_labels)
        unique, counts = np.unique(all_labels, return_counts=True)
        print("[VIEWS] Unique values and their counts:")
        for u, c in zip(unique, counts):
            print(f"item {u}: {c} times")


    def __len__(self):
        return len(self.dataset_samples)

    def _get_random_subbatch(self, count):
        if self.VCL is None:
            return 0, count
        else:
            # if video is small then VCL, return full video
            if count <= self.VCL:
                return 0, count
            max_ = count - self.VCL
            begin = randint(0, max_)
            end = begin + self.VCL
            return begin, end

    def _add_video_filler(self, frames):
        try:
            filler_count = self.VCL - len(frames)
        except TypeError:
            return frames
        if filler_count > 0:
            filler = np.full((filler_count,) + frames.shape[1:], 0)
            frames = np.concatenate((frames, filler), axis=0)
        return frames

    def load_images_zip(self, dataset_sample):
        clip_id, frame_seq = dataset_sample
        clip_name = self.clip_names[clip_id]
        timesteps = [self.clip_timesteps[clip_id][idx] for idx in frame_seq]
        filenames = [f"{str(ts).zfill(4)}{self.video_ext}" for ts in timesteps]
        images = []
        with zipfile.ZipFile(os.path.join(self.root_path, "frames", clip_name, "images.zip"), 'r') as zipf:
            for fname in filenames:
                with zipf.open(fname) as file:
                    img = Image.open(io.BytesIO(file.read()))  # Convert bytes to a PIL Image
                if img is None:
                    print("Image doesn't exist! ", fname)
                    exit(1)
                images.append(np.asarray(img))
        images = np.array(images)
        video_len_orig = len(images)
        images = self._add_video_filler(images)
        return images.astype('float32'), video_len_orig

    def gather_info(self, index):
        sample = self.dataset_samples[index]
        clip_id, frame_seq = sample

        timesteps = frame_seq
        labels = np.array([self.clip_bin_labels[clip_id][ts] for ts in timesteps])
        stepslabels = np.stack([timesteps, labels], axis=-1)
        ttc = np.array([self.clip_ttc[clip_id][ts] for ts in timesteps])

        info = np.array([self.VCL, index]).astype('float')
        return info, stepslabels, ttc

    def __getitem__(self, index):
        sample = self.dataset_samples[index]
        # read RGB video (trimmed)
        video_data, video_len_orig = self.load_images_zip(sample)
        # gather info AND frame-level targets (because we can have multiple anomaly ranges in one video)
        data_info, timesteps_targets, ttcs = self.gather_info(index)

        # pre-process
        if self.transforms['image'] is not None:
            video_data = self.transforms['image'](video_data)  # (T, C, H, W)

        if hasattr(self, 'hflipper'):
            _, video_data = self.hflipper(video_data)

        if hasattr(self, 'vflipper'):
            _, video_data = self.vflipper(video_data)

        return video_data, data_info, timesteps_targets, ttcs


def setup_dada(Dada, cfg, num_workers=-1,
               VCL=None, phase=None, split_file=None):
    mean = cfg.get('data_mean', [0.218, 0.220, 0.209])
    std = cfg.get('data_std', [0.277, 0.280, 0.277])
    params = {
        'input_shape': cfg.input_shape,
        'mean': mean,
        'std': std,
    }

    vertical_flip_prob = cfg.get('vertical_flip_prob', 0.)
    horizontal_flip_prob = cfg.get('horizontal_flip_prob', 0.)

    #  def transf_train(x):
    transform_dict = {
        'image': transforms.Compose([
            pad_frames(cfg.input_shape),
            transforms.Lambda(lambda x: torch.tensor(x)),
            # [T, H, W, C] -> [T, C, H, W]
            transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
            transforms.Lambda(lambda x: x / 255.0),
            transforms.Normalize(params['mean'], params['std']),
            # [T, C, H, W]
        ]),
    }

    transform_dict_train = {
        'image': transforms.Compose([
            pad_frames(cfg.input_shape),
            transforms.Lambda(lambda x: torch.tensor(x)),
            # [T, H, W, C] -> [T, C, H, W]
            transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
            T.AugMix(),
            transforms.Lambda(lambda x: x / 255.0),
            transforms.Normalize(params['mean'], params['std']),
            # [T, C, H, W]
        ]),
    }

    transform_dict_to_play = {
        'image': None,
    }

    traindata_loader, testdata_loader = None, None

    # training dataset
    pin_memory_val = True
    if platform.system() == 'Windows':
        pin_memory_val = False

    # testing dataset
    if phase == 'train':
        train_data = Dada(
            cfg.data_path, 'train',
            transforms=transform_dict_train,
            VCL=VCL,
            vertical_flip_prob=vertical_flip_prob,
            horizontal_flip_prob=horizontal_flip_prob,
            split_file=split_file)
        sampler = ShortSampler(train_data, num_samples_per_epoch=3272)  # 409 * batch
        traindata_loader = DataLoader(
            dataset=train_data, batch_size=cfg.batch_size,
            sampler=sampler, # shuffle=True,
            drop_last=True, num_workers=num_workers,
            pin_memory=pin_memory_val)
        print("# train set: {}".format(len(train_data)))
        cfg.update(FPS=train_data.tfps)
    else:
        if phase in ('test', 'test_csv'):
            # validation dataset
            test_data = Dada(cfg.data_path, 'val', transforms=transform_dict, split_file=split_file)
            testdata_loader = DataLoader(
                dataset=test_data, batch_size=1, shuffle=False,
                drop_last=True, num_workers=num_workers,
                pin_memory=pin_memory_val)
            print("# test set: %d" % (len(test_data)))
        elif phase == 'play':
            # validation dataset
            test_data = Dada(
                cfg.data_path, 'val',
                transforms=transform_dict_to_play,
                split_file=split_file)
            testdata_loader = DataLoader(
                dataset=test_data, batch_size=1, shuffle=False,
                drop_last=True, num_workers=num_workers,
                pin_memory=pin_memory_val)
            print("# test set: %d" % (len(test_data)))
        cfg.update(FPS=test_data.tfps)

    return traindata_loader, testdata_loader
