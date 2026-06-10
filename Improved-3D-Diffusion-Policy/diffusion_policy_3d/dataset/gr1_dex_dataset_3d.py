from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer, StringNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
import diffusion_policy_3d.model.vision_3d.point_process as point_process
from termcolor import cprint

class GR1DexDataset3D(BaseDataset):
    def __init__(self,
            zarr_path,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            num_points=4096,
            use_pc_augmentation=False,
            pc_jitter_std=0.005,
            pc_dropout_ratio=0.0,
            ):
        super().__init__()
        cprint(f'Loading GR1DexDataset from {zarr_path}', 'green')
        self.task_name = task_name

        self.num_points = num_points
        self.use_pc_augmentation = use_pc_augmentation
        self.pc_jitter_std = pc_jitter_std
        self.pc_dropout_ratio = pc_dropout_ratio


        buffer_keys = [
            'state', 
            'action',]
        
        buffer_keys.append('point_cloud')


            
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=buffer_keys)
        
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        val_set.use_pc_augmentation = False
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {'action': self.replay_buffer['action']}
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        normalizer['point_cloud'] = SingleFieldLinearNormalizer.create_identity()
        normalizer['agent_pos'] = SingleFieldLinearNormalizer.create_identity()
        
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _augment_point_cloud(self, point_cloud):
        """Apply small per-frame jitter/dropout to the (already-sampled) point cloud.

        Jitter only perturbs XYZ (channels :3), leaving color channels untouched.
        Dropout overwrites a fraction of points (all channels) with copies of other
        points in the same frame, simulating missing/occluded depth returns while
        preserving the point count.
        """
        T, N, _ = point_cloud.shape

        if self.pc_jitter_std > 0:
            noise = np.random.normal(0, self.pc_jitter_std, size=(T, N, 3)).astype(np.float32)
            point_cloud[..., :3] += noise

        if self.pc_dropout_ratio > 0:
            num_drop = round(N * self.pc_dropout_ratio)
            for t in range(T):
                drop_idx = np.random.choice(N, size=num_drop, replace=False)
                replace_idx = np.random.randint(0, N, size=num_drop)
                point_cloud[t, drop_idx] = point_cloud[t, replace_idx]

        return point_cloud

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32)
        point_cloud = sample['point_cloud'][:,].astype(np.float32)
        point_cloud = point_process.uniform_sampling_numpy(point_cloud, self.num_points)
        if self.use_pc_augmentation:
            point_cloud = self._augment_point_cloud(point_cloud)
        data = {
            'obs': {
                'agent_pos': agent_pos,
                'point_cloud': point_cloud,
                },
            'action': sample['action'].astype(np.float32)}
           
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        to_torch_function = lambda x: torch.from_numpy(x) if x.__class__.__name__ == 'ndarray' else x
        torch_data = dict_apply(data, to_torch_function)
        return torch_data

