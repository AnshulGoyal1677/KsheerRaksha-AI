import os
import random
import cv2
import numpy as np

class GaitDataset:
    """
    Video Dataset for Cattle Lameness Detection.
    Loads video files, uniformly extracts T frames, resizes, normalizes,
    and applies video-consistent spatial augmentations during training.
    """
    def __init__(self, samples, num_frames=16, target_size=(224, 224), is_training=False):
        """
        samples: list of dicts with keys 'filename', 'label', 'path'
        num_frames: number of uniformly sampled frames per video (default 16)
        target_size: (H, W) for frame resizing (default 224, 224)
        is_training: if True, applies training augmentations (horizontal flip, brightness)
        """
        self.samples = samples
        self.num_frames = num_frames
        self.target_size = target_size
        self.is_training = is_training
        
        # ImageNet normalization statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        
        self.label_map = {"NORMAL": 0, "LAME": 1}

    def __len__(self):
        return len(self.samples)

    def _extract_frames(self, video_path):
        """Uniformly samples self.num_frames from the video."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # In case frame count metadata is invalid, count manually
        if total_frames <= 0:
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            total_frames = len(frames)
            if total_frames == 0:
                raise ValueError(f"Could not read any frames from {video_path}")
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            sampled = [frames[i] for i in frame_indices]
            return sampled

        # Direct indexed sampling
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        sampled_frames = []
        
        current_idx = 0
        target_set = set(frame_indices)
        
        # Read sequentially to ensure reliability across all video decoders
        while cap.isOpened() and len(sampled_frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if current_idx in target_set:
                # Count duplicates if video has fewer frames than requested
                repeat_count = np.sum(frame_indices == current_idx)
                for _ in range(repeat_count):
                    sampled_frames.append(frame.copy())
            current_idx += 1
            
        cap.release()
        
        # Fallback padding if video was shorter than expected
        if len(sampled_frames) == 0:
            raise ValueError(f"Failed to extract frames from {video_path}")
        while len(sampled_frames) < self.num_frames:
            sampled_frames.append(sampled_frames[-1].copy())
            
        return sampled_frames[:self.num_frames]

    def _apply_augmentation(self, frames):
        """
        Applies video-level augmentation (same transform across all frames in clip)
        to preserve temporal consistency.
        """
        # 1. Random Horizontal Flip (simulates cattle walking left vs right)
        do_flip = random.random() < 0.5
        
        # 2. Random Brightness/Contrast Jitter (mild)
        alpha = random.uniform(0.85, 1.15)  # contrast
        beta = random.uniform(-15, 15)      # brightness
        
        processed = []
        for f in frames:
            img = f
            if do_flip:
                img = cv2.flip(img, 1)
            if alpha != 1.0 or beta != 0.0:
                img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
            processed.append(img)
            
        return processed

    def _preprocess_frame(self, frame):
        """Resize, convert BGR to RGB, scale to [0, 1], and normalize."""
        resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - self.mean) / self.std
        # Transpose from (H, W, C) to (C, H, W) for PyTorch
        return normalized.transpose(2, 0, 1)

    def get_sample(self, index):
        sample_info = self.samples[index]
        video_path = sample_info["path"]
        label_str = sample_info["label"]
        label_idx = self.label_map[label_str]
        
        raw_frames = self._extract_frames(video_path)
        
        if self.is_training:
            raw_frames = self._apply_augmentation(raw_frames)
            
        tensor_frames = np.stack([self._preprocess_frame(f) for f in raw_frames], axis=0)
        # Returns shape: (T, C, H, W) -> (16, 3, 224, 224) as float32
        return tensor_frames.astype(np.float32), label_idx, sample_info
