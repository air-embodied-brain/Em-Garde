"""
FFmpeg-based video reader to replace decord.VideoReader
Optimized for ARM architecture with minimal computational overhead
"""
import subprocess
import numpy as np
import torch
from typing import List, Tuple, Optional, Union
import os


class FFmpegVideoReader:
    """
    FFmpeg-based video reader that mimics decord.VideoReader interface
    """
    
    def __init__(self, video_path: str, ctx=None, num_threads: int = 1):
        """
        Initialize video reader
        
        Args:
            video_path: Path to video file
            ctx: Context (ignored, for compatibility with decord)
            num_threads: Number of threads (ignored, for compatibility)
        """
        self.video_path = video_path
        self._probe_video()
        
    def _probe_video(self):
        """Probe video to get metadata using ffprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=avg_frame_rate,width,height,duration,nb_frames',
                '-of', 'default=noprint_wrappers=1',
                self.video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            self._width = None
            self._height = None
            self._fps = None
            self._duration = None
            self._total_frames = None
            
            for line in result.stdout.strip().split('\n'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    if key == 'width':
                        self._width = int(value)
                    elif key == 'height':
                        self._height = int(value)
                    elif key == 'avg_frame_rate':
                        # Parse fraction like "30000/1001"
                        if '/' in value:
                            num, den = value.split('/')
                            self._fps = float(num) / float(den) if float(den) != 0 else 0
                        else:
                            self._fps = float(value)
                    elif key == 'duration':
                        self._duration = float(value)
                    elif key == 'nb_frames':
                        if value != 'N/A':
                            self._total_frames = int(value)
            
            # Fallback: calculate total frames if not provided
            if self._total_frames is None and self._duration and self._fps:
                self._total_frames = int(self._duration * self._fps)
                
        except Exception as e:
            raise RuntimeError(f"Failed to probe video {self.video_path}: {e}")
    
    def __len__(self) -> int:
        """Return total number of frames"""
        return self._total_frames if self._total_frames else 0
    
    def get_avg_fps(self) -> float:
        """Return average FPS"""
        return self._fps if self._fps else 30.0
    
    def get_frame_timestamp(self, idx: int) -> Tuple[float, float]:
        """
        Get timestamp for frame at index
        Returns (start_time, end_time) for compatibility with decord
        """
        fps = self.get_avg_fps()
        start_time = idx / fps
        end_time = (idx + 1) / fps
        return (start_time, end_time)
    
    def get_frame_index(self, timestamp: float) -> int:
        """Get frame index closest to given timestamp"""
        fps = self.get_avg_fps()
        return int(timestamp * fps)
    
    def _read_frames_at_indices(self, indices: List[int]) -> np.ndarray:
        """
        Read frames at specific indices using FFmpeg
        Optimized to minimize subprocess calls
        """
        if not indices:
            return np.array([])
        
        # Sort indices to potentially batch sequential reads
        sorted_indices = sorted(set(indices))
        
        # For single frame or few frames, read individually
        if len(sorted_indices) <= 3:
            frames = []
            for idx in sorted_indices:
                frame = self._read_single_frame(idx)
                if frame is not None:
                    frames.append(frame)
            return np.array(frames) if frames else np.array([])
        
        # For multiple frames, try to extract a range
        min_idx = min(sorted_indices)
        max_idx = max(sorted_indices)
        
        # If indices are close together, extract the whole range
        if max_idx - min_idx < len(sorted_indices) * 3:
            return self._read_frame_range(sorted_indices, min_idx, max_idx)
        
        # Otherwise, read individually
        frames = []
        for idx in sorted_indices:
            frame = self._read_single_frame(idx)
            if frame is not None:
                frames.append(frame)
        return np.array(frames) if frames else np.array([])
    
    def _read_single_frame(self, idx: int) -> Optional[np.ndarray]:
        """Read a single frame at given index"""
        fps = self.get_avg_frame_rate()
        timestamp = idx / fps
        
        cmd = [
            'ffmpeg', '-loglevel', 'error',
            '-ss', str(timestamp),
            '-i', self.video_path,
            '-frames:v', '1',
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            if result.returncode != 0:
                return None
            
            frame_size = self._width * self._height * 3
            if len(result.stdout) < frame_size:
                return None
            
            frame = np.frombuffer(result.stdout[:frame_size], dtype=np.uint8)
            frame = frame.reshape((self._height, self._width, 3))
            return frame
        except:
            return None
    
    def _read_frame_range(self, target_indices: List[int], start_idx: int, end_idx: int) -> np.ndarray:
        """Read a range of frames and select target indices"""
        fps = self.get_avg_fps()
        start_time = start_idx / fps
        duration = (end_idx - start_idx + 1) / fps
        
        cmd = [
            'ffmpeg', '-loglevel', 'error',
            '-ss', str(start_time),
            '-t', str(duration),
            '-i', self.video_path,
            '-vf', f'fps={fps}',
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if result.returncode != 0:
                # Fallback to individual frame reading
                frames = []
                for idx in target_indices:
                    frame = self._read_single_frame(idx)
                    if frame is not None:
                        frames.append(frame)
                return np.array(frames) if frames else np.array([])
            
            frame_size = self._width * self._height * 3
            total_frames_in_range = end_idx - start_idx + 1
            expected_size = frame_size * total_frames_in_range
            
            if len(result.stdout) < expected_size:
                # Partial read, adjust
                total_frames_in_range = len(result.stdout) // frame_size
            
            video = np.frombuffer(result.stdout[:frame_size * total_frames_in_range], dtype=np.uint8)
            video = video.reshape((total_frames_in_range, self._height, self._width, 3))
            
            # Select target indices
            selected_frames = []
            for idx in target_indices:
                local_idx = idx - start_idx
                if 0 <= local_idx < len(video):
                    selected_frames.append(video[local_idx])
            
            return np.array(selected_frames) if selected_frames else np.array([])
        except:
            # Fallback to individual frame reading
            frames = []
            for idx in target_indices:
                frame = self._read_single_frame(idx)
                if frame is not None:
                    frames.append(frame)
            return np.array(frames) if frames else np.array([])
    
    def get_batch(self, indices: Union[List[int], np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Get batch of frames at given indices
        Mimics decord.VideoReader.get_batch()
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        elif isinstance(indices, np.ndarray):
            indices = indices.tolist()
        
        return self._read_frames_at_indices(indices)
    
    def __getitem__(self, idx: int) -> 'FrameWrapper':
        """Get single frame at index, returns wrapper for .numpy() compatibility"""
        frame = self._read_single_frame(idx)
        if frame is None:
            raise IndexError(f"Frame {idx} not found")
        return FrameWrapper(frame)
    
    def get_avg_frame_rate(self) -> float:
        """Alias for get_avg_fps() for compatibility"""
        return self.get_avg_fps()


class FrameWrapper:
    """Wrapper to provide .numpy() method for compatibility with decord"""
    def __init__(self, frame: np.ndarray):
        self._frame = frame
    
    def numpy(self) -> np.ndarray:
        return self._frame
    
    def __array__(self):
        return self._frame


# Compatibility alias for decord.VideoReader
VideoReader = FFmpegVideoReader


def cpu(device_id: int = 0):
    """
    Dummy function for compatibility with decord.cpu()
    Returns None as FFmpeg doesn't need context
    """
    return None
