import decord
import os
import numpy as np
import torch
from qwen_vl_utils import smart_resize
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
from typing import Union

IMAGE_FACTOR = 28

def read_video_decord(video_path, start_time, end_time, fps, max_pixels=560*560, min_pixels=84*84, allow_end_truncation=False):
    """
    Reads a video using decord with specified start time, end time, and fps.

    Args:
        video_path (str): Path to the video file.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.
        fps (float): Desired frames per second.

    Returns:
        tuple: A tuple containing:
            - frame_timestamps (list of float): Timestamps of the extracted frames.
            - frames (numpy.ndarray): Extracted frames as a NumPy array.
    """
    # Initialize the video reader
    vr = decord.VideoReader(video_path)
    video_fps = vr.get_avg_fps()
    total_frames = len(vr)
    duration = total_frames / video_fps

    # # Ensure start and end times are within the video duration
    # start_time = max(0, start_time)
    # end_time = min(duration, end_time)
    if allow_end_truncation:
        end_time = min(duration, end_time)
    
    if start_time >= end_time or end_time > duration:
        return None, None

    # Calculate frame indices to sample
    start_frame = int(start_time * video_fps)
    end_frame = int(end_time * video_fps)
    frame_indices = np.arange(start_frame, end_frame, video_fps / fps).astype(int)

    # Read frames and calculate timestamps
    video = vr.get_batch(frame_indices).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)
    
    nframes, _, height, width = video.shape
    frame_timestamps = frame_indices / video_fps
    resized_height, resized_width = smart_resize(height, width, IMAGE_FACTOR, min_pixels=min_pixels, max_pixels=max_pixels)
    
    video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
    return frame_timestamps, video

def read_until_end(video_path, history_length, fps, max_pixels=560*560, min_pixels=84*84):
    vr = decord.VideoReader(video_path)
    video_fps = vr.get_avg_fps()
    total_frames = len(vr)
    duration = total_frames / video_fps

    start_time = max(0, duration - history_length)
    end_time = duration

    return read_video_decord(
        video_path,
        start_time,
        end_time,
        fps,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
        allow_end_truncation=True,
    )

def read_video_decord_strict(
    video_path,
    start_time,
    end_time,
    fps,
    max_pixels=560*560,
    min_pixels=84*84,
    allow_end_truncation=False,
):

    vr = decord.VideoReader(video_path)
    duration = vr.get_frame_timestamp(len(vr) - 1)[1]

    if allow_end_truncation:
        end_time = min(end_time, duration)

    if start_time >= end_time:
        return None, None

    # Target timestamps (ground truth timeline)
    target_ts = np.arange(start_time, end_time, 1.0 / fps)

    # Map timestamps → nearest frame index
    frame_indices = []
    actual_ts = []

    for t in target_ts:
        idx = vr.get_frame_index(t)
        if idx < 0 or idx >= len(vr):
            continue
        frame_indices.append(idx)
        actual_ts.append(vr.get_frame_timestamp(idx)[0])

    if not frame_indices:
        return None, None

    frames = vr.get_batch(frame_indices).asnumpy()
    video = torch.from_numpy(frames).permute(0, 3, 1, 2)

    _, _, h, w = video.shape
    resized_h, resized_w = smart_resize(
        h, w, IMAGE_FACTOR, min_pixels=min_pixels, max_pixels=max_pixels
    )

    video = transforms.functional.resize(
        video,
        [resized_h, resized_w],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()

    return np.array(actual_ts), video

import subprocess
import ffmpeg

def read_video_ffmpeg_fixed(
    video_path,
    start_time,
    end_time,
    fps=5,
    max_pixels=560 * 560,
):
    """
    Timestamp-accurate video decoding using FFmpeg.
    Frames are sampled exactly every 1/fps seconds.
    """

    probe = ffmpeg.probe(video_path)
    v = next(s for s in probe["streams"] if s["codec_type"] == "video")
    iw, ih = int(v["width"]), int(v["height"])

    # compute resize
    scale = min((max_pixels / (iw * ih)) ** 0.5, 1.0)
    ow, oh = int(iw * scale), int(ih * scale)
    
    if start_time >= end_time:
        return None

    vf = f"fps={fps},scale={ow}:{oh}"

    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-ss", str(start_time),
        "-to", str(end_time),
        "-i", video_path,
        "-vf", vf,
        "-pix_fmt", "rgb24",
        "-f", "rawvideo",
        "-"
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    raw = proc.stdout.read()
    proc.stdout.close()
    proc.wait()

    frame_size = ow * oh * 3
    nframes = len(raw) // frame_size

    video = np.frombuffer(raw, np.uint8)
    video = video[: nframes * frame_size]
    video = video.copy()
    video = video.reshape(nframes, oh, ow, 3)

    video = torch.from_numpy(video).permute(0, 3, 1, 2).float()
    return video
    
def ffmpeg_once(
    src_path: str,
    dst_path: str,
    *,
    fps: int = None,
    resolution: int = None,
    pad: str = '#000000',
    mode='bicubic',
    start_time: Union[str, float] = None,   # e.g. "00:00:05" or 5.0
    end_time: Union[str, float] = None      # e.g. "00:00:10" or 10.0
):
    if os.path.exists(dst_path):
        return
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    command = [
        './ffmpeg/ffmpeg',
        '-y',
    ]

    # Fast seek (put -ss before -i)
    if start_time is not None:
        command += ['-ss', str(start_time)]

    command += [
        '-sws_flags', mode,
        '-i', src_path,
        '-an',
        '-threads', '10',
    ]

    # End control
    if end_time is not None:
        if start_time is not None and isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
            duration = float(end_time) - float(start_time)
            if duration <= 0:
                raise ValueError("end_time must be greater than start_time")
            command += ['-t', str(duration)]
        else:
            # absolute end timestamp
            command += ['-to', str(end_time)]

    if fps is not None:
        command += ['-r', str(fps)]

    if resolution is not None:
        command += ['-vf',
            f"scale='if(gt(iw\\,ih)\\,{resolution}\\,-2)':'if(gt(iw\\,ih)\\,-2\\,{resolution})',"
            f"pad={resolution}:{resolution}:(ow-iw)/2:(oh-ih)/2:color='{pad}'"
        ]

    command += [dst_path]

    subprocess.run(command, check=True)