import subprocess
from pathlib import Path
import glob


def extract_frames_from_video(video_path, output_dir, frame_rate=1, downsample=1):
    """
    Extracts frames from a video file using FFmpeg.
    
    Args:
        video_path (str or Path): Path to the input video file.
        output_dir (str or Path): Directory to save the extracted images.
        frame_rate (int): Number of frames to extract per second of video.
        downsample (int): Downsampling factor for the images (e.g., 2 means 2x smaller).
    
    Returns:
        List[Path]: List of paths to the extracted image files.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build video filter
    filters = [f'fps={frame_rate}']
    if downsample > 1:
        filters.append(f'scale=iw/{downsample}:ih/{downsample}')
    
    # Build FFmpeg command
    output_pattern = output_dir / 'frame_%05d.png'
    cmd = [
        'ffmpeg', '-i', str(video_path),
        '-vf', ','.join(filters),
        '-y',  # overwrite existing files
        str(output_pattern)
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        extracted_paths = [Path(p) for p in sorted(glob.glob(str(output_dir / 'frame_*.png')))]
        
        print(f"Extracted {len(extracted_paths)} frames from {video_path}.")
        return extracted_paths
        
    except subprocess.CalledProcessError as e:
        raise IOError(f"Failed to extract frames: {e}.")
    except FileNotFoundError:
        raise IOError("FFmpeg not found. Install FFmpeg and add to PATH.")