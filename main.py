import shutil
import random
from pathlib import Path
from typing import List, Tuple

import cv2

def get_class_folders(dataset_path: Path) -> List[Path]:
    """
    Get the list of class directories inside the dataset path.
    """
    return [f for f in dataset_path.iterdir() if f.is_dir()]


def list_videos(class_folder: Path) -> List[Path]:
    """
    List all video files in a class folder.
    """
    return [f for f in class_folder.iterdir() if f.suffix.lower() in ['.mp4', '.avi']]


def split_videos(videos: List[Path], train_ratio: float, eval_ratio: float, seed: int = 42) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Randomly split videos into train, eval, and test sets based on provided ratios.
    """
    random.seed(seed)
    videos = videos.copy()
    random.shuffle(videos)

    n_total = len(videos)
    n_train = int(n_total * train_ratio)
    n_eval = int(n_total * eval_ratio)
    n_test = n_total - n_train - n_eval

    train_videos = videos[:n_train]
    eval_videos = videos[n_train:n_train + n_eval]
    test_videos = videos[n_train + n_eval:]

    return train_videos, eval_videos, test_videos


def ensure_dir_exists(path: Path):
    """
    Create directory if it does not exist.
    """
    path.mkdir(parents=True, exist_ok=True)


def convert_mp4_to_avi(src_path: Path, dest_path: Path):
    """
    Convert an MP4 video to AVI format using OpenCV.
    """
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file {src_path}")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(dest_path), fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()


def copy_and_convert_video(src_path: Path, dest_path: Path):
    """
    Copy video to destination, converting from MP4 to AVI if necessary.
    """
    if src_path.suffix.lower() == '.mp4':
        dest_path = dest_path.with_suffix('.avi')
        convert_mp4_to_avi(src_path, dest_path)
    else:
        shutil.copy2(src_path, dest_path)


def build_dataset_structure(
    raw_dataset_path: str,
    output_dataset_path: str,
    train_ratio: float = 0.7,
    eval_ratio: float = 0.15,
    test_ratio: float = 0.15,
    overwrite: bool = False,
    seed: int = 42
):
    """
    Build a dataset with train, eval, and test splits, converting videos as needed.
    Args:
        raw_dataset_path (str): Path to the original raw dataset with class folders.
        output_dataset_path (str): Path where the new dataset will be created.
        train_ratio (float): Ratio of videos for training set.
        eval_ratio (float): Ratio of videos for evaluation set.
        test_ratio (float): Ratio of videos for test set.
        overwrite (bool): If True, overwrite existing dataset folder.
        seed (int): Random seed for reproducibility.
    """
    raw_dataset = Path(raw_dataset_path)
    output_dataset = Path(output_dataset_path)

    if output_dataset.exists():
        if overwrite:
            shutil.rmtree(output_dataset)
        else:
            print("Dataset folder already exists and overwrite is False. Exiting.")
            return

    splits = ['train', 'eval', 'test']
    for split in splits:
        ensure_dir_exists(output_dataset / split)

    class_folders = get_class_folders(raw_dataset)
    for class_folder in class_folders:
        class_name = class_folder.name
        videos = list_videos(class_folder)
        train_videos, eval_videos, test_videos = split_videos(
            videos, train_ratio, eval_ratio, seed=seed
        )
        split_map = {
            'train': train_videos,
            'eval': eval_videos,
            'test': test_videos
        }

        for split in splits:
            split_class_dir = output_dataset / split / class_name
            ensure_dir_exists(split_class_dir)
            for src_video in split_map[split]:
                dest_filename = src_video.stem
                if src_video.suffix.lower() == ".mp4":
                    dest_path = split_class_dir / f"{dest_filename}.avi"
                else:
                    dest_path = split_class_dir / src_video.name
                copy_and_convert_video(src_video, dest_path)

    print(f"Dataset created at {output_dataset.resolve()}")


if __name__ == "__main__":
    # === CONFIGURE YOUR PARAMETERS HERE ===
    RAW_DATASET_PATH = "raw_dataset"  # Path to your raw dataset
    OUTPUT_DATASET_PATH = "dataset"   # Path for the output dataset
    TRAIN_RATIO = 0.7
    EVAL_RATIO = 0.15
    TEST_RATIO = 0.15
    OVERWRITE = True                  # Set to True to overwrite existing dataset
    SEED = 42

    assert abs(TRAIN_RATIO + EVAL_RATIO + TEST_RATIO - 1.0) < 1e-6, "Ratios must sum to 1."

    build_dataset_structure(
        raw_dataset_path=RAW_DATASET_PATH,
        output_dataset_path=OUTPUT_DATASET_PATH,
        train_ratio=TRAIN_RATIO,
        eval_ratio=EVAL_RATIO,
        test_ratio=TEST_RATIO,
        overwrite=OVERWRITE,
        seed=SEED
    )