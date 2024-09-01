from multiprocessing.sharedctypes import Value
import os
import cv2
import numpy as np


def loadvideo(filename: str) -> np.ndarray:
    """Loads a video from a file.
    
    Inputs:
        filename (str): filename of video
    Returns:
        A np.array with dimensions (channels=3, frames, height, width).
        The values will be uint8's raining from 0 to 255.

    Raises:
        FileNotFoundError: Could not find 'filename'
        ValueError: An error occurred while reading the video
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    
    capture = cv2.VideoCapture(filename)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    v = np.zeros((frame_count, frame_height, frame_width, 3), np.uint8)

    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError(f"Failed to load frame #{count} of {filename}.")
        rame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        v[count, ...] = frame

    v = v.transpose((3,0,1,2))

    return v