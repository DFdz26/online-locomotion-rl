"""
Class: VideoSaver
created by: Daniel Mauricio Fernandez Gonzalez
e-mail: dafer21@student.sdu.dk
date: 8 April 2023

Video Saver
"""

import numpy as np
from PIL import Image
import imageio


class VideoFormat:
    RGB = "RGB"
    RGBA = "RGBA"


def check_video_format_accepted(vFormat: str):
    return hasattr(VideoFormat, vFormat)


class VideoSaver:
    video_format: str

    def __init__(self, fps, n_frames, height, width, name_video="video", vFormat=VideoFormat.RGBA):
        self.fps = fps
        self.started_record = False
        self.n_stored_frames = 0
        self.n_tot_frames = n_frames
        self.filename = name_video
        self.height = height
        self.width = width
        self.buffer_full = False

        if not(check_video_format_accepted(vFormat)):
            raise Exception(f"The format {vFormat} is not implemented.")

        self.stored_frames = np.zeros((n_frames, height, width, len(vFormat)), dtype=np.uint8)
        self.video_format = vFormat

    def start_record(self):
        self.started_record = True
        self.buffer_full = False
        self.n_stored_frames = 0

    def store_frame(self, frame):
        if self.n_stored_frames < self.n_tot_frames and self.started_record:
            self.stored_frames[self.n_stored_frames] = frame.copy()
            self.n_stored_frames += 1

            if self.n_stored_frames == self.n_tot_frames:
                self.buffer_full = True

    def save_video(self, filename=None, saveFile=True):

        if self.started_record:
            filename = self.filename if filename is None else filename
            frames = []

            filename += ".mp4"

            # Iterate over each frame in the stored array
            for frame in range(self.n_stored_frames):
                # Convert the numpy array to an image
                img = Image.fromarray(self.stored_frames[frame], self.video_format)
                # Append the image to the frames list
                frames.append(img)

            # Write the frames to a video file using imageio
            if saveFile:
                imageio.mimwrite(filename, frames, fps=self.fps)

            self.stop_record()

    def stop_record(self):
        self.started_record = False


if __name__ == "__main__":
    height = 480
    width = 640

    video_class = VideoSaver(30, 100, height, width)

    video_class.start_record()
    video = np.random.randint(0, 255, size=(100, height, width, 4), dtype=np.uint8)

    for _frame in video:
        video_class.store_frame(_frame)

    video_class.save_video()
