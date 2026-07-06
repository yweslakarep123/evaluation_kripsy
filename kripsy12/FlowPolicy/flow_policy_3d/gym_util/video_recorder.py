"""H.264 video recorder (PyAV) for MuJoCo rollouts."""

from typing import Optional

import numpy as np

try:
    import av
except ImportError:
    av = None


class VideoRecorder:
    def __init__(self, fps, codec, input_pix_fmt, **kwargs):
        self.fps = fps
        self.codec = codec
        self.input_pix_fmt = input_pix_fmt
        self.kwargs = kwargs
        self._reset_state()

    def _reset_state(self):
        self.container = None
        self.stream = None
        self.shape = None
        self.dtype = None

    @classmethod
    def create_h264(
        cls,
        fps,
        codec="h264",
        input_pix_fmt="rgb24",
        output_pix_fmt="yuv420p",
        crf=18,
        profile="high",
        **kwargs,
    ):
        if av is None:
            raise ImportError(
                "PyAV (av) is required for H.264 recording. Install with: pip install av"
            )
        return cls(
            fps=fps,
            codec=codec,
            input_pix_fmt=input_pix_fmt,
            pix_fmt=output_pix_fmt,
            options={"crf": str(crf), "profile": profile},
            **kwargs,
        )

    def __del__(self):
        self.stop()

    def is_ready(self):
        return self.stream is not None

    def start(self, file_path, start_time=None):
        if self.is_ready():
            self.stop()
        self.container = av.open(file_path, mode="w")
        self.stream = self.container.add_stream(self.codec, rate=self.fps)
        codec_context = self.stream.codec_context
        for k, v in self.kwargs.items():
            setattr(codec_context, k, v)

    def write_frame(self, img: np.ndarray, frame_time=None):
        if not self.is_ready():
            raise RuntimeError("Must run start() before writing!")
        if self.shape is None:
            self.shape = img.shape
            self.dtype = img.dtype
            h, w, _c = img.shape
            self.stream.width = w
            self.stream.height = h
        assert img.shape == self.shape
        assert img.dtype == self.dtype
        frame = av.VideoFrame.from_ndarray(img, format=self.input_pix_fmt)
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def stop(self):
        if not self.is_ready():
            return
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        self._reset_state()
