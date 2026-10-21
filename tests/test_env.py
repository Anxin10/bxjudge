import shutil
import subprocess

import pytest


def test_ffmpeg_present():
    assert shutil.which("ffmpeg") and shutil.which("ffprobe")


@pytest.mark.gpu
def test_torch_sees_3090():
    import torch

    assert torch.cuda.is_available()
    assert "3090" in torch.cuda.get_device_name(0)


@pytest.mark.gpu
def test_ffmpeg_lists_cuda_hwaccel():
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-hwaccels"], capture_output=True, text=True
    ).stdout
    assert "cuda" in out
