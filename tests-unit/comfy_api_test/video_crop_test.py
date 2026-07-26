import pytest
import torch
import av
import struct
from fractions import Fraction
from comfy_api.latest._input_impl.video_types import VideoFromFile, VideoFromComponents
from comfy_api.latest._util.video_types import VideoComponents, normalize_crop_rect


RED = torch.tensor([1.0, 0.0, 0.0])
GREEN = torch.tensor([0.0, 1.0, 0.0])
BLUE = torch.tensor([0.0, 0.0, 1.0])
WHITE = torch.tensor([1.0, 1.0, 1.0])


@pytest.fixture(scope="module")
def quadrant_components():
    """64x64, 2 frames; each 32x32 quadrant a distinct solid color to locate crops"""
    images = torch.zeros(2, 64, 64, 3)
    images[:, :32, :32] = RED
    images[:, :32, 32:] = GREEN
    images[:, 32:, :32] = BLUE
    images[:, 32:, 32:] = WHITE
    return VideoComponents(images=images, frame_rate=Fraction(30))


@pytest.fixture(scope="module")
def src(quadrant_components, tmp_path_factory):
    path = str(tmp_path_factory.mktemp("video") / "src.mp4")
    VideoFromComponents(quadrant_components).save_to(path)
    return path


def probe_dimensions(path):
    with av.open(path) as container:
        stream = container.streams.video[0]
        return stream.width, stream.height


def decoded_center_color(path):
    """Mean RGB of the center 8x8 of the first decoded frame"""
    with av.open(path) as container:
        frame = next(container.decode(container.streams.video[0]))
        rgb = torch.from_numpy(frame.to_ndarray(format="rgb24")).float() / 255.0
        h, w = rgb.shape[0], rgb.shape[1]
        return rgb[h // 2 - 4:h // 2 + 4, w // 2 - 4:w // 2 + 4].mean(dim=(0, 1))


def assert_color(actual, expected, tolerance=0.15):
    assert torch.allclose(actual, expected, atol=tolerance), f"{actual} != {expected}"


class TestNormalizeCropRect:
    def test_clamps_to_frame_and_even_aligns(self):
        assert normalize_crop_rect(10, 10, 100, 100, 64, 64) == (10, 10, 54, 54)
        assert normalize_crop_rect(0, 0, 33, 31, 64, 64) == (0, 0, 32, 30)

    def test_empty_and_full_frame_are_none(self):
        assert normalize_crop_rect(0, 0, 0, 0, 64, 64) is None
        assert normalize_crop_rect(0, 0, -2, 10, 64, 64) is None
        assert normalize_crop_rect(0, 0, 64, 64, 64, 64) is None

    def test_negative_origin_clamps_to_zero(self):
        assert normalize_crop_rect(-10, -10, 32, 32, 64, 64) == (0, 0, 32, 32)

    def test_full_frame_on_odd_source_is_none(self):
        assert normalize_crop_rect(0, 0, 65, 63, 65, 63) is None


class TestVideoFromComponentsCrop:
    def test_slices_images(self, quadrant_components):
        cropped = VideoFromComponents(quadrant_components).as_cropped(32, 0, 32, 32)
        components = cropped.get_components()
        assert components.images.shape == (2, 32, 32, 3)
        assert_color(components.images[0].mean(dim=(0, 1)), GREEN, tolerance=0.01)

    def test_noop_returns_self(self, quadrant_components):
        video = VideoFromComponents(quadrant_components)
        assert video.as_cropped(0, 0, 0, 0) is video
        assert video.as_cropped(0, 0, 64, 64) is video

    def test_slices_alpha(self, quadrant_components):
        alpha = torch.zeros(2, 64, 64)
        alpha[:, :32, :32] = 1.0
        with_alpha = VideoComponents(
            images=quadrant_components.images,
            frame_rate=quadrant_components.frame_rate,
            alpha=alpha,
        )
        components = VideoFromComponents(with_alpha).as_cropped(0, 0, 32, 32).get_components()
        assert components.alpha.shape == (2, 32, 32)
        assert components.alpha.min() == 1.0


class TestVideoFromFileCrop:
    def test_dimensions_and_components(self, src):
        cropped = VideoFromFile(src).as_cropped(0, 32, 32, 32)
        assert cropped.get_dimensions() == (32, 32)
        components = cropped.get_components()
        assert components.images.shape == (2, 32, 32, 3)
        assert_color(components.images[0].mean(dim=(0, 1)), BLUE)

    def test_save_streams_cropped_output(self, src, tmp_path):
        path = str(tmp_path / "cropped.mp4")
        VideoFromFile(src).as_cropped(32, 32, 32, 32).save_to(path)
        assert probe_dimensions(path) == (32, 32)
        assert_color(decoded_center_color(path), WHITE)

    def test_noop_returns_self(self, src):
        video = VideoFromFile(src)
        assert video.as_cropped(0, 0, 0, 0) is video
        assert video.as_cropped(0, 0, 64, 64) is video

    def test_nested_crop_composes_against_normalized_outer(self, src):
        outer = VideoFromFile(src).as_cropped(0, 0, 33, 31)
        assert outer.get_dimensions() == (32, 30)
        assert outer.as_cropped(0, 0, 32, 30) is outer

        inner = outer.as_cropped(0, 0, 16, 16)
        components = inner.get_components()
        assert components.images.shape == (2, 16, 16, 3)
        assert_color(components.images[0].mean(dim=(0, 1)), RED)

    def test_composes_with_trim(self, src, tmp_path):
        video = VideoFromFile(src).as_trimmed(0, 1 / 30, strict_duration=False).as_cropped(32, 0, 32, 32)
        components = video.get_components()
        assert components.images.shape == (1, 32, 32, 3)
        assert_color(components.images[0].mean(dim=(0, 1)), GREEN)

        path = str(tmp_path / "trim_crop.mp4")
        video.save_to(path)
        assert probe_dimensions(path) == (32, 32)
        assert_color(decoded_center_color(path), GREEN)

    def test_crop_survives_trim(self, src):
        video = VideoFromFile(src).as_cropped(32, 0, 32, 32).as_trimmed(0, 1 / 30, strict_duration=False)
        components = video.get_components()
        assert components.images.shape == (1, 32, 32, 3)
        assert_color(components.images[0].mean(dim=(0, 1)), GREEN)

    def test_crop_of_crop_composes(self, src):
        cropped = VideoFromFile(src).as_cropped(32, 0, 32, 64).as_cropped(0, 32, 32, 32)
        components = cropped.get_components()
        assert components.images.shape == (2, 32, 32, 3)
        assert_color(components.images[0].mean(dim=(0, 1)), WHITE)


def tag_rotation_90(path):
    """Patch the mp4 tkhd display matrix to a 90-degree rotation (phone-style metadata)"""
    with open(path, "rb") as f:
        data = bytearray(f.read())
    index = data.find(b"tkhd")
    assert index != -1
    matrix_offset = index + 4 + 40
    data[matrix_offset:matrix_offset + 36] = struct.pack(
        ">9i", 0, 65536, 0, -65536, 0, 0, 0, 0, 1073741824
    )
    with open(path, "wb") as f:
        f.write(bytes(data))


@pytest.fixture(scope="module")
def rotated_src(tmp_path_factory):
    """64x32 coded video (left RED, right GREEN) tagged with a 90-degree display
    matrix, so its display space is 32x64 with RED on top and GREEN below"""
    images = torch.zeros(2, 32, 64, 3)
    images[:, :, :32] = RED
    images[:, :, 32:] = GREEN
    path = str(tmp_path_factory.mktemp("video") / "rotated.mp4")
    VideoFromComponents(VideoComponents(images=images, frame_rate=Fraction(30))).save_to(path)
    tag_rotation_90(path)
    return path


class TestRotatedSourceCrop:
    def test_components_are_rotation_corrected(self, rotated_src):
        components = VideoFromFile(rotated_src).get_components()
        assert components.images.shape == (2, 64, 32, 3)

    def test_crop_applies_in_display_space(self, rotated_src):
        cropped = VideoFromFile(rotated_src).as_cropped(0, 32, 32, 32)
        assert cropped.get_dimensions() == (32, 32)
        components = cropped.get_components()
        assert components.images.shape == (2, 32, 32, 3)
        assert_color(components.images[0].mean(dim=(0, 1)), GREEN)

    def test_save_crops_in_display_space(self, rotated_src, tmp_path):
        path = str(tmp_path / "rotated_crop.mp4")
        VideoFromFile(rotated_src).as_cropped(0, 32, 32, 32).save_to(path)
        assert probe_dimensions(path) == (32, 32)
        assert_color(decoded_center_color(path), GREEN)
