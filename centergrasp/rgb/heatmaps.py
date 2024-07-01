import cv2
import numpy as np
from typing import List, Tuple
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy.stats import multivariate_normal


def heatmap_from_segmentation(segmentation: np.ndarray, indices: List[int]) -> Tuple[np.ndarray]:
    binary_masks = binary_masks_from_segmentation(segmentation, indices)
    heatmap, simple_bmasks = heatmap_from_bmasks(binary_masks)
    return heatmap, simple_bmasks


def binary_masks_from_segmentation(segmentation: np.ndarray, indices: List[int]):
    binary_masks = np.array([segmentation == idx for idx in indices])
    return binary_masks


def heatmap_from_bmask(mask: np.ndarray, peak_concentration: float = 1) -> np.ndarray:
    heatmap = np.zeros(mask.shape)
    if np.sum(mask) < 2:
        return heatmap
    coords = np.indices(mask.shape)
    coords = coords.reshape([2, -1]).T
    mask_f = mask.flatten()
    indices = coords[np.where(mask_f > 0)]
    mean_value = np.floor(np.average(indices, axis=0))
    cov = np.cov((indices - mean_value).T)
    cov = cov * peak_concentration
    try:
        multi_var = multivariate_normal(mean=mean_value, cov=cov)  # type: ignore
    except np.linalg.LinAlgError:
        # Matrix not invertible, return
        return heatmap
    density = multi_var.pdf(coords)
    heatmap[coords[:, 0], coords[:, 1]] = density
    heatmap = heatmap / np.max(heatmap)
    return heatmap


def heatmap_from_bmasks(bmasks: np.ndarray, peak_concentration: float = 1) -> Tuple[np.ndarray]:
    # Heatmap
    heatmaps = np.array([heatmap_from_bmask(mask, peak_concentration) for mask in bmasks])
    heatmap = np.max(heatmaps, axis=0)
    # Simple bmasks
    simple_bmasks = np.zeros_like(bmasks, dtype=np.uint8)
    heatmap_argmax = np.argmax(heatmaps, axis=0)
    for i in range(len(heatmaps)):
        simple_bmasks[i] = heatmap_argmax == i
    simple_bmasks = np.where(heatmap > 0.25, simple_bmasks, 0)
    return heatmap, simple_bmasks


def extract_peaks_from_heatmap(heatmap, min_distance=20, min_confidence=0.75, debug=False):
    # Alternative in `simnet.lib.net.post_processing.pose_outputs.extract_peaks_from_centroid_nms`?
    assert isinstance(heatmap, np.ndarray)
    # Make sure we operate in the 0..1 range
    if heatmap.dtype == np.uint8:
        heatmap = np.copy(heatmap).astype(float) / 255.0
    peaks = peak_local_max(
        heatmap,
        min_distance=min_distance,
        threshold_abs=min_confidence,
        exclude_border=False,
    )
    return peaks


def binary_masks_from_heatmap(heatmap, peaks, threshold=0.25):
    # From https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html#sphx-glr-auto-examples-segmentation-plot-watershed-py
    peaks_mask = np.zeros(heatmap.shape, dtype=np.uint8)
    for i, peak in enumerate(peaks):
        peaks_mask[tuple(peak)] = i + 1
    instances_mask = watershed(-heatmap, peaks_mask, mask=heatmap > threshold)
    binary_masks = [
        np.where(instances_mask == i, 1, 0).astype(np.uint8) for i in range(1, len(peaks) + 1)
    ]
    return binary_masks


def visualize_heatmap(rgb, heatmap: np.ndarray, with_peaks: bool = False):
    if heatmap.dtype == np.float32 or heatmap.dtype == np.float64:
        heatmap = (heatmap * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    super_imposed_img = cv2.addWeighted(colored_heatmap, 0.4, rgb, 0.6, 0)
    # cv2.imshow("super_imposed_img", super_imposed_img)
    # cv2.waitKey(0)
    if with_peaks:
        peaks = extract_peaks_from_heatmap(heatmap)
        for peak in peaks:
            cv2.circle(
                super_imposed_img, center=(int(peak[1]), int(peak[0])), radius=5, color=(0, 0, 255)
            )
    rgb_out = cv2.cvtColor(super_imposed_img, cv2.COLOR_BGR2RGB)
    return rgb_out
