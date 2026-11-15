import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates


def align_shell(path):
    pixels = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255
    green = (pixels[:, :, 1] > pixels[:, :, 0] * 1.12) & (pixels[:, :, 1] > pixels[:, :, 2] * 1.12) & (pixels[:, :, 1] > 0.05)
    green[:, :100] = False
    green[:, 410:] = False
    rows, columns = np.nonzero(green)
    coordinates = np.column_stack([rows, columns])
    center = coordinates.mean(axis=0)
    _, axes = np.linalg.eigh(np.cov(coordinates.T))
    vertical = axes[:, -1]
    if vertical[0] < 0:
        vertical = -vertical
    horizontal = np.array([-vertical[1], vertical[0]])
    projected = (coordinates - center) @ np.column_stack([vertical, horizontal])
    top, bottom = np.quantile(projected[:, 0], [0.001, 0.999])
    left, right = np.quantile(projected[:, 1], [0.001, 0.999])
    rows, columns = np.meshgrid(np.linspace(top, bottom, 160), np.linspace(left, right, 56), indexing='ij')
    grid = center[:, None, None] + vertical[:, None, None] * rows + horizontal[:, None, None] * columns
    return np.stack([map_coordinates(pixels[:, :, channel], grid, order=1, mode='nearest') for channel in range(3)], axis=-1)
