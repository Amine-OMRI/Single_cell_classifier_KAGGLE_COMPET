import os

import matplotlib.pyplot as plt
import numpy as np

import HPA_Cell_Segmentation.hpacellseg.cellsegmentator as cellsegmentator
from HPA_Cell_Segmentation.hpacellseg.utils import label_cell


NUC_MODEL = 'HPA_Cell_Segmentation/dpn_unet_nuclei_v1.pth'
CELL_MODEL = 'HPA_Cell_Segmentation/dpn_unet_cell_3ch_v1.pth'

segmentator = cellsegmentator.CellSegmentator(
    NUC_MODEL,
    CELL_MODEL,
    scale_factor=0.25,
    device='cuda',
    padding=False,
    multi_channel_model=True
)


TRAIN_SET = os.path.join("static", "images")


def build_image_names(image_id: str, folder: str = TRAIN_SET) -> tuple:
    # mt is the mitchondria
    mt = os.path.join(folder, image_id + '_red.png')
    
    # er is the endoplasmic reticulum
    er = os.path.join(folder, image_id + '_yellow.png')
    
    # nu is the nuclei
    nu = os.path.join(folder, image_id + '_blue.png')
    
    return [mt], [er], [nu], [[mt], [er], [nu]]


def segmentCell(image):
    # For nuclei
    nuc_segmentations = segmentator.pred_nuclei(image[2])
    
    # For full cells
    cell_segmentations = segmentator.pred_cells(image)
    
    # post-processing
    nuclei_mask, cell_mask = label_cell(nuc_segmentations[0], cell_segmentations[0])
    
    del nuc_segmentations; del cell_segmentations
    
    return nuclei_mask, cell_mask 


def plot_cell_segments(cell_mask, mt, er, nu):
    
    i = 0
    microtubule = plt.imread(mt[i])    
    endoplasmicrec = plt.imread(er[i])    
    nuclei = plt.imread(nu[i])
    img = np.dstack((microtubule, endoplasmicrec, nuclei))
    
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Image')
    plt.axis('off')
    image_name= "cell_image.png"
    plt.savefig(os.path.join("static", "images",image_name))

    plt.subplot(1, 3, 2)
    plt.imshow(cell_mask)
    plt.title('Mask')
    plt.axis('off')
    image_name= "mask.png"
    plt.savefig(os.path.join("static", "images",image_name))
    
    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.imshow(cell_mask, alpha=0.6)
    plt.title('Image + Mask')
    plt.axis('off')
    image_name= "mask_image.png"
    plt.savefig(os.path.join("static", "images",image_name))


def get_composite_mask(green_channel, cell_mask, nuclei_mask, border_color: list = None):

    def bbox(arr):
        rows = np.any(arr, axis=1)
        cols = np.any(arr, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return [rmin, rmax, cmin, cmax]

    def box_mask(img: np.array, y_start: int, y_end: int, x_start: int, x_end: int, border_size: int = 3):

        y_end += 1
        x_end += 1
        border_mask = np.zeros_like(img)
        border_mask[y_start:y_start + border_size, x_start:x_end] = 1
        border_mask[y_end - border_size:y_end, x_start:x_end] = 1
        border_mask[y_start:y_end, x_start: x_start + border_size] = 1
        border_mask[y_start:y_end, x_end - border_size:x_end] = 1
        return border_mask.astype(np.bool)

    if border_color is None:
        border_color = [67, 0, 77]

    bounding_boxes = []
    for i in range(1, cell_mask.max() + 1):
        bounding_boxes.append(bbox(cell_mask == i))
    rectangles = [box_mask(cell_mask, *bounding_box) for bounding_box in bounding_boxes]

    bin_cell, bin_nuc, bin_extra_nuc = map(lambda x: np.clip(x, 0, 1),
                                           [cell_mask, nuclei_mask, cell_mask - nuclei_mask])
    blue = np.where(green_channel == 0, bin_nuc * 255, 0).astype(np.uint8)
    red = np.where(green_channel == 0, bin_extra_nuc * 255, 0).astype(np.uint8)
    white = np.where(np.logical_and(green_channel == 0, np.logical_and(red == 0, blue == 0)), 255, 0).astype(np.uint8)

    blue += white
    red += white
    green_channel += white

    ret = np.stack([red, green_channel, blue], 2)
    for rectangle in rectangles:
        ret[rectangle] = border_color
    return ret


def check_and_convert_to_8_bits(arr):
    return arr if arr.dtype == np.uint8 else convert_16_to_8_bits(arr)


def convert_16_to_8_bits(arr: np.array):
    return (arr / 256).astype(np.uint8)
