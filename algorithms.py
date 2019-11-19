import mahotas
import numpy as np
import skimage.external.tifffile as tifffile
from scipy import ndimage as ndi
from skimage import filters
from skimage import (img_as_ubyte, img_as_float, img_as_int)
from skimage import io
from skimage import color
import skimage.morphology as morph
from skimage.morphology import watershed
import json

def segment_conidia(input_file, output_file):
    img = io.imread(input_file)
    img = img_as_float(img)
    img = filters.gaussian(img, 1)

    thresholded = img_as_ubyte(img > filters.threshold_otsu(img))
    thresholded = mahotas.close_holes(thresholded)

    distance = ndi.distance_transform_edt(thresholded)
    local_maxi = distance == morph.dilation(distance, morph.square(5))
    local_maxi[thresholded == 0] = 0
    markers = ndi.label(local_maxi)[0]

    labels = watershed(thresholded, markers, mask=thresholded)
    tifffile.imsave(output_file, img_as_int(labels), compress=5)
    # tifffile.imsave(output_file + ".vis.tif", img_as_ubyte(color.label2rgb(labels, bg_label=0)), compress=5)


def filter_conidia(input_file, output_file):
    # Maximum area of a conidium
    min_area = np.pi * 5 ** 2
    max_area = np.pi * 12 ** 2

    img = io.imread(input_file)
    keys, counts = np.unique(img, return_counts=True)

    for i in range(len(counts)):
        key = int(keys[i])
        count = counts[i]
        if key > 0:
            if count < min_area or count > max_area:
                img[img == key] = 0

    tifffile.imsave(output_file, img_as_int(img), compress=5)



def quantify_conidia(label_dir, output_file, experiments):
    data = {}
    for experiment in experiments:
        img = io.imread(label_dir + "/" + experiment + ".tif")
        data[experiment] = len(np.unique(img)) - 1
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)