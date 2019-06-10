import SimpleITK as sitk
import numpy as np
from utility_functions import processing

def read_nii(dir, spacing=(2.0, 2.0, 2.0)):
    sitk_dir = sitk.ReadImage(dir)
    sitk_dir = processing.resample_image(sitk_dir, out_spacing=spacing)
    sitk_dir = processing.zero_mean_unit_var(sitk_dir)
    return sitk.GetArrayFromImage(sitk_dir).T


def extract_centroid_info_from_lml(dir):
    centroids_file = open(dir, 'r')
    iter_centroids_file = iter(centroids_file)
    next(iter_centroids_file)

    labels = []
    centroids = []
    for centroid_line in iter_centroids_file:
        centroid_line_split = centroid_line.split()
        labels.append(centroid_line_split[1].split("_")[0])
        centroids.append(np.array(centroid_line_split[2:5]).astype(float))

    return labels, centroids