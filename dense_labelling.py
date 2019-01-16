import numpy as np
import helper_functions as hf


def generate_dense_labelling(image, real_centroids, radii, scales):

    dense_labelling = np.zeros(image.shape)

    for idx, real_centroid in enumerate(real_centroids):
        # we want to check every pixel inside the square where the lengths of the sides are 2 * radii
        real_lower_bounds = real_centroid - radii
        real_upper_bounds = real_centroid + radii

        idx_lower_bounds = hf.real_to_indexes(real_lower_bounds, scales)
        idx_upper_bounds = hf.real_to_indexes(real_upper_bounds, scales)

        idx_lower_bounds = np.maximum(idx_lower_bounds, 0)
        idx_upper_bounds = np.minimum(idx_upper_bounds, image.shape)

        for t in range(idx_lower_bounds[0], idx_upper_bounds[0]):
            for s in range(idx_lower_bounds[1], idx_upper_bounds[1]):
                for v in range(idx_lower_bounds[2], idx_upper_bounds[2]):
                    image_point = [t, s, v]
                    real_point = hf.indexes_to_real(image_point, scales)
                    if np.linalg.norm(real_point-real_centroid) < radii:
                        dense_labelling[t, s, v] = idx + 1

        print(idx)

    return dense_labelling