import sys
import numpy as np
from utility_functions.labels import LABELS, VERTEBRAE_SIZES


# takes in a volume of predictions (0 and 1s) and takes out all but the largest island of points
def crop_labelling(predictions):
    width, height, depth = predictions.shape
    explored = {}
    largest_island = []
    for i in range(width):
        for j in range(height):
            for k in range(depth):
                point = (i, j, k)
                current_island = get_island(point, explored, predictions)
                if len(largest_island) < len(current_island):
                    largest_island = current_island

    new_predictions = np.zeros(predictions.shape)
    for point in largest_island:
        i, j, k = point
        new_predictions[i, j, k] = 1

    largest_island_np = np.array(largest_island)
    # print(predictions.shape)
    # print(largest_island_np.shape)
    i_min = np.min(largest_island_np[:, 0])
    i_max = np.max(largest_island_np[:, 0])
    j_min = np.min(largest_island_np[:, 1])
    j_max = np.max(largest_island_np[:, 1])
    k_min = np.min(largest_island_np[:, 2])
    k_max = np.max(largest_island_np[:, 2])
    print(new_predictions.shape, i_max - i_min, j_max - j_min, k_max - k_min)
    bounds = (i_min, i_max, j_min, j_max, k_min, k_max)
    return bounds, new_predictions


'''
def get_island(point, explored, predictions):
    i, j, k = point
    if point in explored:
        return []

    explored[point] = True

    if predictions[i, j, k] == 0:
        return []

    acc = [point]
    for i_add in range(-1, 2):
        for j_add in range(-1, 2):
            for k_add in range(-1, 2):
                if i_add != 0 or j_add != 0 or k_add != 0:
                    next_point = (i + i_add, j + j_add, k + k_add)
                    acc += get_island(next_point, explored, predictions)
    return acc
'''


# https://www.geeksforgeeks.org/iterative-depth-first-traversal/
def get_island(point, explored, predictions):
    stack = [point]
    acc = []
    while len(stack) > 0:
        curr_point = stack.pop(-1)
        i, j, k = curr_point
        if curr_point not in explored:
            explored[curr_point] = True
            if predictions[i, j, k]:
                acc.append(curr_point)
                for i_add in range(-1, 2):
                    for j_add in range(-1, 2):
                        for k_add in range(-1, 2):
                            if i_add != 0 or j_add != 0 or k_add != 0:
                                next_point = (i + i_add, j + j_add, k + k_add)
                                if np.all(np.greater_equal(next_point, np.zeros(3)))\
                                        and np.all(np.less(next_point, predictions.shape)):
                                    stack.append(next_point)
    return acc


# NOTE old function
'''
def densely_label(labels, volume_shape, centroid_indexes, spacing, radius, use_labels):

    diameter_in_pixels = radius / np.array(spacing)
    radius_in_pixels = ((diameter_in_pixels - np.ones(3)) / 2.0).astype(int)

    dense_labelling = np.zeros(volume_shape)

    upper_clip = volume_shape - np.ones(3)

    for label, centroid_idx in zip(labels, centroid_indexes):

        corner_a = centroid_idx - radius_in_pixels
        corner_a = np.clip(corner_a, a_min=np.zeros(3), a_max=upper_clip).astype(int)
        corner_b = centroid_idx + radius_in_pixels
        corner_b = np.clip(corner_b, a_min=np.zeros(3), a_max=upper_clip).astype(int)

        label_value = 1
        if use_labels:
            if label == 'L6':
                print("has L6 vertebrae")
            label_value = LABELS.index(label)

        dense_labelling[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1], corner_a[2]:corner_b[2]] = label_value

    return dense_labelling
'''


def densely_label(volume_shape, labels, centroids, spacing, use_labels):
    labelling = np.zeros(volume_shape)

    # do middle centroids
    for i, label in enumerate(labels[1:-1]):
        a = (centroids[i] + centroids[i + 1]) / 2.0
        b = (centroids[i + 1] + centroids[i + 2]) / 2.0
        create_tube(a, b, VERTEBRAE_SIZES[label], spacing, labelling)

    # do first centroid
    b = (centroids[0] + centroids[1]) / 2.0
    a = centroids[0] - (b - centroids[0])
    a = np.clip(a, a_min=np.zeros(3), a_max=volume_shape - np.ones(3)).astype(int)
    create_tube(a, b, VERTEBRAE_SIZES[labels[0]], spacing, labelling)

    # do last centroid
    b = (centroids[-2] + centroids[-1]) / 2.0
    a = centroids[-1] - (b - centroids[-1])
    a = np.clip(a, a_min=np.zeros(3), a_max=volume_shape - np.ones(3)).astype(int)
    create_tube(a, b, VERTEBRAE_SIZES[labels[-1]], spacing,  labelling)

    return labelling


def create_tube(a, b, diameter, spacing, labelling):
    dist = np.linalg.norm(b - a)
    spline = np.round(np.linspace(a, b, num=np.round(dist).astype(int) * 2)).astype(int)
    radius = np.round((diameter / 2.0) / spacing[0]).astype(int)
    for center_point in spline:
        for x in range(-radius - 1, radius + 1):
            for y in range(-radius, radius):
                point = center_point + np.array([x, y, 0])
                point = np.clip(point, a_min=np.zeros(3), a_max=labelling.shape - np.ones(3)).astype(int)
                if np.linalg.norm(np.array([x, y])) <= radius + 1:
                    labelling[point[0], point[1], point[2]] = 1