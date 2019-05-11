import sys
import numpy as np


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
                                stack.append(next_point)
    return acc