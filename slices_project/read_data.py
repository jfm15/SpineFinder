import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

base_directory = '/homes/jfm15/SpineFinder/'
mask_directory = base_directory + 'masks'
spine_directory = '/vol/bitbucket2/jfm15/spine-1'

from utility_functions import opening_files


def get_slices():

    slices = []
    masks = []

    for patient_file in next(os.walk(mask_directory))[1]:
        for scan_number in next(os.walk(mask_directory + '/' + patient_file))[1]:

            masks_directory = '/'.join([mask_directory, patient_file, scan_number])
            cut_indices_loc = masks_directory + '/' + scan_number + '-sagittal-indices.npy'
            sagittal_slices_loc = masks_directory + '/' + scan_number + '-sagittal-slices.npy'

            cut_indices = np.load(cut_indices_loc)
            sagittal_slices = np.load(sagittal_slices_loc)

            #masked_data = np.ma.masked_where(sagittal_slices[0] == 0, sagittal_slices[0])

            scan_directory = '/'.join([spine_directory, patient_file, scan_number])
            scan_file_loc = scan_directory + '/' + scan_number + '.nii'

            scan = utility_functions.opening_files.read_nii(scan_file_loc)

            for i in range(0, cut_indices.shape[0]):
                slices.append(scan[cut_indices[i], :, :])
                masks.append(sagittal_slices[i])

            #plt.imshow(selected_slice.T, interpolation="none", aspect=8, origin='lower')
            #plt.imshow(masked_data.T, interpolation="none", aspect=8, origin='lower', cmap=cm.jet, alpha=0.3)

    patches_per_image = 10

    X = np.zeros([len(slices) * patches_per_image, 320, 40])
    Y = np.zeros([len(slices) * patches_per_image, 320, 40])

    patch_size = np.array([320, 40])

    i = 0
    for cur_slice, mask in zip(slices, masks):
        random_area = np.array(cur_slice.shape) - patch_size
        for _ in range(patches_per_image):
            top_left = np.random.rand(2) * random_area
            top_left = np.floor(top_left).astype(int)
            bottom_right = top_left + patch_size

            slice_patch = cur_slice[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
            mask_patch = mask[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

            X[i] = slice_patch
            Y[i] = mask_patch
            i += 1

    return X, Y
