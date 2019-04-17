def save_2d_dense_labelling(path, cut_indices,  dense_labellings):
    print(path + ".txt")
    file = open(path + ".txt", "w+")
    file.write(str(len(dense_labellings)) + "\n")
    file.write(str(dense_labellings[0].shape) + "\n")
    for idx, dense_labelling in zip(cut_indices, dense_labellings):
        file.write("cut-" + str(idx) + "\n")
        for row in dense_labelling:
            for val in row:
                file.write(str(int(val)))
            file.write("\n")
    file.close()
