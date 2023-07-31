from PIL import Image
import numpy as np
import eden

def get_image(image_path: str):
    nchannels = 0
    extension = image_path[-4:]
    if extension == ".png":
        nchannels = 4
    elif extension == ".jpg":
        nchannels = 3
    else:
        raise NameError("File name needs to finish either with .png or .jpg extensions.")
    
    image = Image.open(image_path)
    image_array = np.array(image)
    
    split_matrices = np.array_split(image_array, nchannels, axis=2)
    matrices = []
    dim1 = split_matrices[0].shape[0]
    dim2 = split_matrices[0].shape[1]
    for i in range(nchannels):
        matrices.append(eden.Matrix(split_matrices[i].reshape(dim1, dim2)))
        
    return matrices
    
