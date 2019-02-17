from scipy.ndimage.interpolation import zoom
import numpy as np
from tensorflow.keras import backend as K


def grad_cam(input_model, image, cls, layer_name, size, pad):
    """
    GradCAM method for visualizing input saliency.
    based on https://github.com/totti0223/gradcamplusplus
    """
    y_c = input_model.output[0, cls]
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]

    gradient_function = K.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    cam = np.maximum(cam, 0)
    padded = np.zeros((cam.shape[0] + pad, cam.shape[1] + pad))
    padded[pad // 2:pad // 2 + cam.shape[0], pad // 2:pad // 2 + cam.shape[1]] = cam
    cam = zoom(padded, size / padded.shape[0])
    cam = cam / cam.max()
    return cam

