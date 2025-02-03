#@title GradCAM: Gradient-weighted Class Activation Mapping

#@markdown Our gradCAM implementation registers a forward hook
#@markdown on the model at the specified layer. This allows us
#@markdown to save the intermediate activations and gradients
#@markdown at that layer.

#@markdown To visualize which parts of the image activate for
#@markdown a given caption, we use the caption as the target
#@markdown label and backprop through the network using the
#@markdown image as the input.
#@markdown In the case of CLIP models with resnet encoders,
#@markdown we save the activation and gradients at the
#@markdown layer before the attention pool, i.e., layer4.
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import filters
from torch import nn

#@title Helper functions

#@markdown Some helper functions for overlaying heatmaps on top
#@markdown of images and visualizing with matplotlib.



def generate_gradcam(attributions, image):
    """Computes Grad-CAM for the last vision layer in SigLIP."""
    print("generating gradcam...")

    print("attributions shape: ", attributions.shape)

    # Compute Grad-CAM
    cam = attributions.sum(dim=0)  # Ensure sum is applied correctly
    cam = F.relu(cam)
    cam -= cam.min()
    cam /= cam.max()
    cam = cam.squeeze().numpy()

    # Resize to match image size
    width, height = image.size  # Get image size from PIL Image
    cam = cv2.resize(cam, (width, height))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(image), 0.5, heatmap, 0.5, 0)
    
    return Image.fromarray(overlay)

