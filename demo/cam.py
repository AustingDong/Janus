
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import filters
from torch import nn



# def generate_gradcam(attributions, image):
#     """Computes Grad-CAM for the last vision layer in SigLIP."""
#     print("generating gradcam...")

#     print("attributions shape: ", attributions.shape)

#     # Compute Grad-CAM
#     cam = attributions.sum(dim=0)  # Ensure sum is applied correctly
#     cam = F.relu(cam)
#     cam -= cam.min()
#     cam /= cam.max()
#     cam = cam.squeeze().to(float).detach().cpu().numpy()

#     # Resize to match image size
#     width, height = image.size  # Get image size from PIL Image
#     cam = cv2.resize(cam, (width, height))
#     heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
#     overlay = cv2.addWeighted(np.array(image), 0.5, heatmap, 0.5, 0)
    
#     return Image.fromarray(overlay)

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer):
        """
        model: your model that outputs patch tokens (e.g., shape (1,576,1024))
        target_layer: the layer at which to capture activations and gradients
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        # Forward hook to capture activations
        def forward_hook(module, input, output):
            self.activations = output.detach()
        # Backward hook to capture gradients
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        h_forward = self.target_layer.register_forward_hook(forward_hook)
        h_backward = self.target_layer.register_backward_hook(backward_hook)
        self.hook_handles.extend([h_forward, h_backward])

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()

    def __call__(self, input_tensor, target_token_idx=0):
        """
        input_tensor: image tensor of shape (1, C, H, W)
        target_token_idx: which patch token (0 to 575) to use for computing gradients.
                          You can experiment with different indices.
        """
        # Forward pass: model output shape will be (1, 576, 1024)
        output = self.model(**input_tensor)
        # Select a target scalar value from the token of interest.
        # For example, you can sum the activations for the target token.
        target = output[0, target_token_idx].sum()
        self.model.zero_grad()
        target.backward(retain_graph=True)
        # At this point, self.activations and self.gradients are populated.
        # Compute the weights by global average pooling the gradients over channels.
        # Both activations and gradients have shape (1, 576, 1024)
        weights = self.gradients.sum(dim=-1, keepdim=True)  # shape: (1, 576, 1)
        # Weighted combination: multiply activations by weights and sum over channels.
        cam = (weights * self.activations).sum(dim=-1)  # shape: (1, 576)
        cam = F.relu(cam)
        # Normalize CAM per sample (here, only one sample)
        cam_min = cam.min()
        cam_max = cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam  # shape: (1, 576)
    

def generate_gradcam(
    attributions, 
    image,
    size = (384, 384),
    alpha=0.5, 
    colormap=cv2.COLORMAP_JET, 
    aggregation='mean'
):
    """
    Generates a Grad-CAM heatmap overlay on top of the input image.

    Parameters:
      attributions (torch.Tensor): A tensor of shape (C, H, W) representing the
        intermediate activations or gradients at the target layer.
      image (PIL.Image): The original image.
      alpha (float): The blending factor for the heatmap overlay (default 0.5).
      colormap (int): OpenCV colormap to apply (default cv2.COLORMAP_JET).
      aggregation (str): How to aggregate across channels; either 'mean' or 'sum'.

    Returns:
      PIL.Image: The image overlaid with the Grad-CAM heatmap.
    """
    print("Generating Grad-CAM with attributions shape:", attributions.shape)

    # Aggregate the channel dimension to get a 2D map.

    # Apply ReLU to the aggregated map
    cam = F.relu(attributions)

    # Normalize the map to [0, 1]
    cam_min = cam.min()
    cam_max = cam.max()
    if cam_max - cam_min > 0:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = cam - cam_min  # This should result in a zero map if constant

    # Convert tensor to numpy array
    cam_np = cam.squeeze().detach().to(float).cpu().numpy()

    # Resize the cam to match the image size
    # width, height = image.size  # PIL image size is (width, height)
    width, height = size
    cam_resized = cv2.resize(cam_np, (width, height))

    # Convert the normalized map to a heatmap (0-255 uint8)
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    # OpenCV produces heatmaps in BGR, so convert to RGB for consistency
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Convert original image to a numpy array
    image_np = np.array(image)
    image_np = cv2.resize(image_np, (width, height))

    # Blend the heatmap with the original image
    overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)

    return Image.fromarray(overlay)

