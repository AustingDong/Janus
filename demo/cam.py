import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import filters
from torch import nn


class AttentionGuidedCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = []
        self.activations = []
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """ Registers hooks to extract activations and gradients from ALL attention layers. """
        for layer in self.target_layers:
            self.hooks.append(layer.register_forward_hook(self._forward_hook))
            self.hooks.append(layer.register_backward_hook(self._backward_hook))

    def _forward_hook(self, module, input, output):
        """ Stores attention maps (before softmax) """
        self.activations.append(output)

    def _backward_hook(self, module, grad_in, grad_out):
        """ Stores gradients """
        self.gradients.append(grad_out[0])

    
    def remove_hooks(self):
        """ Remove hooks after usage. """
        for hook in self.hooks:
            hook.remove()

    # def normalize(self, arr):
    #     arr = F.relu(arr)
    #     arr = arr - arr.min()
    #     arr = arr / (arr.max() - arr.min())
    #     return arr
    
    def generate_cam(self, input_tensor, class_idx=None):
        raise NotImplementedError




class AttentionGuidedCAMClip(AttentionGuidedCAM):
    def __init__(self, model, target_layers):
        self.target_layers = target_layers
        super().__init__(model)
    
    def generate_cam(self, input_tensor, class_idx=None, visual_pooling_method="CLS"):
        """ Generates Grad-CAM heatmap for ViT. """
        self.model.zero_grad()
        
        # Forward pass
        output_full = self.model(**input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output_full.logits, dim=1).item()

        if visual_pooling_method == "CLS":
            output = output_full.image_embeds
        elif visual_pooling_method == "avg":
            output = self.model.visual_projection(output_full.vision_model_output.last_hidden_state).mean(dim=1)
        else:
            # project -> pooling
            output, _ = self.model.visual_projection(output_full.vision_model_output.last_hidden_state).max(dim=1)

            # pooling -> project
            # output_mx, _ = output_full.vision_model_output.last_hidden_state.max(dim=1)
            # output = self.model.visual_projection(output_mx)

        output.backward(output_full.text_embeds[class_idx:class_idx+1], retain_graph=True)

        # Aggregate activations and gradients from ALL layers
        print(self.activations, self.gradients)
        cam_sum = None
        for act, grad in zip(self.activations, self.gradients):

            # act = torch.sigmoid(act[0])
            act = F.relu(act[0])
            
            grad_weights = grad.mean(dim=-1, keepdim=True)
            

            print("act shape", act.shape)
            print("grad_weights shape", grad_weights.shape)
            
            # cam = (act * grad_weights).sum(dim=-1)  # Weighted activation map
            cam, _ = (act * grad_weights).max(dim=-1)
            # cam, _ = grad_weights.max(dim=-1)
            # cam = self.normalize(cam)
            print(cam.shape)

            # Sum across all layers
            if cam_sum is None:
                cam_sum = cam
            else:
                cam_sum += cam  
                

        # Normalize
        cam_sum = F.relu(cam_sum)
        cam_sum = cam_sum - cam_sum.min()
        cam_sum = cam_sum / (cam_sum.max() - cam_sum.min())

        # thresholding
        cam_sum = cam_sum.to(torch.float32)
        percentile = torch.quantile(cam_sum, 0.2)  # Adjust threshold dynamically
        cam_sum[cam_sum < percentile] = 0

        # Reshape
        print("cam_sum shape: ", cam_sum.shape)
        cam_sum = cam_sum[0, 1:]

        num_patches = cam_sum.shape[-1]  # Last dimension of CAM output
        grid_size = int(num_patches ** 0.5)
        print(f"Detected grid size: {grid_size}x{grid_size}")
        
        cam_sum = cam_sum.view(grid_size, grid_size).detach()

        return cam_sum, output_full, grid_size


class AttentionGuidedCAMJanus(AttentionGuidedCAM):
    def __init__(self, model, target_layers):
        self.target_layers = target_layers
        super().__init__(model)


    def generate_cam(self, input_tensor, tokenizer, temperature, top_p, class_idx=None, visual_pooling_method="CLS"):
        """ Generates Grad-CAM heatmap for ViT. """
        self.model.zero_grad()
        
        # Forward pass
        image_embeddings, inputs_embeddings, outputs = self.model(input_tensor, tokenizer, temperature, top_p)


        input_ids = input_tensor.input_ids

        # Pooling
        if visual_pooling_method == "CLS":
            image_embeddings_pooled = image_embeddings[:, 0, :]
        elif visual_pooling_method == "avg":
            image_embeddings_pooled = image_embeddings[:, 1:, :].mean(dim=1) # end of image: 618
        elif visual_pooling_method == "max":
            image_embeddings_pooled, _ = image_embeddings[:, 1:, :].max(dim=1)

        print("image_embeddings_shape: ", image_embeddings_pooled.shape)
        


        inputs_embeddings_pooled = inputs_embeddings[:, 620: -4].mean(dim=1)




        # inputs_embeddings_pooled = inputs_embeddings[
        #     torch.arange(inputs_embeddings.shape[0], device=inputs_embeddings.device),
        #     input_ids.to(dtype=torch.int, device=inputs_embeddings.device).argmax(dim=-1),
        # ]


        # Backpropagate to get gradients
        image_embeddings_pooled.backward(inputs_embeddings_pooled, retain_graph=True)
        # similarity = F.cosine_similarity(image_embeddings_mean, inputs_embeddings_mean, dim=-1)
        # similarity.backward()

        # Aggregate activations and gradients from ALL layers
        cam_sum = None
        for act, grad in zip(self.activations, self.gradients):
            # act = torch.sigmoid(act)
            act = F.relu(act[0])
 

            # Compute mean of gradients
            grad_weights = grad.mean(dim=-1, keepdim=True)

            print("act shape", act.shape)
            print("grad_weights shape", grad_weights.shape)

            cam, _ = (act * grad_weights).max(dim=-1)
            print(cam.shape)

            # Sum across all layers
            if cam_sum is None:
                cam_sum = cam
            else:
                cam_sum += cam  

        # Normalize
        cam_sum = F.relu(cam_sum)
        cam_sum = cam_sum - cam_sum.min()
        cam_sum = cam_sum / cam_sum.max()

        # thresholding
        cam_sum = cam_sum.to(torch.float32)
        percentile = torch.quantile(cam_sum, 0.2)  # Adjust threshold dynamically
        cam_sum[cam_sum < percentile] = 0

        # Reshape
        # if visual_pooling_method == "CLS":
        cam_sum = cam_sum[0, 1:]
        print("cam_sum shape: ", cam_sum.shape)
        num_patches = cam_sum.shape[-1]  # Last dimension of CAM output
        grid_size = int(num_patches ** 0.5)
        print(f"Detected grid size: {grid_size}x{grid_size}")

        # Fix the reshaping step dynamically
        
        cam_sum = cam_sum.view(grid_size, grid_size)


        return cam_sum, grid_size





def generate_gradcam(
    cam, 
    image,
    size = (384, 384),
    alpha=0.5, 
    colormap=cv2.COLORMAP_JET, 
    aggregation='mean',
    normalize=True
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
    print("Generating Grad-CAM with shape:", cam.shape)

    if normalize:
        cam_min, cam_max = cam.min(), cam.max()
        cam = cam - cam_min
        cam = cam / (cam_max - cam_min)
    # Convert tensor to numpy array
    cam = torch.nn.functional.interpolate(cam.unsqueeze(0).unsqueeze(0), size=size, mode='bilinear').squeeze()
    cam_np = cam.squeeze().detach().cpu().numpy()

    # Apply Gaussian blur for smoother heatmaps
    cam_np = cv2.GaussianBlur(cam_np, (5,5), sigmaX=0.8)

    # Resize the cam to match the image size
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

