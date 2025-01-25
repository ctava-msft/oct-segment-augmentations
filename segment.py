"""
Segment module encapsulated the ThinEdgeSegmenter class.
"""
import os
import json
import numpy as np
from PIL import Image
import torch
from torch import nn
import logging

# ThinEdgeSegmenter class
class ThinEdgeSegmenter(nn.Module):
    """
    Class that performs thinedge sgementation using guassian, sobel, and thin kernels.
    """
    # Gaussian Kernel
    def get_gaussian_kernel(self, k=3, mu=0, sigma=1, normalize=True):
        """
        Create guassian kernel.
        """
        # compute a 1 dimension gaussian
        gaussian_1d = np.linspace(-1, 1, k)
        # compute a grid distance from center
        x, y = np.meshgrid(gaussian_1d, gaussian_1d)
        distance = (x ** 2 + y ** 2) ** 0.5
        # compute the 2 dimension gaussian
        gaussian_2d = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
        gaussian_2d = gaussian_2d / (2 * np.pi * sigma ** 2)
        # normalize part (mathematically)
        if normalize:
            gaussian_2d = gaussian_2d / np.sum(gaussian_2d)
        return gaussian_2d

    # Sobel Kernel
    def get_sobel_kernel(self, k=3):
        """
        Create sobel kernel.
        """
        range_space = np.linspace(-(k // 2), k // 2, k)
        # compute a grid the numerator and the axis-distances
        x, y = np.meshgrid(range_space, range_space)
        sobel_2d_numerator = x
        sobel_2d_denominator = x ** 2 + y ** 2
        sobel_2d_denominator[:, k // 2] = 1  # avoid division by zero
        sobel_2d = sobel_2d_numerator / sobel_2d_denominator
        return sobel_2d

    # Thin Kernels
    def get_thin_kernels(self):
        """
        Create thin kernels.
        """
        kernels = []
        for _ in range(8):
            kernel = np.zeros((3, 3), dtype=np.float32)
            # Placeholder for thinning kernels
            kernel[1, 1] = 1.
            kernels.append(kernel)
        return kernels
    
    # Save Outputs
    def save_outputs(self):
        """
        Save semgentation outputs.
        """
        # Create new image from thin_edges
        img = (self.thin_edges * 255).squeeze().detach().cpu().numpy().astype(np.uint8)
        if img.ndim == 3 and img.shape[0] == 1:
            img = img[0]  # Remove the channel dimension if it is 1
        image = Image.fromarray(img)
        # Save thin edges as image, numpy array and JSON
        image.save(os.path.join(self.output_dir, 'thin_edges.png'))
        thin_edges_np = self.thin_edges.squeeze().detach().cpu().numpy()
        np.save(os.path.join(self.output_dir, "thin_edges.npy"), thin_edges_np)
        with open(os.path.join(self.output_dir, "thin_edges.json"), 'w', encoding='utf-8') as f:
            json.dump(self.thin_edges.tolist(), f, indent=4)

    # Forward
    def forward(self, img, low_threshold=0.15, high_threshold=0.30, hysteresis=False):
        """
        Forward pass of the neural net.
        """
        self.forward_call_count += 1
        print(f'Forward called {self.forward_call_count} times')
        # set the steps tensors
        b, c, h, w = img.shape
        blurred = torch.zeros((b, c, h, w)).to(self.device)
        grad_x = torch.zeros((b, 1, h, w)).to(self.device)
        grad_y = torch.zeros((b, 1, h, w)).to(self.device)
        grad_magnitude = torch.zeros((b, 1, h, w)).to(self.device)
        grad_orientation = torch.zeros((b, 1, h, w)).to(self.device)
        for i in range(c):
            # apply gaussian filter
            blurred[:, i:i+1] = self.gaussian_filter(img[:, i:i+1])
            # apply sobel filter
            grad_x = grad_x + self.sobel_filter_x(blurred[:, i:i+1])
            grad_y = grad_y + self.sobel_filter_y(blurred[:, i:i+1])
        # thick edges
        grad_x, grad_y = grad_x / c, grad_y / c
        grad_magnitude = (grad_x ** 2 + grad_y ** 2) ** 0.5
        grad_orientation = torch.atan(grad_y / grad_x)
        grad_orientation = grad_orientation * (360 / np.pi) + 180 # convert to degree
        grad_orientation = torch.round(grad_orientation / 45) * 45  # keep a split by 45
        # # thin edges
        directional = self.directional_filter(grad_magnitude)
        # get indices of positive and negative directions
        positive_idx = (grad_orientation / 45) % 8
        self.thin_edges = grad_magnitude.clone()
        # # non maximum suppression direction by direction
        for pos_i in range(4):
            neg_i = pos_i + 4
            # get the oriented grad for the angle
            is_oriented_i = (positive_idx == pos_i) * 1
            is_oriented_i = is_oriented_i + (positive_idx == neg_i) * 1
            pos_directional = directional[:, pos_i]
            neg_directional = directional[:, neg_i]
            selected_direction = torch.stack([pos_directional, neg_directional])
            # get the local maximum pixels for the angle
            is_max = selected_direction.min(dim=0)[0] > 0.0
            is_max = torch.unsqueeze(is_max, dim=1)
            # apply non maximum suppression
            to_remove = (is_max == 0) * 1 * (is_oriented_i) > 0
            self.thin_edges[to_remove] = 0.0
        # thresholds
        if low_threshold is not None:
            low = self.thin_edges > low_threshold
            if high_threshold is not None:
                high = self.thin_edges > high_threshold
                # get black/gray/white only
                self.thin_edges = low * 0.5 + high * 0.5
                if hysteresis:
                    # get weaks and check if they are high or not
                    weak = (self.thin_edges == 0.5) * 1
                    weak_is_high = (self.hysteresis(self.thin_edges) > 1) * weak
                    self.thin_edges = high * 1 + weak_is_high * 1
                    # Only keep final edges with value 1
                    self.thin_edges[self.thin_edges < 1] = 0
                else:
                    self.thin_edges = low * 1
        final_coords = []
        for row in range(h):
            for col in range(w):
                if self.thin_edges[0, 0, row, col] == 1:
                    final_coords.append((row, col))
        self.thin_edges_coordinates = final_coords
        np.save(os.path.join(self.output_dir, "thin_edges_coordinates.npy"), final_coords)
        with open(os.path.join(self.output_dir, "thin_edges_coordinates.json"), 'w', encoding='utf-8') as f:
            json.dump(final_coords, f, indent=4)
        return blurred, grad_x, grad_y, grad_magnitude, grad_orientation, self.thin_edges_coordinates, self.thin_edges, self.layers_data

    # Constructor
    def __init__(self,
                 output_dir,
                 k_gaussian=3,
                 mu=0,
                 sigma=1,
                 k_sobel=3,
                 use_cuda=False):
        super(ThinEdgeSegmenter, self).__init__()
        self.output_dir = output_dir
        self.device = 'cuda' if use_cuda else 'cpu'
        self.forward_call_count = 0
        self.thin_edges_coordinates = []
        self.thin_edges = torch.zeros(1)
        #self.register_buffer('thin_edges', torch.zeros(1))
        self.layers_data = None
        # gaussian kernel
        gaussian_2d = self.get_gaussian_kernel(k_gaussian, mu, sigma)
        self.gaussian_filter = nn.Conv2d(in_channels=1,
                                         out_channels=1,
                                         kernel_size=k_gaussian,
                                         padding=k_gaussian // 2,
                                         bias=False)
        with torch.no_grad():
            self.gaussian_filter.weight[:] = torch.from_numpy(gaussian_2d)
        # sobel kernel
        sobel_2d = self.get_sobel_kernel(k_sobel)
        self.sobel_filter_x = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        self.sobel_filter_y = nn.Conv2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=k_sobel,
                                    padding=k_sobel // 2,
                                    bias=False)
        with torch.no_grad():
            self.sobel_filter_x.weight[:] = torch.from_numpy(sobel_2d)
            self.sobel_filter_y.weight[:] = torch.from_numpy(sobel_2d.T)
        # thin kernels
        thin_kernels = self.get_thin_kernels()
        directional_kernels = np.stack(thin_kernels)
        self.directional_filter = nn.Conv2d(in_channels=1,
                                            out_channels=8,
                                            kernel_size=thin_kernels[0].shape,
                                            padding=thin_kernels[0].shape[-1] // 2,
                                            bias=False)
        with torch.no_grad():
            self.directional_filter.weight[:, 0] = torch.from_numpy(directional_kernels)
        # hysteresis
        hysteresis = np.ones((3, 3)) + 0.25
        self.hysteresis = nn.Conv2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=3,
                                    padding=1,
                                    bias=False)
        with torch.no_grad():
            self.hysteresis.weight[:] = torch.from_numpy(hysteresis)
