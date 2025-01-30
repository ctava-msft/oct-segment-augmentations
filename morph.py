"""
Morph module encapsulated the MeshSegmentMorpher class.
"""
import json
import os
import numpy as np
from PIL import Image
import torchio as tio

# SegmentMorpher class
class SegmentMorpher():
    """
    Class that morphs images based on segmentation data.
    """
    # Constructor
    def __init__(self,output_dir,image_data,thin_edges_coordinates,thin_edges,layers_data):
        super(SegmentMorpher, self).__init__()
        self.output_dir = output_dir
        self.image_data_orig = image_data
        self.image_data_morphed = image_data.copy()
        self.thin_edges_coordinates = thin_edges_coordinates
        self.thin_edges = thin_edges
        self.layers_data = layers_data

    ## Compare the original and morphed images
    def compare_images(self):
        """
        Compare image_data_orig and image_data_morphed and write differences to a file.
        """
        difference_image = np.abs(self.image_data_orig.astype(np.float32) - self.image_data_morphed.astype(np.float32))
        threshold = 1e-5
        differences = difference_image > threshold
        if np.any(differences):
            np.save(os.path.join(self.output_dir, "morphed_differences.npy"), differences)
            with open(os.path.join(self.output_dir, "morphed_differences.json"), 'w', encoding='utf-8') as f:
                json.dump(differences.tolist(), f, indent=4)
            difference_visual = (difference_image * differences).astype(np.uint8) * 255
            image = Image.fromarray(difference_visual)
            image.save(os.path.join(self.output_dir, 'morphed_differences.png'))
            # Save x,y coordinates of changed pixels
            changed_pixels = np.argwhere(differences)
            coordinates = [{"x": int(x), "y": int(y)} for y, x, *rest in changed_pixels]
            with open(os.path.join(self.output_dir, "morphed_differences_coordinates.json"), 'w', encoding='utf-8') as f:
                json.dump(coordinates, f, indent=4)
        else:
            print("No differences found.")

    # Apply random affine transformation to the image within layer coordinates
    def apply_random_affine_within_layer_coordinates(self, scales=(0.9, 1.2), degrees=5, translation_pixels=10):
        """
        Apply tio.RandomAffine transformation with vertical translation
        """
        ### Transforms Definitions - Start
        # transforms_dict = {
        #     tio.RandomAffine(
        #         scales=scales,
        #         degrees=degrees,
        #         translation=(0, translation_pixels)# / self.image_data_morphed.shape[0])
        #     ): 0.70,
        #     tio.RandomElasticDeformation(): 0.20,
        #     tio.RandomFlip(): 0.10
        # }
        # transform = tio.Compose(transforms_dict)
        # transform = tio.RandomAffine(
        #         scales=scales,
        #         degrees=degrees,
        #         translation=(0, translation_pixels)# / self.image_data_morphed.shape[0])
        #     )
        transform = tio.RandomElasticDeformation()
        ### Transforms Definitions - End
        ## Apply transformation only within thin edges coordinates
        mask = np.zeros(self.image_data_morphed.shape[:2], dtype=bool)
        for (y, x) in self.thin_edges_coordinates:
            mask[y, x] = True
        region = self.image_data_orig.copy()
        region[~mask] = 0
        tensor = np.expand_dims(region, axis=0)
        scalar_image = tio.ScalarImage(tensor=tensor)
        transformed = transform(scalar_image)
        transformed_region = transformed.tensor.squeeze(0).numpy()
        self.image_data_morphed[mask] = transformed_region[mask]
        np.save(os.path.join(self.output_dir, "morphed_image.npy"), self.image_data_morphed)
        image = Image.fromarray(self.image_data_morphed.astype(np.uint8))
        image.save(os.path.join(self.output_dir, 'morphed_image.png'))

    # generated morphed images
    def generate_outputs(self):
        """
        Generate morphed images.
        Parameters:
            N/A
        Returns:
            N/A
        """
        # Apply random affine transformation to the image within layer coordinates
        self.apply_random_affine_within_layer_coordinates(translation_pixels=100)
        # Compare the original and morphed images
        self.compare_images()
