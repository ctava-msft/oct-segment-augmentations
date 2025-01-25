"""
Module that kicks off the segmentation and morphing process.
"""
import argparse
import logging
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from morph import SegmentMorpher
from segment import ThinEdgeSegmenter

exp = 1
exp_i = 1
output_dir = f"_output-{exp}-{exp_i}"
os.makedirs(output_dir, exist_ok=True)
output_layers_dir = f"_output-{exp}-{exp_i}/layers"
os.makedirs(output_layers_dir, exist_ok=True)

# Main function
def main(image_path):
    """
    Main function that kicks off the segmentation and morphing process.
    Parameters:
        image_path (str): Path to the input image.
    Returns:
        N/A
    """
    # Set up logging
    logging.basicConfig(
        filename=os.path.join(output_dir, 'script.log'),
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console_logging = True  # Toggle on/off for console logging
    if (console_logging):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)
    # ThinEdge Detection and Layer Segmentation
    logging.info("ThinEdge Detection and Layer Segmentation")
    image = Image.open(image_path).convert('RGB')
    image.save(os.path.join(output_dir, os.path.basename(image_path)))
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)
    segmenter = ThinEdgeSegmenter(output_dir=output_dir, k_gaussian=3, mu=0, sigma=1, k_sobel=3, use_cuda=False)
    _, _, _, _, _, thin_edges_coordinates, thin_edges, layers_data = segmenter(image_tensor)
    #_, _, _, _, _, thin_edges = segmenter(image_tensor)
    segmenter.save_outputs()
    # Segment Morphing
    logging.info("Segment Morphing")
    thin_edges_np = thin_edges.detach().numpy()
    image_data = np.array(image)
    #morpher = SegmentMorpher(output_dir=output_dir, thin_edges=thin_edges_np)
    morpher = SegmentMorpher(output_dir=output_dir,image_data=image_data, thin_edges_coordinates=thin_edges_coordinates,thin_edges=thin_edges_np, layers_data=layers_data)
    #(self,output_dir,image_data,thin_edges_coordinates,thin_edges,layers_data):
    morpher.generate_outputs()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined ThinEdgeSegmenter and GridMeshMorpher")
    parser.add_argument('image_path', type=str, help='Path to the input image')
    args = parser.parse_args()
    main(args.image_path)
