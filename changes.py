    
    # Create a mesh grid based on the non-zero edges in thin_edges_np
    def create_segment_meshes(self, thin_edges_np, grid_density=20):
        """
        Create a mesh grid based on the non-zero edges in thin_edges_np.
        """
        thin_edges_np = thin_edges_np.detach().numpy()
        mesh_points = []
        rows, cols = thin_edges_np.shape[:2]
        # Collect all non-zero positions as potential grid points
        condition = np.any(thin_edges_np > 0, axis=2)
        ys, xs = np.where(condition)[:2]
        for x, y in zip(xs, ys):
            # Optionally apply grid_density spacing
            if x % grid_density == 0 and y % grid_density == 0:
                mesh_points.append((x, y))
        # Guarantee coverage at boundaries
        for y in range(0, rows, grid_density):
            for x in range(0, cols, grid_density):
                mesh_points.append((x, y))
        self.source_mesh_points = list(set(mesh_points))
        self.target_mesh_points = list(set(mesh_points))

    # Extract Layers Data
    def extract_layers_data(self, img, thin_edges):
        """
        Extract pixel data between the y-coordinates where thin_edges is positive.
        """
        layers_data = []
        thin_edges_np = thin_edges.squeeze().detach().cpu().numpy()
        thin_edges_np = (self.thin_edges * 255).squeeze().detach().cpu().numpy().astype(np.uint8)
        if thin_edges_np.ndim == 3 and thin_edges_np.shape[0] == 1:
            thin_edges_np = thin_edges_np[0]  # Remove the channel dimension if it is 1
        print(thin_edges_np.shape)

        img_np = img.squeeze().detach().cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        if img_np.ndim == 3 and img_np.shape[0] == 1:
            img_np = img_np[0]  # Remove the channel dimension if it is 1
        elif img_np.ndim == 3:
            img_np = img_np.transpose(1, 2, 0)  # Transpose to (H, W, C) format
        print(img_np.shape)

        #img_np = img.squeeze().detach().cpu().numpy()
        # identify boundaries for each row
        for col in range(thin_edges_np.shape[1]):
            # get the y-coordinates where thin_edges is positive
            boundary_rows = np.where(thin_edges_np[:, col] == 255)[0]
            # pair up adjacent boundary rows (y1 (current row), y2 (next row))
            for i in range(len(boundary_rows) - 1):
                y1, y2 = boundary_rows[i], boundary_rows[i + 1]
                # extract the segment slice
                segment_slice = img_np[y1:y2, :]
                layers_data.append(segment_slice)
                segment_slice = (segment_slice * 255).astype(np.uint8)  # Convert to valid image format
                image = Image.fromarray(segment_slice)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image.save(os.path.join(self.output_dir, 'layers', f'edges_{i}.png'))
        return layers_data



import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, 