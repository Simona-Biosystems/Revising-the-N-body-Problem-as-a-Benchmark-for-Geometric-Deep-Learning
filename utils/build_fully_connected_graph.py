import torch


def build_graph_with_knn(loc, batch_size, num_nodes, device, num_neighbors):
    if num_neighbors is None:
        num_neighbors = num_nodes - 1

    if num_neighbors >= num_nodes:
        raise ValueError(
            "Graph cannot have more neighbors than there are nodes in simulation - 1"
        )

    # Precompute total number of nodes in all batches
    total_nodes = batch_size * num_nodes

    # Compute pairwise distances for all batches at once
    loc_reshaped = loc.view(
        batch_size, num_nodes, -1
    )  # Shape: (batch_size, num_nodes, num_features)

    # Efficiently compute pairwise distances for all batches
    dist_matrix = torch.cdist(
        loc_reshaped, loc_reshaped
    )  # Shape: (batch_size, num_nodes, num_nodes)

    # Get the k-nearest neighbors for each node (excluding self-loops)
    knn_indices = torch.topk(dist_matrix, k=num_neighbors + 1, largest=False).indices[
        :, :, 1:
    ]  # Shape: (batch_size, num_nodes, num_neighbors)

    # Create an edge index matrix for all batches
    row_indices = (
        torch.arange(num_nodes, device=device)
        .view(1, -1, 1)
        .expand(batch_size, num_nodes, num_neighbors)
    )

    # Add batch offsets to row_indices and knn_indices
    batch_offsets = torch.arange(0, total_nodes, num_nodes, device=device).view(
        batch_size, 1, 1
    )
    row_indices = row_indices + batch_offsets
    knn_indices = knn_indices + batch_offsets

    # Stack row and column indices into a single tensor for the edge index
    edge_index = torch.stack(
        [row_indices.flatten(), knn_indices.flatten()], dim=0
    )  # Shape: (2, batch_size * num_nodes * num_neighbors)
    return edge_index.to(device)
