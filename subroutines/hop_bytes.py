import numpy as np

def hop_matrix_from_coords(torus_shape, phys_nodes):
    """
    Compute a matrix of hop counts between nodes in a sub-torus using NumPy.

    Args:
        torus_shape (tuple): Dimensions of the torus (e.g., (24, 23, 24)).
        phys_nodes (list of tuples): List of node coordinates in the torus, sorted by IDs.

    Returns:
        np.array: A matrix where element (i, j) is the hop count between nodes i and j.
    """
    # Convert the list of coordinates to a NumPy array
    coords = np.array(phys_nodes)

    # Compute pairwise differences for all dimensions
    deltas = np.abs(coords[:, np.newaxis, :] - coords[np.newaxis, :, :])

    # Apply torus wrap-around distances
    wrap_distances = np.minimum(deltas, np.array(torus_shape) - deltas)

    # Sum distances across dimensions to get the Manhattan distance (hop count)
    hop_matrix = wrap_distances.sum(axis=2)

    return hop_matrix

def compute_hop_bytes(mat, sigma, h, h_full):
    # Map processes to processor distances using sigma
    processor_distances = h[sigma[:, None], sigma[None, :]]  # Map 2048x2048 to processor distances

    
    if h_full is not None and  np.array_equal(processor_distances, h_full) == False:
        print("======================> Error arrays not equal!")
    # Element-wise multiply the communication matrix with the processor distances
    hop_bytes = np.sum(mat * processor_distances)

    return int(hop_bytes)


