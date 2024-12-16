import numpy as np

def get_hop_matrix_from_coords_numpy(torus_shape, phys_nodes):
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


def get_hop_matrix_from_coords_naive(torus_shape, node_coords):
    """
    Compute a matrix of hop counts between nodes in a sub-torus using a naive approach.

    Args:
        torus_shape (tuple): Dimensions of the torus (e.g., (24, 23, 24)).
        node_coords (list of tuples): List of node coordinates in the torus, sorted by IDs.

    Returns:
        np.array: A matrix where element (i, j) is the hop count between nodes i and j.    
    """

    size = len(node_coords)
    hop_matrix = np.zeros((size, size), dtype=np.int32) # distance matrix
    # calculate the distance between each pair of logical node
    for i in range(size):
        for j in range(i+1, size):
            distance = 0
            # get manhattan distance between each pair of logical node
            for k in range(len(torus_shape)):
                # each dimension is a ring
                # so the distance is the minimum distance between left-to-right and right-to-left
                # Note: even in physical coordinate, the a and c axis are not rings, but the length is 2, so we can treat them as rings
                left_to_right = abs(node_coords[i][k] - node_coords[j][k])
                right_to_left = torus_shape[k] - left_to_right
                distance += min(left_to_right, right_to_left)
                hop_matrix[i, j] = distance
                hop_matrix[j, i] = distance

    return hop_matrix


def compute_hop_bytes(mat, sigma, h, h_full):
    # Map processes to processor distances using sigma
    processor_distances = h[sigma[:, None], sigma[None, :]]  # Map 2048x2048 to processor distances

    
    if h_full is not None and  np.array_equal(processor_distances, h_full) == False:
        print("======================> Error arrays not equal!")
    # Element-wise multiply the communication matrix with the processor distances
    hop_bytes = np.sum(mat * processor_distances)

    return int(hop_bytes)


