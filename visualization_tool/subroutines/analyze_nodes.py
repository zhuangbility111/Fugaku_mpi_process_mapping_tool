from itertools import permutations
import numpy as np
from collections import Counter


def group_matrix(mat, groups):
    """
    Create a new matrix by grouping the original matrix indices and summing values outside the group.

    Args:
        mat (list of list or np.array): Original matrix (2D array).
        groups (list of list): List of groups, where each group is a list of indices.

    Returns:
        np.array: A new matrix where each group is summed outside its group.
    """
    mat = np.array(mat)  # Ensure `mat` is a NumPy array for efficient slicing
    n = len(groups)
    new_mat = np.zeros((n, n))  # Initialize the resulting grouped matrix
    
    for i, group_i in enumerate(groups):
        for j, group_j in enumerate(groups):
            if(i==j):
                mat[i][j] = 0
            else:
                # Sum over the intesrection of both groups
                new_mat[i][j] = mat[np.ix_(group_i, group_j)].sum()
    
    return new_mat


# Check if rows within each group are identical
def check_groups_identical(array, group_size):
    for i in range(0, len(array), group_size):
        group = array[i:i + group_size]
        if not np.all(np.equal(group, group[0])):  # Compare each row with the first row of the group
            return False
    return True


def count_oversubscribing_X_Y_Z(nodes):
    """
    Checks the minimum and maximum number of times any unique coordinate appears in a set of 3D nodes.

    This function analyzes the given list of 3D coordinates and determines:
    1. The maximum number of times a unique coordinate appears in the set.
    2. The minimum number of times a unique coordinate appears in the set.

    Args:
        nodes (list of list): A list of 3D coordinates, where each coordinate is a list [x, y, z].

    Returns:
        tuple:
            - int: The maximum number of times any coordinate appears.
            - int: The minimum number of times any coordinate appears.

    Example:
        nodes = [
            [0, 0, 0], [0, 0, 0], [0, 0, 1],
            [0, 1, 0], [0, 1, 0], [0, 1, 0]
        ]
        result = check_oversubscribing_X_Y_Z(nodes)
        # Output: (1, 3)
    """

    # Count occurrences of each unique coordinate
    coordinate_counts = Counter(tuple(node) for node in nodes)

    # Find the maximum and minimum counts
    max_count = max(coordinate_counts.values())
    min_count = min(coordinate_counts.values())

    return min_count, max_count

def is_compact_X_Y_Z(nodes):
    """
    Checks whether a set of 3D coordinates (nodes) forms a compact cuboid in the X, Y, and Z dimensions using NumPy.

    Args:
        nodes (list of list): A list of 3D coordinates, where each coordinate is a list [x, y, z].

    Returns:
        tuple:
            - bool: True if the nodes form a compact cuboid, False otherwise.
            - int or None: The dimension along X if compact, None otherwise.
            - int or None: The dimension along Y if compact, None otherwise.
            - int or None: The dimension along Z if compact, None otherwise.
    """
    # Convert nodes to a NumPy array and remove duplicates
    nodes = np.array(remove_duplicates_preserve_order(nodes))

    # Compute min and max for each dimension
    min_coords = nodes.min(axis=0)
    max_coords = nodes.max(axis=0)

    # Calculate dimensions
    dim_x, dim_y, dim_z = max_coords - min_coords + 1

    # Expected number of coordinates for a compact cuboid
    expected_count = dim_x * dim_y * dim_z

    # Check if the actual number of unique coordinates matches the expected count
    if len(nodes) == expected_count:
        return True, dim_x, dim_y, dim_z
    else:
        return False, None, None, None


def check_coordinate_order(coordinates):
    """
    Checks if the given list of coordinates follows one of the six possible orders.

    :param coordinates: List of coordinates, e.g., [[x1, y1, z1], [x2, y2, z2], ...]
    :return: The matching order if valid, or None if no valid order exists.
    """
    # Generate all possible orders
    possible_orders = list(permutations(range(3)))  # (0, 1, 2), (0, 2, 1), etc.
    labels = ['X', 'Y', 'Z']
    
    for order in possible_orders:
        if is_ordered(coordinates, order):
            return [labels[i] for i in order]  # Return the matching order

    return None  # No valid order found


def is_ordered(coordinates, order):
    """
    Checks if the coordinates follow the specified order.

    :param coordinates: List of coordinates.
    :param order: Tuple specifying the order (e.g., (0, 1, 2) for X → Y → Z).
    :return: True if the coordinates follow the order, False otherwise.
    """
    # Extract the sorted order of coordinates based on the given dimension order
    sorted_coords = sorted(coordinates, key=lambda coord: tuple(coord[i] for i in order))
    # print(order)
    # print(sorted_coords)
    return sorted_coords == coordinates

def remove_duplicates_preserve_order(nodes):
    """
    Remove duplicates from a list while preserving the original order.

    :param nodes: List of tuples representing nodes.
    :return: List of unique nodes in original order.
    """
    seen = set()
    unique_nodes = []
    # Convert array to tuple for hashability
    for node in nodes:
        node_tuple = tuple(node)
        if node_tuple not in seen:
            unique_nodes.append(node)
            seen.add(node_tuple)

    # print(unique_nodes)
    return unique_nodes


def remove_consecutive_duplicates(nodes):
    """
    Remove consecutive duplicates from a list 

    :param nodes: List of tuples representing nodes.
    :return: List of unique nodes in original order.
    """
    res_nodes = []
    cur_node = None
    # Convert array to tuple for hashability
    for node in nodes:
        if cur_node == None:
            cur_node= node
            res_nodes.append(node)
        elif cur_node != node: 
            cur_node= node
            res_nodes.append(node)

    # print(res_nodes)
    return res_nodes


def analyze_3D_nodes(nodes_coords, type: str):

    print(f"\n####### Analyzing {type} topology")
    cons_nodes = remove_consecutive_duplicates(nodes_coords)
    no_dup_nodes = remove_duplicates_preserve_order(nodes_coords)

    result = check_coordinate_order(cons_nodes)

    if result:
        print(f"The {type} X, Y, Z oordinates follow the order: {' → '.join(result)}")
    else:
        print("The coordinates do not follow any valid order.")
        result = check_coordinate_order(no_dup_nodes)
        if result:
            print(f"However, the non-duplicated {type} X, Y, Z coordinates follow the order: {' → '.join(result)}")

        else:
            print(f"Even the non-duplicated {type} X, Y, Z coordinates do not follow any valid order.")

    

    is_compact, dim_x, dim_y, dim_z = is_compact_X_Y_Z(no_dup_nodes)
    if is_compact : 
        print(f"Embedded in a compact {type} 3D grid of dimension: {dim_x}x{dim_y}x{dim_z} of size {dim_x * dim_y * dim_z}")

    else:
        print("Not embedded in a compact 3D grid")


    min_oversub, max_oversub = count_oversubscribing_X_Y_Z(nodes_coords);
    if min_oversub == max_oversub :
        print(f"Constant oversubscribing factor in X, Y, Z of {max_oversub}")
    else:
        print(f"Non-constant oversubscribing factor: factor in [{min_oversub}, {max_oversub}]")


