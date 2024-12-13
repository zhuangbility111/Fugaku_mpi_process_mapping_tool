import numpy as np
from tqdm import tqdm
from config import *
import pickle
import os

def preprocess_coords(coords_3D):
    """
    Preprocess 3D coordinates to remove redundancy.
    
    :param coords_3D: List of 3D coordinates.
    :return: Preprocessed list of unique sorted 3D coordinates.
    """
    # Use a set to ensure uniqueness, then sort for consistent order
    unique_coords = sorted(set(coords_3D))
    return unique_coords

def build_path_lookup_table(coords_3D, shape_3D):
    """
    Build a lookup table for paths between all pairs of 3D coordinates.

    :param coords_3D: List of 3D coordinates.
    :param shape_3D: Shape of the 3D torus (dim_x, dim_y, dim_z).
    :return: Lookup table as a dictionary.
    """
    cache = {}
    lookup_table = {}
    
    # Iterate over all coordinate pairs
    for coord_i in tqdm(coords_3D, desc="Building Lookup Table", ncols= 150):
        for coord_j in coords_3D:
            # Compute the path using caching
            path = torus_dor_path_with_cache(coord_i, coord_j, shape_3D, cache)
            lookup_table[(coord_i, coord_j)] = path
    
    return lookup_table

def torus_dor_path_with_cache(start, end, dimensions, cache):
    """
    Compute the path using cached subpaths for efficiency.

    :param start: Tuple (x1, y1, z1) representing the starting coordinate.
    :param end: Tuple (x2, y2, z2) representing the destination coordinate.
    :param dimensions: Tuple (dim_x, dim_y, dim_z) representing the torus dimensions.
    :param cache: Dictionary to store precomputed subpaths.
    :return: List of links taken as tuples [(current_node, next_node), ...].
    """
    key = (start, end)
    if key in cache:
        return cache[key]

    
    current = list(start)
    path = []
    
    for dim, size in enumerate(dimensions):
        target = end[dim]

        # Check for overlapping subpath
        sub_start = tuple(current)
        sub_key = (sub_start, tuple([target if d == dim else current[d] for d in range(3)]))
        if sub_key in cache:
            sub_path = cache[sub_key]
        else:
            # Calculate subpath in the current dimension
            sub_path = []
            forward_dist = (target - current[dim]) % size
            backward_dist = (current[dim] - target) % size
            step = 1 if forward_dist <= backward_dist else -1
            while current[dim] != target:
                next_node = current.copy()
                next_node[dim] = (current[dim] + step) % size
                sub_path.append((tuple(current), tuple(next_node)))
                current[dim] = next_node[dim]
            cache[sub_key] = sub_path  # Cache the subpath

        path.extend(sub_path)  # Append the subpath to the total path

    cache[key] = path
    return path


def save_links(sorted_links, save_filename):
    """
    Save the sorted links to a file using pickle.

    :param sorted_links: The sorted links to save.
    :param save_filename: Filepath where the links should be saved.
    """
    with open(save_filename, 'wb') as file:
        pickle.dump(sorted_links, file)
    # print(f"Sorted links saved to {save_filename}")


def load_links(save_filename):
    """
    Load the sorted links from a file if it exists.

    :param save_filename: Filepath where the sorted links is saved.
    :return: The sorted links if the file exists, otherwise None.
    """
    if os.path.exists(save_filename):
        with open(save_filename, 'rb') as file:
            sorted_links = pickle.load(file)
        # print(f"Loaded sorted links from {save_filename}")
        return sorted_links
    else:
        # print(f"File {save_filename} does not exist.")
        return None
    
def compute_link_usage_with_lookup(shape_3D, coords_3D, com_mat, save_file):
    """
    Compute link usage using a precomputed lookup table.

    :param shape_3D: Shape of the 3D torus (dim_x, dim_y, dim_z).
    :param coords_3D: List of 3D coordinates.
    :param com_mat: Communication matrix (N x N).
    :param lookup_table: Precomputed lookup table for paths.
    :return: Sorted list of links with their usage values.
    """

    save_filename = tmp_dir+"/"+save_file+".links"

    sorted_links = load_links(save_filename)

    if sorted_links != None:
        return sorted_links
    
    # tupled coordinates
    coords_3D = [tuple(c) for c in coords_3D]
    
    # Step 1: Preprocess coordinates and build the lookup table
    # the lookup table will contains all the path between any pair of nodes in coord_3D using the DIR X->Y->Z rounting algorithm 
    lookup_table = build_path_lookup_table(preprocess_coords(coords_3D), shape_3D)
          
    link_dict = {}

    # Use the lookup table to calculate link usage
    for i, coord_i in tqdm(enumerate(coords_3D), total=len(coords_3D), desc="Computing Link Usage", ncols=150):
        for j, coord_j in enumerate(coords_3D):
            path = lookup_table[(coord_i, coord_j)]
            for link in path:
                key = link
                if key not in link_dict:
                    link_dict[key] = 0
                link_dict[key] += com_mat[i, j]

    # Sort the links by decreasing values
    sorted_links = sorted(link_dict.items(), key=lambda item: item[1], reverse=True)

    # Print sorted links
    # print("Links sorted by decreasing values:")
    # for link, value in sorted_links:
    #     print(f"{link}:\t{value}")

    save_links(sorted_links, save_filename)
    
    return sorted_links
        

def anaylse_link_usage(link_usage, name):
    """
    Print statistics on link usage
    """
    values = np.array(sorted([v for _, v in link_usage]))
    print(f"==== link usage stats for {name}")
    print(f"\tmax:\t{values[-1]:.2e}")
    m_index = int(0.5 * len(link_usage))
    print(f"\tmedian:\t{values[m_index]:.2e}")
    print(f"\tmean:\t{np.mean(values):.2e}")
    print(f"\tmin:\t{values[0]:.2e}")
    sum_values = np.sum(values)
    print(f"\tsum:\t{sum_values:.2e}")

    sum_by_dim = [0,0,0]
    links_by_dim = [0,0,0]
    nb_links = 0
    for link, value in link_usage:
        differing_indices = [i for i, (a, b) in enumerate(zip(link[0], link[1])) if a != b]
        if len(differing_indices) != 1:
            print(f"Error link \"{link}\" is using more than one dimentsion at a time...")
            quit()
        sum_by_dim[differing_indices[0]] += value
        links_by_dim[differing_indices[0]] += 1
        nb_links += 1


    print("\tSum by dimension:")
    print(f"\t\tX: {sum_by_dim[0]:.2e} ({100*sum_by_dim[0]/sum_values:2.2f}%)\t{links_by_dim[0]} ({100*links_by_dim[0]/nb_links:2.2f}%)")
    print(f"\t\tY: {sum_by_dim[1]:.2e} ({100*sum_by_dim[1]/sum_values:2.2f}%)\t{links_by_dim[1]} ({100*links_by_dim[1]/nb_links:2.2f}%)")
    print(f"\t\tZ: {sum_by_dim[2]:.2e} ({100*sum_by_dim[2]/sum_values:2.2f}%)\t{links_by_dim[2]} ({100*links_by_dim[2]/nb_links:2.2f}%)")
