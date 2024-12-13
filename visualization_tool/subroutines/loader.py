import numpy as np
import re

'''
Load (physical and logical) coordinates from a file into a NumPy array.
Parse the file line by line and extract the variables from each line.
'''

def load_comm_mat(filename):
    """
    Load a communication matrix from a file into a NumPy array.
    
    Args:
        filename (str): Path to the file containing the matrix.

    Returns:
        np.array: The loaded matrix as a NumPy array of integers.
    """
    if filename.endswith(".npy"):
        return np.load(filename).astype(np.int64)
    elif filename.endswith(".txt"):
        return np.loadtxt(filename, dtype=np.int64)
    else:
        print(f"Error: file {filename} is not a .npy file or a text file.")
        quit()


def remove_abc_coords(physical_node_coords, mode="group", physical_node_shape=None):
    """
    Remove the ABC coordinates from the physical node coordinates. 
    For the group mode, the ABC coordinates are removed from the physical group coordinates directly.
    For the node mode, the ABC coordinates are combined with XYZ coordinates to form a 3D array. X = X * len(B_axis) + B, Y = Y * len(A_axis) + A, Z = Z * len(C_axis) + C.
    Combination method: https://www.fugaku.r-ccs.riken.jp/doc_root/en/manuals/tcsds-1.2.40/lang/MPI/j2ul-2567-01enz0.pdf#page=9.09

    :param physical_node_coords: An array where each row is a point (x, y, z, a, b, c).
    :param mode: The mode of the analysis (group or node).
    :return: An array where each row is a point (x, y, z) without the ABC coordinates.
    """
    if mode == "group":
        return physical_node_coords[:, 0:3]
    elif mode == "node":
        if physical_node_shape is None:
            print("Error: The physical node shape is required for the node mode.")
            exit(1)

        # Extract the ABC coordinates
        a_coords = physical_node_coords[:, 3]
        b_coords = physical_node_coords[:, 4]
        c_coords = physical_node_coords[:, 5]

        # Extract the XYZ coordinates
        x_coords = physical_node_coords[:, 0]
        y_coords = physical_node_coords[:, 1]
        z_coords = physical_node_coords[:, 2]

        # Extract the length of the ABC axes
        len_a_axis = physical_node_shape[3]
        len_b_axis = physical_node_shape[4]
        len_c_axis = physical_node_shape[5]

        # Combine the ABC and XYZ coordinates
        x_coords = x_coords * len_b_axis + b_coords
        y_coords = y_coords * len_a_axis + a_coords
        z_coords = z_coords * len_c_axis + c_coords

        return np.column_stack((x_coords, y_coords, z_coords))


def extract_variables(input_string):
    """
    Extracts variables from the input string by splitting on ',' and parsing each part.

    :param input_string: String containing the data to parse.
    :return: A list with extracted values.
    """
    # Initialize results
    rank, hostname, cpu_id = None, None, None
    X, Y, Z = None, None, None
    x, y, z, a, b, c = None, None, None, None, None, None

    # Split the string by ',' and process each part
    parts = input_string.split(', ')
    for part in parts:
        part = part.strip()  # Remove extra spaces
        if "rank=" in part:
            rank = int(part.split('=')[1])
        elif "hostname=" in part:
            hostname = part.split('=')[1]
        elif "cpu_id=" in part:
            cpu_id = int(part.split('=')[1])
        elif "(X,Y,Z)=" in part:
            X, Y, Z = map(int, part.split('=')[1].strip('()').split(','))
        elif "(x,y,z,a,b,c)=" in part:
            x, y, z, a, b, c = map(int, part.split('=')[1].strip('()').split(','))

    # Return the values in the desired order
    return [rank, hostname, cpu_id, X, Y, Z, x, y, z, a, b, c]


def parse_coord_file(coord_filename, mode="group") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    :param coord_filename: The name of the file containing the coordinates.
    :param mode: The mode of the analysis (group or node). group: remove ABC coordinates from the physical group coordinates directly. node: combine ABC and XYZ coordinates to form a 3D array.
    :return: The physical and logical coordinates, the shape of the logical coordinates, the shape of the physical coordinates, the dimension of the logical coordinates, and the dimension of the physical coordinates.
    """
    phys_coords = {}
    logical_coords = {}

    with open(coord_filename, 'r') as file:  
        for line in file.readlines():
            if "(X,Y,Z)" in line:
                var = extract_variables(line)
                # print(line ,":", var, "-", var[3:6])
                phys_coords[var[0]] = var[6:12]
                logical_coords[var[0]] = var[3:6]
            
            if "My Dimen" in line:
                print("reading head of file...")
                # read first line to get the dimension of logical coordinate, and the shape of logical coordinate
                # and set the dimension of physical coordinate, and the shape of physical coordinate
                # use re to get the number of dimension
                dimension_pattern = r'Dimension\s*=\s*(\d+)' # pattern to match the dimension
                dimension_match = re.search(dimension_pattern, line)
                logical_coord_dim = int(dimension_match.group(1)) if dimension_match else 3
                physical_coord_dim = 6 # always 6

    ids = sorted(phys_coords.keys()) #keys are the ranks (var[0]). So we sort by rank in case this in not how the file is arranged

    if not np.all(np.arange(len(ids)) == ids):
        print(f"Error: not all the ranks from 0 to {len(ids)-1} are present in {coord_filename}.")
        quit()
    
    phys_nodes_coord_6D = []
    logical_nodes_coord_3D = []
    for rank in ids :
        # print(id, ":", phys_coords[id], "-", logical_coords[id])
        phys = phys_coords[rank]
        phys_nodes_coord_6D.append(phys)

        logical = logical_coords[rank]
        logical_nodes_coord_3D.append(logical) # only the X_Y_Z

    phys_nodes_coord_6D = np.array(phys_nodes_coord_6D, dtype=np.int32)
    logical_nodes_coord_3D = np.array(logical_nodes_coord_3D, dtype=np.int32)

    phys_nodes_coord_3D = remove_abc_coords(phys_nodes_coord_6D, mode=mode)

    physical_node_shape = np.array([24, 23, 24, 2, 3, 2], dtype=np.int32)
    logical_nodes_shape = np.array([logical_nodes_coord_3D[:, 0].max() + 1, logical_nodes_coord_3D[:, 1].max() + 1, logical_nodes_coord_3D[:, 2].max() + 1], dtype=np.int32)

    return phys_nodes_coord_3D, phys_nodes_coord_6D, logical_nodes_coord_3D, logical_nodes_shape, physical_node_shape, logical_coord_dim, physical_coord_dim
