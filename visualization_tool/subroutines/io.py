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

def load_coord_file(coord_filename):
    phys_coords = {}
    logical_coords = {}

    logical_node_shape = None
    physical_node_shape = None
    with open(coord_filename, 'r') as file:  
        for line in file.readlines():
            if "(X,Y,Z)" in line:
                var = extract_variables(line)
                # print(line ,":", var, "-", var[3:6])
                phys_coords[var[0]] = var[6:12] 
                logical_coords[var[0]] = var[3:6] 
            
            if line.startswith("My Dimen"):
                print("reading head of file...")
                # read first line to get the dimension of logical coordinate, and the shape of logical coordinate
                # and set the dimension of physical coordinate, and the shape of physical coordinate
                # use re to get the number of dimension
                dimension_pattern = r'Dimension\s*=\s*(\d+)' # pattern to match the dimension
                dimension_match = re.search(dimension_pattern, line)
                logical_coord_dim = int(dimension_match.group(1)) if dimension_match else 3
                physical_coord_dim = 6 # always 6

                logical_node_shape = np.zeros(logical_coord_dim, dtype=np.int32) # shape of logical coordinate
                physical_node_shape = np.zeros(physical_coord_dim, dtype=np.int32) # shape of physical coordinate                

    ids = sorted(phys_coords.keys()) #keys are the ranks (var[0]). So we sort by rank in case this in not how the file is arranged

    if not np.all(np.arange(len(ids)) == ids):
        print(f"Error: not all the ranks from 0 to {len(ids)-1} are present in {coord_filename}.")
        quit()
    
    phys_nodes_3D = []
    phys_nodes_6D = []
    for id in ids :
        # print(id, ":", phys_coords[id], "-", logical_coords[id])
        phys = phys_coords[id]
        phys_nodes_3D.append(phys[0:3]) # only the X_Y_Z
        phys_nodes_6D.append(phys)

    log_nodes_3D = []
    for id in ids :
        # print(id, ":", phys_coords[id], "-", logical_coords[id])
        logical = logical_coords[id]
        log_nodes_3D.append(logical[0:3]) # only the X_Y_Z

    return phys_nodes_3D, phys_nodes_6D, log_nodes_3D
