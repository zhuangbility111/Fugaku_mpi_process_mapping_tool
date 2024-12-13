import numpy as np

def load_comm_mat(filename):
    """
    Load a communication matrix from a file into a NumPy array.
    
    Args:
        filename (str): Path to the file containing the matrix.

    Returns:
        np.array: The loaded matrix as a NumPy array of integers.
    """
    return np.loadtxt(filename, dtype=int)


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

def parse_coord_file(coord_filename):
    phys_coords = {};
    logical_coords = {};
    with open(coord_filename, 'r') as file:  
        for line in file.readlines():
            if "(X,Y,Z)" in line:
                var = extract_variables(line)
                # print(line ,":", var, "-", var[3:6])
                phys_coords[var[0]] = var[6:12] 
                logical_coords[var[0]] = var[3:6] 


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
