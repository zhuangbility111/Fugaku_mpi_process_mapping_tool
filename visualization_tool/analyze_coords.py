#!/usr/bin/python3
import sys
import os
import math
import re
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import numpy as np
import argparse
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './subroutines')))

from subroutines.loader import parse_coord_file, load_comm_mat

# configure this one
from config import * 

# my subroutines
from topology_draw import *
from mapping import *
from routing import *
from my_utils import *
from hop_bytes import *
from analyze_nodes import analyze_3D_nodes, remove_duplicates_preserve_order
import T6Dcoord2ID

grid_shape = (24, 23, 24, 2, 3, 2)  # Fugaku Grid size

def parse_arguments():
    """
    Parse command-line arguments for visualization and input file.
    - `-v`: Flag to enable visualization (default: off).
    - `-f <filename>`: Mandatory parameter for the input file.
    """
    parser = argparse.ArgumentParser(description="Analayze coordinate file for physical and logical topology")
    
    # Optional flag for visualization
    parser.add_argument(
        "-v", 
        action="store_true", 
        help="Enable visualization"
    )
    
    # Mandatory argument for input file
    parser.add_argument(
        "-f", 
        required=True, 
        metavar="<coord filename>", 
        help="Path to the coordinate input file"
    )

    parser.add_argument(
        "-m", 
        required=False, 
        metavar="<comm filename>", 
        help="Path to the the communication matrix filename"
    )
    
    # Parse the arguments
    args = parser.parse_args()
    return args



args = parse_arguments()            

coord_filename = args.f

# compute a unique ID for caching data
md5 = compute_md5(coord_filename)
out_filename = os.path.basename(coord_filename)+"_"+md5

# create tmp_dir (where data are cached) if necessary
os.makedirs(tmp_dir, exist_ok=True)

# these lists of coordinnates are sorted by ranks
phys_nodes_coord_3D, phys_nodes_coord_6D, logical_nodes_coord_3D, logical_nodes_shape, physical_node_shape, logical_coord_dim, physical_coord_dim = parse_coord_file(coord_filename) 

analyze_3D_nodes(phys_nodes_3D, "physical")
analyze_3D_nodes(log_nodes_3D, "logical")

# extact unique coordinates: makes rendering faster
phys_unique_3D = remove_duplicates_preserve_order(phys_nodes_3D)
if args.v : 
    # Call the function with the grid shape and selected nodes
    display_3d_torus_with_selected_nodes(grid_shape[0:3], phys_unique_3D)
    
if args.m:
    print("\n####### Computing Hop-Bytes in the 6D Torus")
    mat = load_comm_mat(args.m)
    hop_matrix = hop_matrix_from_coords(grid_shape, phys_nodes_coord_6D) 
    hop_bytes = mat * hop_matrix
    print(f"No Mappping Hop-Bytes = {hop_bytes.sum()}")

    # Convert to NumPy array
    phys_nodes_coord_6D = np.array(phys_nodes_coord_6D)

    # Get unique rows without sorting
    # Extract 6D coordinates that are unique but keep the order of the original coordinates (s they are sorted by ranks
    _, indices = np.unique(phys_nodes_coord_6D, axis=0, return_index=True)
    phys_unique_6D = phys_nodes_coord_6D[np.sort(indices)]

    hop_matrix_unique = hop_matrix_from_coords(grid_shape, phys_unique_6D)
    sigma = np.repeat(np.arange(len(phys_unique_6D)), len(mat)//len(phys_unique_6D) )
    if compute_hop_bytes(mat, sigma, hop_matrix_unique, hop_matrix) != hop_bytes.sum():
        print("compute_hop_bytes() failed to find the correct value.\n")
        quit()

    phys_ids=[];
    # convert 6D coordinate into node id
    for node in phys_nodes_coord_6D:
        phys_ids.append(str(T6Dcoord2ID.coord_2_id(node[0], node[1], node[2], node[3] , node[4], node[5])))
    sigma_tm_sub_fugaku = topomatch_sub_fugaku(out_filename, args.m, mat, phys_ids, len(mat), hop_matrix_unique)
    sigma_tm_phys, sigma_scotch = topomatch_phys_grid(out_filename, args.m, mat, phys_ids, hop_matrix, len(mat), hop_matrix_unique)

    mat_tm_sub_fugaku = permute_mat(mat, sigma_tm_sub_fugaku)
    mat_tm_phys = permute_mat(mat, sigma_tm_phys)
    mat_scotch = permute_mat(mat, sigma_scotch)
    
    link_usage = compute_link_usage_with_lookup(grid_shape[0:3], phys_nodes_3D, mat, out_filename+"_RR")
    link_usage_tm_sub_fugaku = compute_link_usage_with_lookup(grid_shape[0:3], phys_nodes_3D, mat_tm_sub_fugaku, out_filename+"_tm_sub")
    link_usage_tm_phys = compute_link_usage_with_lookup(grid_shape[0:3], phys_nodes_3D, mat_tm_phys, out_filename+"_tm_phys")
    link_usage_scotch = compute_link_usage_with_lookup(grid_shape[0:3], phys_nodes_3D, mat_tm_sub_fugaku, out_filename+"_scotch")
    anaylse_link_usage(link_usage, coord_filename + " RR")
    anaylse_link_usage(link_usage_tm_sub_fugaku, coord_filename + " TM Sub Fugaku")
    anaylse_link_usage(link_usage_tm_phys, coord_filename + " TM Phys")
    anaylse_link_usage(link_usage_scotch, coord_filename + " Scotch")
    if args.v:
        display_3d_torus_with_selected_nodes(grid_shape[0:3], phys_unique_3D, link_usage, name= coord_filename + " RR", quantile = 0)
        display_3d_torus_with_selected_nodes(grid_shape[0:3], phys_unique_3D, link_usage_tm_sub_fugaku, name=coord_filename + " TM Sub Fugaku",quantile = 0.33)
        display_3d_torus_with_selected_nodes(grid_shape[0:3], phys_unique_3D, link_usage_tm_phys, name= coord_filename + " TM Phys",quantile = 0.33)
        display_3d_torus_with_selected_nodes(grid_shape[0:3], phys_unique_3D, link_usage_scotch, name= coord_filename + " Scotch",quantile = 0.33)
        
