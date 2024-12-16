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

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './subroutines')))

from subroutines.loader import parse_coord_file, load_comm_mat
from subroutines.utils import compute_md5
from subroutines.config_template import *
from subroutines.analyze_nodes import analyze_3D_nodes, remove_duplicates_preserve_order
from subroutines.hop_bytes import get_hop_matrix_from_coords_numpy
from subroutines.mapping import do_process_mapping_for_diff_mapping
from subroutines.routing import compute_link_usage_for_diff_mapping
from subroutines.process_mapping_enum import ProcessMappingMethod

from subroutines.config_template import *

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

    parser.add_argument(
        "--process_mapping_method",
        "-pmm",
        nargs="*", # 0 or more arguments
        default=None,
        help="Process mapping method. Set multiple values separated by spaces or use 'all'. Default is None. Options: 'all', 'sub_fugaku', 'tm_phys', 'scotch'"
    )

    # Parse the arguments
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()            

    coord_filename = args.f
    comm_filename = args.m
    enable_visualization = args.v

    # Process mapping method
    process_mapping_methods = [ProcessMappingMethod.RR]
    if args.pmm is not None:
        if "all" in args.pmm:
            process_mapping_methods.append([ProcessMappingMethod.SUB_FUGAKU, ProcessMappingMethod.TM_PHYS, ProcessMappingMethod.SCOTCH])
        elif "sub_fugaku" in args.pmm:
            process_mapping_methods.append(ProcessMappingMethod.SUB_FUGAKU)
        elif "tm_phys" in args.pmm:
            process_mapping_methods.append(ProcessMappingMethod.TM_PHYS)
        elif "scotch" in args.pmm:
            process_mapping_methods.append(ProcessMappingMethod.SCOTCH)


    # compute a unique ID for caching data
    md5 = compute_md5(coord_filename)
    out_filename = os.path.basename(coord_filename)+"_"+md5

    # create tmp_dir (where data are cached) if necessary
    os.makedirs(tmp_dir, exist_ok=True)

    # these lists of coordinnates are sorted by ranks
    phys_nodes_coord_3D, phys_nodes_coord_6D, logical_nodes_coord_3D, logical_nodes_shape, physical_node_shape, logical_coord_dim, physical_coord_dim = parse_coord_file(coord_filename)

    analyze_3D_nodes(phys_nodes_coord_3D, "physical")
    analyze_3D_nodes(logical_nodes_coord_3D, "logical")

    # extact unique coordinates: makes rendering faster
    unique_phys_nodes_coord_3D = remove_duplicates_preserve_order(phys_nodes_coord_3D)
    if enable_visualization: 
        # TODO - need to refactor the following code
        # Call the function with the grid shape and selected nodes
        display_3d_torus_with_selected_nodes(physical_node_shape[:3], unique_phys_nodes_coord_3D)
        
    if comm_filename:
        print("\n####### Computing Hop-Bytes in the 6D Torus")
        comm_mat = load_comm_mat(comm_filename)
        hop_matrix = get_hop_matrix_from_coords_numpy(physical_node_shape, phys_nodes_coord_6D) 
        hop_bytes = comm_mat * hop_matrix
        print(f"No Mappping Hop-Bytes = {hop_bytes.sum()}")

        # Convert to NumPy array
        phys_nodes_coord_6D = np.array(phys_nodes_coord_6D)

        # Do process mapping according to the selected methods
        comm_mat_for_diff_mapping = do_process_mapping_for_diff_mapping(comm_mat, physical_node_shape, phys_nodes_coord_6D, hop_matrix, process_mapping_methods, out_filename, comm_filename)

        # Compute link usage
        link_usage_for_diff_mapping = compute_link_usage_for_diff_mapping(physical_node_shape[0:3], phys_nodes_coord_3D, comm_mat_for_diff_mapping, process_mapping_methods, out_filename, comm_filename)

        if enable_visualization:
            display_3d_torus_with_selected_nodes(physical_node_shape[0:3], unique_phys_nodes_coord_3D, link_usage, name= coord_filename + " RR", quantile = 0)
            display_3d_torus_with_selected_nodes(physical_node_shape[0:3], unique_phys_nodes_coord_3D, link_usage_tm_sub_fugaku, name=coord_filename + " TM Sub Fugaku",quantile = 0.33)
            display_3d_torus_with_selected_nodes(physical_node_shape[0:3], unique_phys_nodes_coord_3D, link_usage_tm_phys, name= coord_filename + " TM Phys",quantile = 0.33)
            display_3d_torus_with_selected_nodes(physical_node_shape[0:3], unique_phys_nodes_coord_3D, link_usage_scotch, name= coord_filename + " Scotch",quantile = 0.33)
            