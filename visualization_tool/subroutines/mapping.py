import os
import numpy as np
import math 
from config_template import *
from hop_bytes import *
from collections import defaultdict
from T6Dcoord2ID import coord_2_id



def permute_comm_mat(mat, sigma):
    """
    Apply the sigma mepping to the matrix
    with 
    mat = [[   0    1  100    1],
    [   1    0    1  100],
    [ 100    1    0    1],
    [   1 1000    0    1]]
    sigma = [0, 3, 1, 2]
    we opbtain:

 permuted_mat = [[   0  100    1    1],
    [ 100    0    1    1],
    [   1    0    1 1000],
    [   1    1  100    0]]
    """
    # Compute the inverse of sigma
    inverse_sigma = np.argsort(sigma)

    # Permute rows and columns based on the inverse permutation
    permuted_mat = mat[np.ix_(inverse_sigma, inverse_sigma)]

    return permuted_mat

def topomatch_phys_grid(out_filename, file_mat, mat, ids, hop_matrix, nb_procs, hop_matrix_unique):
    print("\t==== Topomatch on allocated Fugaku nodes")

    topology_file =  tmp_dir+"/"+out_filename+"_topology.grf"
    if not os.path.exists(topology_file):
        print("\tCreating grf topology file...")
        str_res = ""
        prev_id = -1
        nb_vertices = 0
        nb_edges = 0
        for i, line in enumerate(hop_matrix):
            if ids[i] != prev_id:
                nb_vertices += 1
                neighbours = [str(ids[j]) for j in range(len(line)) if line[j] == 1]  # Find indices and convert to strings
                neighbours = sorted(set(neighbours))
                nb_edges += len(neighbours)
                str_res += f"{ids[i]} 1 {len(neighbours)} {' '.join(neighbours)}\n"
                prev_id = ids[i]

        with open(topology_file,"w") as file:
            # File format
            file.write("0\n")
            # Number_of_verticies number_of_"edges"
            file.write(f"{nb_vertices} {nb_edges}\n")
            # Chaco format
            file.write("0 101\n") # 0 : base value (as in C) ; 1 : vertex weights ;  0 : no edge weights ; 1 : vertex label provided 
            file.write(str_res)
    else:
        nb_vertices = 0
        prev_id = -1
        for i, line in enumerate(hop_matrix):
            if ids[i] != prev_id:
                nb_vertices += 1
                prev_id = ids[i]
            
    phys_tgt = tmp_dir+"/"+out_filename+"_phys.tgt"
    if not os.path.exists(phys_tgt):
        print("\tCreating tgt topology file from grf file...") 
        cmd= f"{amk_grf} {topology_file} > {phys_tgt}"
        os.system(cmd)

    # Now run topomatch on this topology
    cmd = f"{mapping} -t {phys_tgt} -c {file_mat} -a 0"

    target_oversub = math.ceil(nb_procs/nb_vertices)
    if target_oversub > 1:
        cmd += f" -o {target_oversub}"

    res_file = tmp_dir+"/"+out_filename+"_phys_mapping.map"
    cmd += f" > {res_file}"
    
    if not os.path.exists(res_file):
        print(f"\tRunning TopoMatch to {res_file}...")
        os.system(cmd)

    # Output TopoMatch result
    # cmd = f"grep TopoMatch {res_file}"
    # os.system(cmd)
        
    sigma_tm = parse_topomach_mapping(res_file, ids, logical_numbering = True)
    print(f"\tTM-2 Hop-Bytes = {compute_hop_bytes(mat, sigma_tm, hop_matrix_unique, None)}")

    
    print("\t==== Mapping directely with Scotch...")
    
    com_mat_grf = tmp_dir+"/"+out_filename+"_com_mat.grf"
    if not os.path.exists(com_mat_grf):
        print("\tCreating grf comunication matrix...")
        com_mat_to_grf(mat, com_mat_grf)

    res_file = tmp_dir+"/"+out_filename+"_scotch.res"
    # start = "\"m{asc=b{width=3,bnd=d{pass=40,dif=1,rem=0}f{move=80,pass=-1,bal=0},org=f{move=80,pass=-1,bal=0}},low=r{job=t,bal=0,map=t,poli=S,sep=(m{asc=b{bnd=f{move=120,pass=-1,bal=0,type=b},org=f{move=120,pass=-1,bal=0,type=b},width=3},low=h{pass=10}f{move=120,pass=-1,bal=0,type=b},vert=120,rat=0.8}|m{asc=b{bnd=f{move=120,pass=-1,bal=0,type=b},org=f{move=120,pass=-1,bal=0,type=b},width=3},low=h{pass=10}f{move=120,pass=-1,bal=0,type=b},vert=120,rat=0.8}f{move=80,pass=-1,bal=0,type=b})},vert=10000,rat=0.8,type=h}x{bal=0}f{move=80,pass=-1,bal=0}\""
    # cmd = f"{gmap} -m{strat} {com_mat_grf} {phys_tgt} > {res_file}"
    if not os.path.exists(res_file): 
        print("\tRunning Scotch gmap...")
        cmd = f"{gmap} -cb {com_mat_grf} {phys_tgt} > {res_file}"
        # print(cmd)
        os.system(cmd)

    target_oversub = math.ceil(nb_procs/nb_vertices)
    max_oversub, sigma_scotch = analyze_scotch_mapping(res_file)
    if max_oversub > target_oversub : 
        print(f"\t=====> Error: Scotch computed a unbalanced solution ({max_oversub})!")
        quit()
    else:
        print(f"\tScotch Hop-Bytes = {compute_hop_bytes(mat, sigma_scotch, hop_matrix_unique, None)}")

    return sigma_tm, sigma_scotch

def parse_topomach_mapping(res_file, ids, logical_numbering = False):
    with open(res_file) as file:
        for line in file:        
            if "TopoMatch" in line:
                values = line.split(":")[1].strip()
                sigma_phys = values.split(",")
                sigma = []
                if logical_numbering:
                    sigma = [int(val) for val in sigma_phys]
                else:
                    for val in sigma_phys:
                        sigma.append(ids.index(val))

        
    return np.array(sigma)

def topomatch_sub_fugaku(out_filename, file_mat, mat, ids, nb_procs, hop_matrix_unique):

    print("\t==== Topomatch on Sub-Fugaku topology")
    # remove duplicates
    ids = sorted(set(ids))

    nb_slots = len(ids)
    
    # save ids to file
    sub_nodes_file =  tmp_dir+"/sub_nodes.txt"
    with open(sub_nodes_file,"w") as file:
        file.write(str(len(ids))+" "+" ".join(ids))

    # create a sub arch from the full Fugaku topology by extracting only the node we use
    tofu_tgt = tmp_dir+"/"+out_filename+"_tofu.tgt"
    if not os.path.exists(tofu_tgt):
        cmd= f"{amk_grf} {tofu_grf} -l{sub_nodes_file} > {tofu_tgt}"
        print("\tGenerating tgt topology file from subnodes...");
        os.system(cmd)

    # Now run topomatch on this topology
    cmd = f"{mapping} -t {tofu_tgt} -c {file_mat} -a 0 "
    
    target_oversub = math.ceil(nb_procs/nb_slots)
    if target_oversub > 1:
        cmd += f" -o {target_oversub}"
        
    res_file = tmp_dir+"/"+out_filename+"_sub_fugaku_mapping.map"

    if not os.path.exists(res_file):
        cmd += f" > {res_file}"
        print(f"\tRunning TopoMatch to {res_file}...")
        os.system(cmd)

    # Output TopoMatch result
    # cmd = f"grep TopoMatch {res_file}"
    # os.system(cmd)
        
    sigma = parse_topomach_mapping(res_file, ids)
    print(f"\tTM-1 Hop-Bytes = {compute_hop_bytes(mat, sigma, hop_matrix_unique, None)}")
    return sigma 
   
def com_mat_to_grf(mat, com_mat_grf, sparse_quantile= 0):

    threshold = np.percentile(mat,sparse_quantile*100)  # Use 10th percentile (100 - 90)
    
    #set to zeroes values under the threshold 
    mat[mat < threshold] = 0
    min_val = np.min(mat[mat>0])
    scale_factor = 1
    # scale_factor = 100 / min_val
    # scale_factor = min_val / min_val
    mat = np.round(mat * scale_factor).astype(int)
    
    # plot_mat(mat)
    
    # weight = int(2**31 /(10 * len(mat)))
    weight = 2
    
    nb_edges = 0
    str_res = ""
    for i, values in enumerate(mat):
        values = [str(val)+" "+str(j) for j,val  in enumerate(values) if val != 0 ]
        n = len(values)
        nb_edges += n
        str_res += f"{i} {weight} {n} {' '.join(values)}\n"


    with open(com_mat_grf,"w") as file:
        file.write("0\n")
        # Number_of_verticies number_of_"edges"
        nb_vertices = len(mat)
        file.write(f"{nb_vertices} {nb_edges}\n")
        # Chaco format
        file.write("0 111\n") # 0 : base value (as in C) ; 1 : vertex weights ;  1 : edge weights ; 1 : vertex label provided
        file.write(str_res)

def analyze_scotch_mapping(res_file):
    mapping = defaultdict(list)
    with open(res_file) as file:
        nb_procs = int(file.readline().strip())
        sigma = [None] * nb_procs
        for line in file:
            # print (line)
            rank, proc_id = map(int, line.split())
            # print(rank,"->",proc_id)
            mapping[proc_id].append(rank)
            sigma[rank] = proc_id


    max_oversub = 0
    for proc_id, ranks in mapping.items():
        n = len(ranks)
        # print(f"Processor {proc_id} ({n}): Ranks {ranks}")
        if n > max_oversub:
            max_oversub = n

    return max_oversub, np.array(sigma)


def refine_solution(mat, ids, res_file, h):
    mapping = defaultdict(list)
    with open(res_file) as file:
        nb_procs = int(file.readline().strip())
        sigma = [] * nb_procs
        for line in file:
            # print (line)
            rank, proc_id = map(int, line.split())
            sigma[rank] = proc_id
            mapping[proc_id].append(rank)

    
    max_oversub = 0
    for proc_id, ranks in mapping.items():
        n = len(ranks)
        print(f"Processor {proc_id} ({n}): Ranks {ranks}")
        if n > max_oversub:
            max_oversub = n

def do_process_mapping_for_diff_mapping(comm_mat: np.ndarray, physical_node_shape: list, phys_nodes_coord_6D: np.ndarray, hop_matrix: np.ndarray, process_mapping_methods: list, out_filename: str, comm_filename: str) -> dict:
    """
    Perform process mapping using the specified methods.

    Parameters:
        comm_mat (numpy.ndarray): Communication pattern matrix.
        physical_node_shape (list): Shape of the physical node (e.g., 3D torus dimensions).
        phys_nodes_coord_6D (list of tuples): 6D coordinates of the physical nodes.
        hop_matrix (numpy.ndarray): Hop matrix (distance matrix).
        process_mapping_methods (list of ProcessMappingMethod): List of process mapping methods to use.
            Options:
                - ProcessMappingMethod.TM_SUB_FUGAKU
                - ProcessMappingMethod.TM_PHYS
                - ProcessMappingMethod.SCOTCH
        out_filename (str): Output filename to store results.
        comm_filename (str): Communication pattern filename for input.
            
    Returns:
        dict: A dictionary containing the communication matrix for each process mapping method.
              Keys are the method names, and values are the resulting communication matrices.
    """
    # Get unique rows without sorting
    # Extract 6D coordinates that are unique but keep the order of the original coordinates (s they are sorted by ranks
    # The reason for getting the unique coordinates is that 4 process may have the same coordinates (4 MPI processes per node)
    _, indices = np.unique(phys_nodes_coord_6D, axis=0, return_index=True)
    unique_phys_nodes_coord_6D = phys_nodes_coord_6D[np.sort(indices)]

    hop_matrix_unique = get_hop_matrix_from_coords_numpy(physical_node_shape, unique_phys_nodes_coord_6D)
    # sigma = np.repeat(np.arange(len(phys_unique_6D)), len(mat)//len(phys_unique_6D))
    # if compute_hop_bytes(mat, sigma, hop_matrix_unique, hop_matrix) != hop_bytes.sum():
    #     print("compute_hop_bytes() failed to find the correct value.\n")
    #     quit()

    phys_nodes_ids=[]
    # convert 6D coordinate into node id
    for node in phys_nodes_coord_6D:
        phys_nodes_ids.append(str(coord_2_id(node[0], node[1], node[2], node[3] , node[4], node[5])))

    comm_mat_dict = {}

    if ProcessMappingMethod.RR in process_mapping_methods:
        comm_mat_dict[ProcessMappingMethod.RR] = comm_mat

    if ProcessMappingMethod.TM_SUB_FUGAKU in process_mapping_methods:
        sigma_tm_sub_fugaku = topomatch_sub_fugaku(out_filename, comm_filename, comm_mat, phys_nodes_ids, len(comm_mat), hop_matrix_unique)
        mat_tm_sub_fugaku = permute_comm_mat(comm_mat, sigma_tm_sub_fugaku)
        comm_mat_dict[ProcessMappingMethod.TM_SUB_FUGAKU] = mat_tm_sub_fugaku
    
    if ProcessMappingMethod.TM_PHYS in process_mapping_methods:
        sigma_tm_phys, _ = topomatch_phys_grid(out_filename, comm_filename, comm_mat, phys_nodes_ids, hop_matrix, len(comm_mat), hop_matrix_unique)
        mat_tm_phys = permute_comm_mat(comm_mat, sigma_tm_phys)
        comm_mat_dict[ProcessMappingMethod.TM_PHYS] = mat_tm_phys

    if ProcessMappingMethod.SCOTCH in process_mapping_methods:
        _, sigma_scotch = topomatch_phys_grid(out_filename, comm_filename, comm_mat, phys_nodes_ids, hop_matrix, len(comm_mat), hop_matrix_unique)
        mat_scotch = permute_comm_mat(comm_mat, sigma_scotch)
        comm_mat_dict[ProcessMappingMethod.SCOTCH] = mat_scotch

    return comm_mat_dict
