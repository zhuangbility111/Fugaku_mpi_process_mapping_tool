# Customize this to your environment
# Save it as config.py (config.py is in .gitignore)
# where is the analyze_coords.py scripts
prefix = "/Users/ejeannot/recherche/src/Fugaku-Topology-Management/TopologyGenerator/Fugaku_mpi_process_mapping_tool"
tofu_grf = prefix+"/data/tofu.grf"
tmp_dir = prefix+"/tmp" # Here I save data to avoid to recompute them: it acts as a cache 

# where are TopoMatch and Scotch binaries
bin_path = "/usr/local/bin"
amk_grf = bin_path+"/amk_grf"
gmap = bin_path+"/gmap"
mapping = bin_path+"/mapping"

