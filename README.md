# Fugaku_mpi_process_mapping_tool
## Configuration
* in subroutines
* `cp  config_template.py config.py`
* Then edit config.py and customize it to your environment

## Usage
### Cache file
The tmp file is used as a directory for cache file where computed data are stored to avoid to recompute them later
### analyze_coords.py 
* usage: analyze_coords.py [-h] [-v] -f <coord filename> [-m <comm filename>]
* Example
`./analyze_coords.py -f data/coords_out.1.0 -m data/comm_data_volume_fp32_mag240M_paper_cites_paper_mode_post_aggr_size_2048_layer_forward0.txt -v`