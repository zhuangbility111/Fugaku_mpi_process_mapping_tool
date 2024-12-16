from enum import Enum

class ProcessMappingMethod(Enum):
    RR = "RR" # default mapping method
    TM_SUB_FUGAKU = "TM_Sub_Fugaku" # TM_Sub_Fugaku
    TM_PHYS = "TM_Phys" # TM_Phys
    SCOTCH = "Scotch" # Scotch
