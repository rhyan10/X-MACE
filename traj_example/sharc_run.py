import sys, os
import mace
from mace.calculators.sharcinterface import SHARC_NN

models_e = ["/home/rhyan/X-MACE_lorenz/energy_model_2_duation.model"]
models_s = ["/home/rhyan/X-MACE_lorenz/socs_model_duation.model"]

th = None # for active learning, e.g., {'energy': 0.004}
nn = SHARC_NN(modelpath_e=models_e, modelpath_s=models_s, 
   atoms="FeCCCCCCOOOOOO", # symbols of sample molecule
   n_states={'n_singlets': 4, 'n_doublets': 0, 'n_triplets': 6, 'n_quartets': 0, 'n_quintets': 3}, # dict of state numbers
   thresholds=th,
   cutoff=5.0, # for building representation
   nac_key="smooth_nacs", # model trained on smoothed nacs
   properties=['energy','forces'] # properties predicted by NN
)
nn.run_sharc("./input",0)
