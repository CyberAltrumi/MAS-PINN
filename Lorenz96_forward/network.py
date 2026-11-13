import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from LV_half_observed.network import ModelMix, RandomFourierFeaturesNN, compute_full_RFF_derivatives_LV as basis_with_derivative   # shared trunk + RFF basis
