import torch
import pandas as pd
import numpy as np
import os
from tools import base
from tools import cdhit

GPU_ID = 1
BATCH_SIZE = 64
__all__ = ['torch', 'pd', 'np', 'os', "GPU_ID", "BATCH_SIZE", "base", "cdhit", "preProcessed"]
ec_mic10_best_rf_fts = ['type7raac13glmd8lambda-correlation',
 'type14raac7glmd8lambda-correlation',
 'type14raac6glmd8lambda-correlation',
 'type7raac15glmd8lambda-correlation',
 'type7raac10glmd8lambda-correlation',
 'type7raac9glmd8lambda-correlation',
 'type8raac4glmd2lambda-correlation',
 'type7raac14glmd8lambda-correlation',
 'type8raac4glmd8lambda-correlation',
 'type16raac11glmd2lambda-correlation',
 'type3Braac14glmd8lambda-correlation',
 'type4raac5glmd8lambda-correlation',
 'type7raac16glmd8lambda-correlation',
 'type7raac11glmd8lambda-correlation',
 'CKSAAP_gap2',
 'type7raac12glmd8lambda-correlation',
 'type16raac10glmd3lambda-correlation',
 'type14raac5glmd2lambda-correlation',
 'type7raac6glmd8lambda-correlation',
 'type16raac11glmd3lambda-correlation',
 'type7raac18glmd8lambda-correlation',
 'type3Braac5glmd9lambda-correlation',
 'type14raac5glmd8lambda-correlation',
 'type3Braac11glmd2lambda-correlation',
 'type7raac19glmd8lambda-correlation',
 'type14raac6glmd2lambda-correlation',
 'type14raac7glmd3lambda-correlation',
 'type7raac12glmd5lambda-correlation',
 'type14raac9glmd8lambda-correlation',
 'type7raac17glmd8lambda-correlation']