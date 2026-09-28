from datetime import datetime
from datetime import datetime
import time
import sys
import os
import warnings
import logging

# Comprehensive warning suppression
warnings.filterwarnings('ignore')
os.environ['RDKIT_QUIET'] = '1'
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow warnings
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# Suppress all logging
logging.getLogger().setLevel(logging.ERROR)
for logger_name in ['rdkit', 'tensorflow', 'torch', 'transformers', 'pytorch']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# Also set RDKit logger to suppress messages
try:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    pass

# Redirect stderr to suppress various C++ warnings (like NNPACK)
import io
from contextlib import redirect_stderr
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from pep_compass.optimization.baselines.latent_cmaes import LatentCMAESOptimizer
from pep_compass.optimization.black_box.apex_black_box import APEXBlackBox
from pep_compass.optimization.black_box.battleamp_black_box import BattleAMPBlackBox
from pep_compass.optimization.black_box.csv_observer import CSVObserver
from pep_compass.optimization.black_box.toxipep_black_box import ToxiPepBlackBox
from pep_compass.optimization.black_box.hydramp_black_box_wrapper import HydrAMPBlackBoxWrapper
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# discrete_black_box = APEXBlackBox(
#     mic_aggregate="mean",
#     mic_bacteria=[1, 2, 3],
#     device=DEVICE,
# )

# discrete_black_box = BattleAMPBlackBox(device=DEVICE)

discrete_black_box = ToxiPepBlackBox(device=DEVICE)

# Wrap it with HydrAMPBlackBoxWrapper to enable latent space optimization
black_box = HydrAMPBlackBoxWrapper(
    black_box=discrete_black_box,
    device=DEVICE,
    jacobian_eps=1e-3,
    field_eps=1e-3,
)

observer = CSVObserver(maximize=True)
black_box.set_observer(observer)

optimizer = LatentCMAESOptimizer(
    black_box=black_box,
    device=DEVICE,
)


proteins = {
    "middle-1": ("FLYKWWIRIGRLKL", 5),
    "jurand-4": ("KYCRRFRWLTFRWL", 5),
    "jurand-2": ("KFRNRHRWKFKLIFRN", 5),
    "jurand-7": ("KKYWLIRKWIRLWFLT", 5),
    "mammuthusin-3": ("KTLKIIRLLF", 5),
    "hydrodamin-2": ("RMARNLVRYVQGLKKKKVI", 5),
}

for i in range(5):
    for name, (sequence, _) in proteins.items():
        print(f"Starting optimization for {name}, iteration {i+1}")
        rng_seed = int(time.time())  # Create a unique rng_seed for each iteration
        observer.initialize_observer(
            black_box.get_black_box_info(),
            {
                "experiment_id": f"{sequence}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "experiment_path": "./results/cma_es",
            },
            rng_seed,
            encoder_decoder=black_box.encoder_decoder,
        )
        # Suppress stderr during optimization to hide RDKit and NNPACK warnings
        devnull = io.StringIO()
        old_stderr = sys.stderr
        try:
            sys.stderr = devnull
            with redirect_stderr(devnull):
                optimizer.optimize(
                    evaluation_budget=1400, starting_point=sequence, rng_seed=rng_seed
                )
        finally:
            sys.stderr = old_stderr

