# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

GAMMA = 2.2
SEED=0

def set_random_seed():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Ensures deterministic behavior (slightly slower performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False