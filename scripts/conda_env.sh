# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

conda env create -f environment.yml
conda activate iris
pip install torch_scatter -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch # tested with tinycudann-1.7
