#!/bin/bash

rm -rf rtdocs chroma_db
wget -r -l inf -A "*.html" -P rtdocs --no-parent --cut-dirs=2 -X /deeplearning/nccl/archives https://docs.nvidia.com/deeplearning/nccl/

python build_chroma.py
