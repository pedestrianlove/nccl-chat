#!/bin/bash

rm -rf rtdocs
rm -rf chroma_db
wget -r -l inf -A "*.html" -P rtdocs --no-parent -X /deeplearning/nccl/archives,/deeplearning/nccl/release-notes,/deeplearning/nccl/sla,/deeplearning/nccl/bsd https://docs.nvidia.com/deeplearning/nccl/
wget -r -l inf -A "*.html" -P rtdocs --no-parent https://pytorch.org/docs/stable/ 
wget -r -l inf -A "*.html" -P rtdocs --no-parent https://hta.readthedocs.io/en/latest/

python build_chroma.py
