#!/bin/bash

rm -rf rtdocs chroma_db
wget -r -l inf -A "*.html" -P rtdocs --no-parent --cut-dirs=4 https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/
# wget -r -A.html -P rtdocs --no-parent --cut-dirs=4 https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/
