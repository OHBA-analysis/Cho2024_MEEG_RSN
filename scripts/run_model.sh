#!/bin/bash

#SBATCH -D /well/woolrich/users/olt015/Cho2023_EEG_RSN/results/dynamic/lemon
#SBATCH -p gpu_short
#SBATCH --gres gpu:1
#SBATCH --mem-per-gpu 80G
#SBATCH --constraint "a100|v100"

# Check the number of arguments
if [ $# -lt 4 ]; then
    echo "Too few arguments. Usage: $0 [lemon|camcan] [hmm|dynemo] [int] [int]"
    exit 1
fi

# Validate CLI input arguments
if [ "${1}" != "lemon" ] && [ "${1}" != "camcan" ]; then
    echo "Invalid data name. Usage: $0 [lemon|camcan] [hmm|dynemo] [int] [int]"
    exit 1
fi
if [ "${2}" != "hmm" ] && [ "${2}" != "dynemo" ]; then
    echo "Invalid model type. Usage: $0 ${1} [hmm|dynemo] [int] [int]"
    exit 1
fi
if ! [[ ${3} =~ ^[0-9]+$ ]]; then
    echo "Invalid run ID. Usage: $0 ${1} ${2} [int] [int]"
    exit 1
fi
if ! [[ ${4} =~ ^[0-9]+$ ]]; then
    echo "Invalid number of states. Usage: $0 ${1} ${2} ${3} [int]"
    exit 1
fi
echo "Input arguments submitted. Data Name: ${1}, MODEL TYPE: ${2}, RUN #: ${3}, NUM STATES: ${4}"

# Set up your environment
module load Anaconda3/2022.05
eval "$(conda shell.bash hook)"
module load cuDNN

# Run scripts
conda activate /well/woolrich/users/olt015/conda/skylake/envs/osld
python /well/woolrich/users/olt015/Cho2023_EEG_RSN/scripts/${1}_${2}.py ${3} ${4}
