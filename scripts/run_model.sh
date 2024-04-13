#!/bin/bash

#SBATCH -D /well/woolrich/users/olt015/Cho2024_MEEG_RSN/results/dynamic_no_struct/camcan
#SBATCH -p gpu_short
#SBATCH --gres gpu:1
#SBATCH --mem-per-gpu 80G
#SBATCH --constraint "a100|v100"

# Check the number of arguments
if [ $# -lt 6 ]; then
    echo "Too few arguments. Usage: $0 [lemon|camcan] [hmm|dynemo] [int] [int] [full|split1|split2] [subject|standard]"
    exit 1
fi

# Validate CLI input arguments
if [ "${1}" != "lemon" ] && [ "${1}" != "camcan" ]; then
    echo "Invalid data name. Usage: $0 [lemon|camcan] [hmm|dynemo] [int] [int] [full|split1|split2] [subject|standard]"
    exit 1
fi
if [ "${2}" != "hmm" ] && [ "${2}" != "dynemo" ]; then
    echo "Invalid model type. Usage: $0 ${1} [hmm|dynemo] [int] [int] [full|split1|split2] [subject|standard]"
    exit 1
fi
if ! [[ ${3} =~ ^[0-9]+$ ]]; then
    echo "Invalid run ID. Usage: $0 ${1} ${2} [int] [int] [full|split1|split2] [subject|standard]"
    exit 1
fi
if ! [[ ${4} =~ ^[0-9]+$ ]]; then
    echo "Invalid number of states. Usage: $0 ${1} ${2} ${3} [int] [full|split1|split2] [subject|standard]"
    exit 1
fi
if [ "${5}" != "full" ] && [ "${5}" != "split1" ] && [ "${5}" != "split2" ]; then
    echo "Invalid model type. Usage: $0 ${1} ${2} ${3} ${4} [full|split1|split2] [subject|standard]"
    exit 1
fi
if [ "${6}" != "subject" ] && [ "${6}" != "standard" ]; then
    echo "Invalid model type. Usage: $0 ${1} ${2} ${3} ${4} ${5} [subject|standard]"
    exit 1
fi
echo "Input arguments submitted. Data Name: ${1}, MODEL TYPE: ${2}, RUN #: ${3}, NUM STATES: ${4}, DATA TYPE: ${5}, STRUCTURAL: ${6}"

# Set up your environment
module load Anaconda3/2022.05
eval "$(conda shell.bash hook)"
module load cuDNN

# Run scripts
conda activate /well/woolrich/users/olt015/conda/skylake/envs/osld
python /well/woolrich/users/olt015/Cho2024_MEEG_RSN/scripts/${1}_${2}.py ${3} ${4} ${5} ${6}
