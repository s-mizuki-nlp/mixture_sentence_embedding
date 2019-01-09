#!/bin/sh
#$ -cwd
#$ -l s_gpu=1
#$ -l h_rt=24:00:00

. /etc/profile.d/modules.sh
module load python/3.6.5
module load intel
module load cuda
module load openmpi

source pytorch_0.4.1/bin/activate

gpus=0
while getopts c:s:d:g:lv OPT
do
    case $OPT in
        "c" ) config_module="$OPTARG" ;;
        "d" ) device="$OPTARG" ;;
        "s" ) save_dir="$OPTARG" ;;
        "g" ) gpus="$OPTARG" ;;
        "l" ) flags="$flags --log_validation_only" ;;
        "v" ) flags="$flags --verbose" ;;
        * ) echo "Usage: [-c config=module] [-d device] [-s save_dir] [-l log_validation_only] [-v verbose]" 1>&2
            exit 1 ;;
    esac
done

python ./train_with_rnn.py --config_module=$config_module --save_dir=$save_dir --device=$device $flags --gpus=$gpus
