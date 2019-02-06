#!/bin/sh

# typical usage:
# ./rsync_models_from_tsubame.sh tsubame '*ptb_17*'

local_config_dir="./config/"
local_saved_model_dir="./saved_model/"
hostname=$1
filenames=$2

if [ "$hostname" = "tsubame" ]; then
	remote_dir="/home/4/18D30111/mixture_sentence_embedding"
elif [ "$hostname" = "iris" ]; then
	remote_dir="/home/sakae/mixture_sentence_embedding"
else
	echo "undefined hostname: ${hostname}"
	exit 1
fi

echo "rsync configs from ${hostname}:${remote_dir} to ${local_config_dir}"

rsync -avr --dry-run ${hostname}:${remote_dir}/config/${filenames}.py ${local_config_dir}

echo "----"
read -p "continue(y) or abort(N): " yesno
case "$yesno" in [yY]*) ;; *) echo "abort." ; exit 1 ;; esac

rsync -avr ${hostname}:${remote_dir}/config/${filenames}.py ${local_config_dir}


echo "rsync saved models from ${hostname}:${remote_dir} to ${local_saved_model_dir}"

rsync -avr --dry-run ${hostname}:${remote_dir}/saved_model/${filenames} ${local_saved_model_dir}

echo "----"
read -p "continue(y) or abort(N): " yesno
case "$yesno" in [yY]*) ;; *) echo "abort." ; exit 1 ;; esac

rsync -avr ${hostname}:${remote_dir}/saved_model/${filenames} ${local_saved_model_dir}

echo "finished. good-bye."
