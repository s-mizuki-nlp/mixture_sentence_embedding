#!/bin/sh

# rsync -avr tsubame:log_* ./log/
local_dir="./log/"
hostname=$1

if [ "$hostname" = "tsubame" ]; then
	remote_dir="/home/4/18D30111/mixture_sentence_embedding"
elif [ "$hostname" = "iris" ]; then
	remote_dir="/home/sakae/mixture_sentence_embedding"
else
	echo "undefined hostname: ${hostname}"
	exit 1
fi

echo "rsync logs from ${hostname}:${remote_dir} to ./log/"

rsync -avr --dry-run ${hostname}:${remote_dir}/log_*.log.* ${local_dir}

echo "----"
read -p "continue(y) or abort(N): " yesno
case "$yesno" in [yY]*) ;; *) echo "abort." ; exit 1 ;; esac

rsync -avr ${hostname}:${remote_dir}/log_*.log.* ${local_dir}

echo "finished. good-bye."
