set -ex

CONFIG=$1
GPUS=$4
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$/data_raid5_21T/zgh/ZGh/mmdetection/tools/train.py \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
