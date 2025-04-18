gpu=$1
if [ -z $gpu ]; then
    gpu=1
fi
export CUDA_VISIBLE_DEVICES=$gpu
export WANDB_API_KEY=$(cat .wandb_api_key)
wandb login
# 
file=autodidact.py
python $file \
    2>&1  </dev/null | tee $file.log
