gpu=$1
if [ -z $gpu ]; then
    gpu=1
export CUDA_VISIBLE_DEVICES=$gpu
# 
file=autodidact.py
python $file \
    2>&1  </dev/null | tee $file.log
