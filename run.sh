gpu=$1
if [ -z $gpu ]; then
    gpu=0
fi
export CUDA_VISIBLE_DEVICES=$gpu
# 
file=generate_data.py
python $file \
    2>&1  </dev/null | tee $file.log
