# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test

Train MamBEV
```
./tools/dist_train.sh ./path/to/config num_gpu

e.g.
./tools/dist_train.sh ./projects/configs/MamBEV/mambev_tiny_t3.py 8
```

Eval MamBEV
```
./tools/dist_test.sh ./path/to/config ./path/to/ckpts.pth num_gpu
```
Note: using 1 GPU to eval can obtain slightly higher performance because continuous video may be truncated with multiple GPUs. By default we report the score evaled with 8 GPUs.



# Visualization 

see [visual.py](../tools/analysis_tools/visual.py)
