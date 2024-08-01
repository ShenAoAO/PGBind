# PGBind
Ao Shen
PGBind: Pocket-guided explicit attention learning for protein-ligand docking
# Abstract
As more and more protein structures are discovered, blind protein-ligand docking will play an important role in drug discovery because it can predict protein-ligand complex conformation without pocket information on the target proteins. Recently, deep learning-based methods have made significant advancements in blind protein-ligand docking, but their protein features are suboptimal because they do not fully consider the difference between potential pocket regions and non-pocket regions in protein feature extraction. In this work, we propose a pocket-guided strategy for guiding the ligand to dock to potential docking regions on a protein. To this end, we design a plug-and-play module to enhance the protein features, which can be directly incorporated into existing deep learning-based blind docking methods. The proposed module first estimates potential pocket regions on the target protein and then leverages a pocket-guided attention mechanism to enhance the protein features. Experiments are conducted on integrating our method with EquiBind and FABind, and the results show that their blind-docking performances are both significantly improved and new start-of-the-art performance is achieved by integration with FABind. 
# Equibind-PG
## Training a model yourself and using those weights
To train the model yourself, run:
```shell

    python train.py --config=configs_clean/RDKitCoords_flexible_self_docking.yml
```

The model weights are saved in the `runs` directory.\
You can also start a tensorboard server ``tensorboard --logdir=runs`` and watch the model train. \
To evaluate the model on the test set, change the ``run_dirs:`` entry of the config file `inference_file_for_reproduce.yml` to point to the directory produced in `runs`.
Then you can run``python inference.py --config=configs_clean/inference_file_for_reproduce.yml`` as above!

## Predict Binding Structures!

In the config file `configs_clean/inference.yml` set the path to your input data folder `inference_path: path_to/my_data_folder`.  
Then run:
```shell
    python inference.py --config=configs_clean/inference.yml
```
# FABind-PG

## Re-training
```shell
data_path=pdbbind_2020
# write the default accelerate settings
python -c "from accelerate.utils import write_basic_config; write_basic_config(mixed_precision='no')"
# "accelerate launch" will run the experiments in multi-gpu if applicable 
accelerate launch fabind/main_fabind.py \
    --batch_size 3 \
    -d 0 \
    -m 5 \
    --data-path $data_path \
    --label baseline \
    --addNoise 5 \
    --resultFolder ./results \
    --use-compound-com-cls \
    --total-epochs 500 \
    --exp-name train_tmp \
    --coord-loss-weight 1.0 \
    --pair-distance-loss-weight 1.0 \
    --pair-distance-distill-loss-weight 1.0 \
    --pocket-cls-loss-weight 1.0 \
    --pocket-distance-loss-weight 0.05 \
    --lr 5e-05 --lr-scheduler poly_decay \
    --distmap-pred mlp \
    --hidden-size 512 --pocket-pred-hidden-size 128 \
    --n-iter 8 --mean-layers 4 \
    --refine refine_coord \
    --coordinate-scale 5 \
    --geometry-reg-step-size 0.001 \
    --rm-layernorm --add-attn-pair-bias --explicit-pair-embed --add-cross-attn-layer \
    --noise-for-predicted-pocket 0 \
    --clip-grad \
    --random-n-iter \
    --pocket-idx-no-noise \
    --pocket-cls-loss-func bce \
    --use-esm2-feat
```
## Evaluation
```shell
data_path=pdbbind2020
ckpt_path=ckpt/best_model.bin

python fabind/test_fabind.py \
    --batch_size 4 \
    --data-path $data_path \
    --resultFolder ./results \
    --exp-name test_exp \
    --ckpt $ckpt_path \
    --local-eval
```

# Train pocket_label
cd Equibind_clean
```shell
python train_pocket
```
