## FashionIQ

Details on each step are as follows.

> [!NOTE]
> To reproduce our reported results, please first download the checkpoints and top-k files and save them to paths noted in  [DOWNLOAD.md](DOWNLOAD.md).
>
> We provide all intermediate checkpoints and top-k files, so you could skip training steps and (for instance) develop directly on our Stage 2 model.

##

### Stage 1

#### Train

```bash
# Optional: comet experiment logging --api-key and --workspace
python src/stage1_train.py --dataset FashionIQ \
    --api-key <COMET_API_KEY> --workspace <COMET_WORKSPACE> \
    --blip-learning-rate 2e-5 --blip-max-epoch 20 --num-epochs 20 \
    --batch-size 512 --blip-bs 16 \
    --transform targetpad --target-ratio 1.25 \
    --save-training --save-best --validation-frequency 1 \
    --experiment-name BLIP_stageI_b512_2e-5_cos20 \
    --train
```

 - by default, the script will create a new folder at `./models/<STAGE1_EXP_FOLDER>` for saving the checkpoints. 

##

#### Validate + extracting the top-k files

```bash
python src/validate.py --dataset fashionIQ \
    --stage1-path <STAGE1_EXP_FOLDER> \
    --save-topk --k 200
```

 - replace `<STAGE1_EXP_FOLDER>` with the Stage 1 experiment folder;
 - the top-k files will be saved at `./models/<STAGE1_EXP_FOLDER>/`.

<details>
  <summary><b>Alternatively,</b> to reproduce our stage 1 results, click here</summary>
&emsp; 

```bash
python src/validate.py --dataset fashionIQ \
    --stage1-path stage1/fashionIQ/ \
    --save-topk --k 200
```
 - which will also save the top-k files at `./models/stage1/fashionIQ/`. They should be identical to our provided files.

&emsp; 
</details>


##

### Stage 2

#### Train

```bash
# Optional: comet experiment logging --api-key and --workspace
python src/stage2_train.py --dataset FashionIQ \
    --api-key <COMET_API_KEY> --workspace <COMET_WORKSPACE> \
    --blip-learning-rate 2e-5 --blip-max-epoch 50 \
    --num-epochs 50 --batch-size 16 --blip-bs 16 \
    --transform targetpad --target-ratio 1.25 \
    --save-training --save-best \
    --top-k-path <TOPK_FILE_PATH> --K-value 100 \
    --validation-frequency 1 \
    --experiment-name BLIP_stageII_b16_2e-5_cos50_k100+[BLIP_stageI_b512_2e-5_cos20] \
    --blip-model-path <STAGE1_EXP_FOLDER> \
    --train
```

 - replace `<TOPK_FILE_PATH>` with  the path of the generated files, you might wish to modify [this function](https://github.com/Cuberick-Orion/Candidate-Reranking-CIR/blob/f31a15a704d6e742746c99745ccb17d46ff1394e/src/utils.py#L181);
 - replace `<STAGE1_EXP_FOLDER>` with the Stage 1 experiment folder;
 - by default, the script will create a new folder at `./models/<STAGE2_EXP_FOLDER>` for saving the checkpoints.



<details>
  <summary><b>Alternatively,</b> you can also use our extracted top-k file and/or our stage 1 checkpoint, click here</summary>
&emsp; 


```bash
python src/stage2_train.py --dataset FashionIQ \
    --api-key <COMET_API_KEY> --workspace <COMET_WORKSPACE> \
    --blip-learning-rate 2e-5 --blip-max-epoch 50 \
    --num-epochs 50 --batch-size 16 --blip-bs 16 \
    --transform targetpad --target-ratio 1.25 \
    --save-training --save-best \
    --top-k-path BLIP_stageI_b512_2e-5_cos20 --K-value 100 \
    --validation-frequency 1 \
    --experiment-name BLIP_stageII_b16_2e-5_cos50_k100+[BLIP_stageI_b512_2e-5_cos20] \
    --blip-model-path stage1/fashionIQ \
    --train
```

&emsp; 
</details>

##

#### Validate

```bash
python src/validate_stage2.py --dataset fashionIQ \
    --stage1-path <STAGE1_EXP_FOLDER> --stage2-path <STAGE2_EXP_FOLDER> \
    --top-k-path <TOPK_FILE_PATH> \
    --k 100
```

 - replace `<TOPK_FILE_PATH>` with  the path of the generated files, you might wish to modify [this function](https://github.com/Cuberick-Orion/Candidate-Reranking-CIR/blob/f31a15a704d6e742746c99745ccb17d46ff1394e/src/utils.py#L181);
 - replace `<STAGE1_or_2_EXP_FOLDER>` with the previous experiment folders.

<details>
  <summary><b>Alternatively,</b> to reproduce our stage 2 results, click here</summary>
&emsp; 

```bash
python src/validate_stage2.py --dataset fashionIQ \
    --stage1-path stage1/fashionIQ --stage2-path stage2/fashionIQ \
    --top-k-path BLIP_stageI_b512_2e-5_cos20 \
    --k 100
```

&emsp; 
</details>
