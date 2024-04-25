# ExplainHM - Debugging
Official PyTorch implementation for the paper - **Towards Explainable Harmful Meme Detection through Multimodal Debate between Large Language Models**.

(**WWW 2024**: The ACM Web Conference 2024, May 2024, Singapore.) [[`paper`](https://arxiv.org/pdf/2401.13298.pdf)]


## Install

```bash
conda create -n meme python=3.8
conda activate meme
pip install -r requirements.txt
```

## Data

Please refer to [data](https://github.com/HKBUNLP/ExplainHM-WWW2024/tree/main/data).

## Training
```bash
export DATA="/path/to/data/folder"
export LOG="/path/to/save/ckpts/name"

rm -rf $LOG
mkdir $LOG

CUDA_VISIBLE_DEVICES=0,1 python run.py with data_root=$DATA \
    num_gpus=2 num_nodes=1 task_train per_gpu_batchsize=8 batch_size=32 \
    clip32_base224 text_t5_base image_size=224 vit_randaug max_text_len=512 \
    log_dir=$LOG precision=32 max_epoch=10 learning_rate=5e-4
```

## Inference

```bash
export DATA="/path/to/data/folder"
export LOG="/path/to/log/folder"

CUDA_VISIBLE_DEVICES=0,1 python run.py with data_root=$DATA \
    num_gpus=2 num_nodes=1 task_train per_gpu_batchsize=8 batch_size=32 test_only=True \
    clip32_base224 text_t5_base image_size=224 vit_randaug \
    log_dir=$LOG precision=32 \
    max_text_len=512 load_path="/path/to/label_learn.ckpt"
```

## Citation

```
@inproceedings{lin2024explainable,
    title={Towards Explainable Harmful Meme Detection through Multimodal Debate between Large Language Models},
    author={Hongzhan Lin and Ziyang Luo and Wei Gao and Jing Ma and Bo Wang and Ruichao Yang},
    booktitle={The ACM Web Conference 2024},
    year={2024},
    address={Singapore},
}
```

## Acknowledgements

The code is based on [ViLT](https://github.com/dandelin/ViLT) and [METER](https://github.com/zdou0830/METER/tree/main).

