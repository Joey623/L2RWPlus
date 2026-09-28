# L2RW+

Welcome to our repo for L2RW+.

## Environment

Run the following commands to clone the repository and install the dependencies in a Conda environment:

```bash
git clone https://github.com/Joey623/L2RWPlus.git
cd L2RWPlus
conda create -n l2rwplus -c conda-forge python=3.11.15 pip -y
conda activate l2rwplus
python -m pip install -r requirement.txt \
    'torch==2.13.0+cu130' 'torchvision==0.28.0+cu130' \
    --extra-index-url https://download.pytorch.org/whl/cu130
```

These commands install the CUDA 13.0 build of PyTorch. Use a Linux machine with a compatible NVIDIA driver and keep the `l2rwplus` environment activated for training and evaluation.

## Data and checkpoints

The prepared data are available on [Hugging Face](https://huggingface.co/datasets/yan-jiang/L2RWPLUS/tree/main/data). Place the data and checkpoints in the following structure:

```text
L2RWPLUS/
├── data/
│   ├── lmdb/
│   │   ├── sysu.lmdb
│   │   ├── regdb.lmdb
│   │   ├── llcm.lmdb
│   │   ├── vcm.lmdb
│   │   └── bupt.lmdb
│   └── indices/
│       ├── sysu_uci.json
│       ├── regdb_uci.json
│       ├── llcm_uci.json
│       ├── vcm_uci.json
│       └── bupt_uci.json
└── checkpoints/
    ├── sysu_uci.pth
    ├── regdb/
    │   ├── trial_01.pth
    │   ├── ...
    │   ├── trial_10.pth
    │   └── results.json
    ├── llcm_uci.pth
    ├── vcm_uci.pth
    └── bupt_uci.pth
```

## Train

```bash
python train.py --config configs/sysu.json  --data-root data --gpu 0 --output outputs/sysu
python train.py --config configs/llcm.json  --data-root data --gpu 0 --output outputs/llcm
python train.py --config configs/vcm.json   --data-root data --gpu 0 --output outputs/vcm
python train.py --config configs/bupt.json  --data-root data --gpu 0 --output outputs/bupt
```

For RegDB, run this entire block to train all ten official trials and then evaluate their final checkpoints. It stops if any command fails and saves per-trial scores, their mean, and sample standard deviation to `outputs/regdb/results.json`:

```bash
(
    set -e
    for trial in {1..10}; do
        python train.py --config configs/regdb.json --data-root data --gpu 0 --trial "$trial" --output "outputs/regdb/trial_$trial"
    done
    python evaluate.py --dataset regdb --data-root data --gpu 0 --all-trials --checkpoint 'outputs/regdb/trial_{trial}/global.pth' --output outputs/regdb/results.json
)
```

Use a new output directory for each run. Training saves the final model as `global.pth` and its evaluation results as `metrics.json` in that directory. Each RegDB run uses seed 0 and 50 rounds; `--trial` selects its training and test split.

## Evaluate

Evaluate the released checkpoints:

```bash
python evaluate.py --dataset sysu  --data-root data --gpu 0 --output results/sysu.json
python evaluate.py --dataset regdb --data-root data --gpu 0 --all-trials --output results/regdb.json
python evaluate.py --dataset llcm  --data-root data --gpu 0 --output results/llcm.json
python evaluate.py --dataset vcm   --data-root data --gpu 0 --output results/vcm.json
python evaluate.py --dataset bupt  --data-root data --gpu 0 --output results/bupt.json
```

RegDB `--all-trials` loads `checkpoints/regdb/trial_01.pth` through `trial_10.pth` and reports per-trial scores, their mean, and sample standard deviation. For a single split, replace `--all-trials` with `--trial 1`. Other datasets load `checkpoints/<dataset>_uci.pth`. To evaluate your own trained model, specify `--checkpoint`:

```bash
python evaluate.py --dataset sysu --data-root data --gpu 0 --checkpoint outputs/sysu/global.pth --output outputs/sysu/reevaluation.json
```

The RegDB training block above also evaluates all ten newly trained checkpoints. In its checkpoint template, `{trial}` is replaced with split numbers 1–10. For single-trial evaluation, use the same `--trial` as training.
