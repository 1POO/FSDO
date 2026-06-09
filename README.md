# FSDO: Fine-grained Spatiotemporal Dual-alignment with Object-aware Transformer for Joint Moment Retrieval and Highlight Detection




## Supported Datasets

| Dataset | Domains | Modalities |
|---------|---------|------------|
| [QVHighlights](https://github.com/jayleicn/moment_detr) | YouTube diverse | Video, Text, Subtitles, Audio |
| [TVSum](https://github.com/yalesong/tvsum) | 10 domains (BK, BT, DS, FM, GA, MS, PK, PR, VT, VU) | Video, Text, Subtitles |
| [Charades-STA](https://prior.allenai.org/projects/charades) | Egocentric daily activities | Video, Text, Subtitles |
| [TACoS](http://www.coli.uni-saarland.de/projects/smile/page.php?id=tacos) | Cooking | Video, Text |


## Prerequisites

### 0. Clone and setup

```
git clone <your-repo-url> fsdo
cd fsdo
```

### 1. Prepare datasets
Download the features files or extract the features independently using the method described in the paper.

### 2. Install dependencies
Python version 3.10 is required. Install dependencies using:
```
pip install -r requirements.txt
```

## QVHighlights

### Training

You can train the model using only video features or both video and audio features:

```
python fsdo/scripts/run_train_with_sub.py
```

The best validation accuracy is achieved at the last epoch.

### Inference Evaluation and Codalab Submission

After training, you can generate `hl_val_submission.jsonl` and `hl_test_submission.jsonl` for validation and test sets by running:

```
python fsdo/scripts/run_train_with_sub.py --ckpt_path /path/to/your/checkpoint.pth  --eval_split_name val
python fsdo/scripts/run_train_with_sub.py --ckpt_path /path/to/your/checkpoint.pth  --eval_split_name test 
```
For more details on submission, see [standalone_eval/README.md](standalone_eval/README.md).

----------

## TVSum

### Training

Similar to QVHighlights, you can train the model on the TVSum dataset:

```
python fsdo/scripts/tvsum/run_train_tvsum.py
```
## Charades-STA

### Training

Similar to QVHighlights, you can train the model on the Charades-STA dataset:

```
python fsdo/scripts/charades/run_train_charades.py
```
