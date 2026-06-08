# FSDO: Fine-grained Spatiotemporal Dual-alignment with Object-aware Transformer for Joint Moment Retrieval and Highlight Detection

A multi-modal transformer framework for **video moment retrieval** and **highlight detection**, extending the [TR-DETR](https://ojs.aaai.org/index.php/AAAI/article/view/28304) (AAAI 2024) architecture with fine-grained cross-modal contrastive alignment, object-aware video encoding, and temporal gated fusion.

## Supported Tasks

| Task | Description |
|------|-------------|
| **Moment Retrieval (MR)** | Given a natural language query, localize the most relevant temporal segments in a video. |
| **Highlight Detection (HL)** | Score every clip in the video by its saliency/highlightness relative to the query. |

## Supported Datasets

| Dataset | Domains | Modalities |
|---------|---------|------------|
| [QVHighlights](https://github.com/jayleicn/moment_detr) | YouTube diverse | Video, Text, Subtitles, Audio |
| [TVSum](https://github.com/yalesong/tvsum) | 10 domains (BK, BT, DS, FM, GA, MS, PK, PR, VT, VU) | Video, Text, Subtitles |
| [Charades-STA](https://prior.allenai.org/projects/charades) | Egocentric daily activities | Video, Text, Subtitles |
| [TACoS](http://www.coli.uni-saarland.de/projects/smile/page.php?id=tacos) | Cooking | Video, Text |

## Architecture Overview

FSDO uses a multi-branch transformer with cross-modal fusion:

- **Video Encoder**: Processes multi-stream video features (SlowFast + CLIP + object features) with temporal encoding
- **Object Encoder**: Encodes object-level visual features from CLIP
- **Subtitle Encoder**: Encodes subtitle/ASR features aligned with video frames
- **Text Encoder**: Processes the natural language query
- **Temporal Gated Fusion** (`gate_vs`, `gate_vo`): Adaptive fusion of video-subtitle and video-object features
- **Cross-Modal Fusers** (`VSLFuser`): Bi-directional cross-attention between modalities
- **Transformer Decoder**: Produces span predictions and saliency scores via learnable queries

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| FSDO Model | [`fsdo/model.py`](fsdo/model.py:76) | Main model class with multi-branch encoding and decoding |
| Transformer | [`fsdo/transformer.py`](fsdo/transformer.py) | Encoder-decoder transformer with saliency refinement |
| VTC Loss | [`fsdo/loss_fun/VTCLoss.py`](fsdo/loss_fun/VTCLoss.py) | Global video-text contrastive alignment loss |
| CTC Loss | [`fsdo/loss_fun/new_CTC4.py`](fsdo/loss_fun/new_CTC4.py) | Fine-grained local cross-modal contrastive loss |
| Gate Fuser | [`fsdo/gate_fuser.py`](fsdo/gate_fuser.py) | Temporal gated fusion for multi-modal features |
| CQA Fuser | [`fsdo/interaction/test_CQA.py`](fsdo/interaction/test_CQA.py) | Cross-modal query-aware fusion |
| Position Encoding | [`fsdo/position_encoding.py`](fsdo/position_encoding.py) | Sinusoidal and learned positional embeddings |
| Rotary Embedding | [`fsdo/kernel/rotary.py`](fsdo/kernel/rotary.py) | Rotary position encoding for attention |

### Loss Functions

The model is trained with multiple losses:

- **Span Loss** (L1 + GIoU): Regression of predicted temporal spans
- **Label Loss** (Focal): Foreground/background classification for each query slot
- **Saliency Loss** (Ranking + Margin): Encourages higher scores for relevant clips
- **VTC Loss** (Video-Text Contrastive): Global cross-modal alignment between video and text embeddings
- **CTC Loss** (Cross-modal Temporal Contrastive): Fine-grained local alignment at the clip-token level, applied to 5 modality pairs (video-text, object-text, object-video, subtitle-object, subtitle-text)

---

## Prerequisites

### 0. Clone and setup

```bash
git clone <your-repo-url> fsdo
cd fsdo
