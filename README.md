# Awesome Prompting Papers in Computer Vision

## Introduction 
A curated list of prompt/adapter-based papers in computer vision and vision-language learning.
![task](https://img.shields.io/badge/task-image--classification-green?style=flat-square)
![task](https://img.shields.io/badge/task-point--cloud--recognition-green?style=flat-square)
![task](https://img.shields.io/badge/task-pre--training-green?style=flat-square)
![task](https://img.shields.io/badge/task-semantic--segmentation-green?style=flat-square)
![task](https://img.shields.io/badge/task-instance--segmentation-green?style=flat-square)
![task](https://img.shields.io/badge/task-object--detection-green?style=flat-square)

![specific tag](https://img.shields.io/badge/NAS-blue?style=flat-square)
![](https://img.shields.io/badge/unsupervised-blue?style=flat-square)

## Prompt module design
This section contains papers define the general trends of the prompt module design in computer vision. We organize these papers as follows.

Paper name (2) (its tag) e.g. ![task](https://img.shields.io/badge/task-object--detection-green?style=flat-square):
- Its improvements 1
- Its improvements 2

### Papers

1. `CVPR'22` DenseCLIP: Language-Guided Dense Prediction with Context-Aware Prompting	[[paper]](https://arxiv.org/pdf/2112.01518.pdf)[[code]](https://github.com/raoyongming/denseclip)  ![](https://img.shields.io/badge/task-semantic--segmentation-green?style=flat-square)
![](https://img.shields.io/badge/task-instance--segmentation-green?style=flat-square)
![](https://img.shields.io/badge/task-object--detection-green?style=flat-square)
2. `Arxiv 21/09` **(CoOP)** Learning to Prompt for Vision-Language Models 	[[paper]](https://arxiv.org/abs/2109.01134) [[code]](https://github.com/KaiyangZhou/CoOp) ![](https://img.shields.io/badge/task-image--classification-green?style=flat-square)
    > - `CVPR'22` Prompt Distribution Learning [[paper]](https://arxiv.org/pdf/2205.03340.pdf) 
    > - `CVPR'22` **(CoCoOp)** Conditional Prompt Learning for Vision-Language Models [[paper]](https://arxiv.org/pdf/2203.05557.pdf) [[code]](https://github.com/KaiyangZhou/CoOp)
    > - `Arxiv 22/06` **(DualCoOp)** DualCoOp: Fast Adaptation to Multi-Label Recognition with Limited Annotations [[paper]](https://arxiv.org/abs/2206.09541) 
    > - `Arxiv 22/05` Prompt-aligned Gradient for Prompt Tuning [[paper]](https://arxiv.org/abs/2205.14865) [[code]](https://github.com/BeierZhu/Prompt-align) 

3. `Arxiv 21/10` CLIP-Adapter: Better Vision-Language Models with Feature Adapters [[paper]](https://arxiv.org/abs/2110.04544) [[code]](https://github.com/gaopengcuhk/clip-adapter) ![](https://img.shields.io/badge/task-image--classification-green?style=flat-square)

4. `Arxiv 22/03` **(VPT)** Visual Prompt Tuning [[paper]](https://arxiv.org/pdf/2203.12119.pdf)
    > - `Arxiv 22/03` Exploring Visual Prompts for Adapting Large-Scale Models [[paper]](https://arxiv.org/pdf/2203.17274.pdf) [[code]](https://github.com/hjbahng/visual_prompting) ![](https://img.shields.io/badge/task-image--classification-green?style=flat-square)

5. `Arxiv 22/05` **(AdaptFormer)** AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition
 [[paper]](https://arxiv.org/abs/2205.13535) [[code]](https://github.com/ShoufaChen/AdaptFormer) ![](https://img.shields.io/badge/task-semantic--segmentation-green?style=flat-square) ![](https://img.shields.io/badge/task-image--classification-green?style=flat-square) ![](https://img.shields.io/badge/task-semantic--segmentation-green?style=flat-square)

6. `Arxiv 22/04` **(UPL)** Unsupervised Prompt Learning for Vision-Language Models [[paper]](https://arxiv.org/pdf/2204.03649.pdf) [[code]](https://github.com/tonyhuang2022/UPL) ![task](https://img.shields.io/badge/task-action--recognition-green?style=flat-square) ![](https://img.shields.io/badge/tag-unsupervised-blue?style=flat-square)

7. `Arxiv 22/06` **(NOAH)** Neural Prompt Search [[paper]](https://arxiv.org/abs/2206.04673) [[code]](https://github.com/Davidzhangyuanhan/NOAH) ![](https://img.shields.io/badge/task-image--classification-green?style=flat-square) ![](https://img.shields.io/badge/tag-NAS-blue?style=flat-square)

8. `Arxiv 22/06` Parameter-Efficient Image-to-Video Transfer
Learning [[paper]](https://arxiv.org/pdf/2206.13559.pdf) ![task](https://img.shields.io/badge/task-action--recognition-green?style=flat-square)


## Prompt module application

- `CVPR'22` Grounded Language-Image Pre-training [[paper]](https://arxiv.org/pdf/2112.03857.pdf) [[code]](https://github.com/microsoft/GLIP)
- `CVPR'22` PointCLIP: Point Cloud Understanding by CLIP	[[paper]](https://arxiv.org/pdf/2112.02413.pdf) [[code]](https://github.com/ZrrSkywalker/PointCLIP)
- `CVPR'22` Align and Prompt: Video-and-Language Pre-training with Entity Prompts [[paper]](https://arxiv.org/abs/2112.09583) [[code]](https://github.com/salesforce/ALPRO)

- `CVPR'22` GroupViT: Semantic Segmentation Emerges from Text Supervision [[paper]](https://arxiv.org/pdf/2202.11094.pdf) [[code]](https://jerryxu.net/GroupViT/)
- `CVPR'22` Prompt-Based Multi-Modal Image Segmentation	[[paper]](https://arxiv.org/abs/2112.10003) [[code]](https://github.com/timojl/clipseg)
- `ACL'22` A Good Prompt Is Worth Millions of Parameters? Low-resource Prompt-based Learning for Vision-Language Models	[[paper]](https://arxiv.org/abs/2110.08484)

- `ICML'21` Unifying Vision-and-Language Tasks via Text Generation [[paper]](https://arxiv.org/abs/2102.02779) [[code]](https://github.com/j-min/VL-T5)
- `ICML'21` **(CLIP)** Learning Transferable Visual Models From Natural Language Supervision [[paper]](https://arxiv.org/abs/2103.00020) [[code]](https://github.com/OpenAI/CLIP)
- `NIPS'21` Multimodal Few-Shot Learning with Frozen Language Models	[[paper]](https://arxiv.org/abs/2106.13884)
- `ICCV'21` StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery	 [[paper]](https://arxiv.org/abs/2103.17249) [[code]](https://github.com/orpatashnik/StyleCLIP)
- `Arxiv 21/12` Learning to Prompt for Continual Learning	 [[paper]](https://arxiv.org/abs/2112.08654) [[code]](https://github.com/google-research/l2p)
- `Arxiv 21/12` Prompting Visual-Language Models for Efficient Video Understanding [[paper]](https://arxiv.org/abs/2112.04478) [[code]](https://github.com/ju-chen/Efficient-Prompt)
- `Arxiv 21/12` Unified Multimodal Pre-training and Prompt-based Tuning for Vision-Language Understanding and Generation [[paper]](https://arxiv.org/abs/2112.05587)
- `Arxiv 21/11` ClipCap: CLIP Prefix for Image Captioning	[[paper]](https://arxiv.org/abs/2111.09734) [[code]](https://github.com/rmokady/CLIP_prefix_caption)
- `Arxiv 21/11` Amortized Prompt: Lightweight Fine-Tuning for CLIP in Domain Generalization [[paper]](https://arxiv.org/abs/2111.12853)
- `Arxiv 21/11` Training-free clip-adapter for better vision-language modeling [[paper]](https://arxiv.org/pdf/2111.03930.pdf) [[code]](https://github.com/gaopengcuhk/tip-adapter)

- `Arxiv 21/09` ActionCLIP: A New Paradigm for Video Action Recognition [[paper]](https://arxiv.org/abs/2109.08472) [[code]](https://github.com/sallymmx/ActionCLIP)
- `Arxiv 21/08` **(CPT)** Colorful Prompt Tuning for Pre-trained Vision-Language Models [[paper]](https://arxiv.org/abs/2109.11797)
- `Arxiv 22/06` **(REPE)** Rethinking the Openness of CLIP [[paper]](https://arxiv.org/abs/2206.01986)

### Other Resources 
- [PromptPapers](https://github.com/thunlp/PromptPapers): A comprehensive curated list for prompting paper (mainly in natural language processing)
- `Arxiv 21/07` Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing	[[paper]](https://arxiv.org/abs/2107.13586)


