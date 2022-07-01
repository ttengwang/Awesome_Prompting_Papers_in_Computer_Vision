# Awesome Prompting Papers in Computer Vision

## Introduction 
A curated list of prompt/adapter-based papers in computer vision and vision-language learning.

### Keyword
Task tag: This paper focus on which task. ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square) ![](https://img.shields.io/badge/semantic--segmentation-759CBC?style=flat-square)


Characteristic tag: Some characteristic makes this paper unique. ![](https://img.shields.io/badge/NAS-BC9575?style=flat-square)
![](https://img.shields.io/badge/unsupervised-BC9575?style=flat-square)

## Prompt module design
This section contains papers designing prompt modules for parameter-efficient tuning of foundation models. We organize these papers as follows.

### e.g
Paper information + its tag
> - Further works: works that motivated by this paper.


### Papers
1. `ICML'21` **(CLIP)** Learning Transferable Visual Models From Natural Language Supervision [[paper]](https://arxiv.org/abs/2103.00020) [[code]](https://github.com/OpenAI/CLIP) ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square)
2. `Arxiv 21/09` **(CoOP)** Learning to Prompt for Vision-Language Models 	[[paper]](https://arxiv.org/abs/2109.01134) [[code]](https://github.com/KaiyangZhou/CoOp) ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square)
    > - `CVPR'22` Prompt Distribution Learning [[paper]](https://arxiv.org/pdf/2205.03340.pdf) 
    > - `CVPR'22` **(CoCoOp)** Conditional Prompt Learning for Vision-Language Models [[paper]](https://arxiv.org/pdf/2203.05557.pdf) [[code]](https://github.com/KaiyangZhou/CoOp)
    > - `Arxiv 22/05` Prompt-aligned Gradient for Prompt Tuning [[paper]](https://arxiv.org/abs/2205.14865) [[code]](https://github.com/BeierZhu/Prompt-align) 

3.  `Arxiv 21/08` **(CPT)** Colorful Prompt Tuning for Pre-trained Vision-Language Models [[paper]](https://arxiv.org/abs/2109.11797) ![](https://img.shields.io/badge/visual--grounding-759CBC?style=flat-square)

3. `Arxiv 21/10` CLIP-Adapter: Better Vision-Language Models with Feature Adapters [[paper]](https://arxiv.org/abs/2110.04544) [[code]](https://github.com/gaopengcuhk/clip-adapter) ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square)
    > - `Arxiv 21/11` **(Tip-Adapter)** Tip-Adapter: Training-free CLIP-Adapter for Better Vision-Language Modeling [[paper]](https://arxiv.org/pdf/2111.03930.pdf) [[code]](https://github.com/gaopengcuhk/tip-adapter)

4. `CVPR'22` DenseCLIP: Language-Guided Dense Prediction with Context-Aware Prompting	[[paper]](https://arxiv.org/pdf/2112.01518.pdf)[[code]](https://github.com/raoyongming/denseclip)  ![](https://img.shields.io/badge/semantic--segmentation-759CBC?style=flat-square)
![](https://img.shields.io/badge/instance--segmentation-759CBC?style=flat-square)
![](https://img.shields.io/badge/object--detection-759CBC?style=flat-square)

5. `CVPR'22` Learning to Prompt for Continual Learning	 [[paper]](https://arxiv.org/abs/2112.08654) [[code]](https://github.com/google-research/l2p) ![](https://img.shields.io/badge/continue--learning-759CBC?style=flat-square)

6. `CVPR'22` **(Br-Prompt)** Bridge-Prompt: Towards Ordinal Action Understanding in Instructional Videos [[paper]](https://arxiv.org/pdf/2203.14104.pdf) [[code]](https://github.com/ttlmh/Bridge-Prompt) ![](https://img.shields.io/badge/action--recognition-759CBC?style=flat-square) ![](https://img.shields.io/badge/action--segmentation-759CBC?style=flat-square)

6. `Arxiv 22/03` **(VPT)** Visual Prompt Tuning [[paper]](https://arxiv.org/pdf/2203.12119.pdf) ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square) 

7. `Arxiv 22/03` Exploring Visual Prompts for Adapting Large-Scale Models [[paper]](https://arxiv.org/pdf/2203.17274.pdf) [[code]](https://github.com/hjbahng/visual_prompting) ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square) 

8. `Arxiv 22/04` **(UPL)** Unsupervised Prompt Learning for Vision-Language Models [[paper]](https://arxiv.org/pdf/2204.03649.pdf) [[code]](https://github.com/tonyhuang2022/UPL) ![task](https://img.shields.io/badge/image--classification-759CBC?style=flat-square) ![](https://img.shields.io/badge/unsupervised-BC9575?style=flat-square)

9. `Arxiv 22/05` **(AdaptFormer)** AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition
 [[paper]](https://arxiv.org/abs/2205.13535) [[code]](https://github.com/ShoufaChen/AdaptFormer) ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square) ![](https://img.shields.io/badge/action--recognition-759CBC?style=flat-square)

10. `Arxiv 22/06` **(NOAH)** Neural Prompt Search [[paper]](https://arxiv.org/abs/2206.04673) [[code]](https://github.com/Davidzhangyuanhan/NOAH) ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square) ![](https://img.shields.io/badge/NAS-BC9575?style=flat-square)

11. `Arxiv 22/06` Parameter-Efficient Image-to-Video Transfer
Learning [[paper]](https://arxiv.org/pdf/2206.13559.pdf) ![task](https://img.shields.io/badge/action--recognition-759CBC?style=flat-square)

12.  `Arxiv 22/06` **(DualCoOp)** DualCoOp: Fast Adaptation to Multi-Label Recognition with Limited Annotations [[paper]](https://arxiv.org/abs/2206.09541) ![task](https://img.shields.io/badge/multi--label--recognition-759CBC?style=flat-square) 


## Prompt module application
This section contains papers using prompt module as tool for specific application (how to design a prompt model is not a contribution in such paper). For example, PointCLIP uses text prompt module designed by CLIP to retrive image.

- `CVPR'22` Grounded Language-Image Pre-training [[paper]](https://arxiv.org/pdf/2112.03857.pdf) [[code]](https://github.com/microsoft/GLIP)
- `CVPR'22` PointCLIP: Point Cloud Understanding by CLIP	[[paper]](https://arxiv.org/pdf/2112.02413.pdf) [[code]](https://github.com/ZrrSkywalker/PointCLIP)
- `CVPR'22` Align and Prompt: Video-and-Language Pre-training with Entity Prompts [[paper]](https://arxiv.org/abs/2112.09583) [[code]](https://github.com/salesforce/ALPRO) ![task](https://img.shields.io/badge/VL--pre--training-759CBC?style=flat-square)

- `CVPR'22` GroupViT: Semantic Segmentation Emerges from Text Supervision [[paper]](https://arxiv.org/pdf/2202.11094.pdf) [[code]](https://jerryxu.net/GroupViT/)
- `CVPR'22` Prompt-Based Multi-Modal Image Segmentation	[[paper]](https://arxiv.org/abs/2112.10003) [[code]](https://github.com/timojl/clipseg)
- `ACL'22` A Good Prompt Is Worth Millions of Parameters? Low-resource Prompt-based Learning for Vision-Language Models	[[paper]](https://arxiv.org/abs/2110.08484)

- `ICML'21` Unifying Vision-and-Language Tasks via Text Generation [[paper]](https://arxiv.org/abs/2102.02779) [[code]](https://github.com/j-min/VL-T5)
- `NIPS'21` Multimodal Few-Shot Learning with Frozen Language Models	[[paper]](https://arxiv.org/abs/2106.13884)
- `ICCV'21` StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery	 [[paper]](https://arxiv.org/abs/2103.17249) [[code]](https://github.com/orpatashnik/StyleCLIP)
- `Arxiv 21/12` Prompting Visual-Language Models for Efficient Video Understanding [[paper]](https://arxiv.org/abs/2112.04478) [[code]](https://github.com/ju-chen/Efficient-Prompt)
- `Arxiv 21/12` Unified Multimodal Pre-training and Prompt-based Tuning for Vision-Language Understanding and Generation [[paper]](https://arxiv.org/abs/2112.05587)
- `Arxiv 21/11` ClipCap: CLIP Prefix for Image Captioning	[[paper]](https://arxiv.org/abs/2111.09734) [[code]](https://github.com/rmokady/CLIP_prefix_caption)
- `Arxiv 21/11` Amortized Prompt: Lightweight Fine-Tuning for CLIP in Domain Generalization [[paper]](https://arxiv.org/abs/2111.12853)
- `Arxiv 21/09` ActionCLIP: A New Paradigm for Video Action Recognition [[paper]](https://arxiv.org/abs/2109.08472) [[code]](https://github.com/sallymmx/ActionCLIP)
- `Arxiv 22/06` **(REPE)** Rethinking the Openness of CLIP [[paper]](https://arxiv.org/abs/2206.01986)

### Other Resources 
- [PromptPapers](https://github.com/thunlp/PromptPapers): A comprehensive curated list for prompting paper (mainly in natural language processing)
- `Arxiv 21/07` Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing	[[paper]](https://arxiv.org/abs/2107.13586)


