# Awesome Prompting Papers in Computer Vision

## Introduction 
A curated list of prompt-based papers in computer vision and vision-language learning.

**Keywords**:
* Task tag, e.g., ![](https://img.shields.io/badge/Image--Classification-759CBC?style=flat-square) ![](https://img.shields.io/badge/VQA-759CBC?style=flat-square)
* Abbreviation tag, e.g., ![](https://img.shields.io/badge/CLIP-CD6155?style=flat-square)
* Characteristic tag: Some characteristic makes this paper unique, e.g., ![](https://img.shields.io/badge/NAS-BC9575?style=flat-square) ![](https://img.shields.io/badge/unsupervised-BC9575?style=flat-square)
* **Bold font**: We highlight some pilot work that may contribute to the prevalence of visual prompting.


## Prompt Learning
This section contains papers designing prompt (containing adapter) modules for parameter-efficient adaptation of foundation models. 

### Vision Prompt
- **Visual Prompt Tuning** [[pdf]](https://arxiv.org/pdf/2203.12119.pdf) [[code]](https://github.com/KMnP/vpt)
   
  `ECCV 2022` ![](https://img.shields.io/badge/VPT-CD6155?style=flat-square) ![](https://img.shields.io/badge/Image--Classification-759CBC?style=flat-square) 

- **Exploring Visual Prompts for Adapting Large-Scale Models** [[pdf]](https://arxiv.org/pdf/2203.17274.pdf) [[code]](https://github.com/hjbahng/visual_prompting)

  `Arxiv 2022/03` ![](https://img.shields.io/badge/Image--Classification-759CBC?style=flat-square) 
  
- AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition [[pdf]](https://arxiv.org/abs/2205.13535) [[code]](https://github.com/ShoufaChen/AdaptFormer)
  
  `Arxiv 2022/05` ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square) ![](https://img.shields.io/badge/action--recognition-759CBC?style=flat-square)

- Neural Prompt Search [[pdf]](https://arxiv.org/abs/2206.04673) [[code]](https://github.com/Davidzhangyuanhan/NOAH)

  `Arxiv 2022/06` ![](https://img.shields.io/badge/NOAH-CD6155?style=flat-square) ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square) ![](https://img.shields.io/badge/NAS-BC9575?style=flat-square)

- Convolutional Bypasses Are Better Vision Transformer Adapters [[pdf]](https://arxiv.org/abs/2207.07039)

  `Arxiv 2022/06`  ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square) 


### Vision-Language Prompt

- **Learning Transferable Visual Models From Natural Language Supervision** [[pdf]](https://arxiv.org/abs/2103.00020) [[code]](https://github.com/OpenAI/CLIP) 

  `ICML 2021` ![](https://img.shields.io/badge/CLIP-CD6155?style=flat-square) ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square)

- **Learning to Prompt for Vision-Language Models** [[pdf]](https://arxiv.org/abs/2109.01134) [[code]](https://github.com/KaiyangZhou/CoOp)

  `IJCV 2022`  ![](https://img.shields.io/badge/CoOP-CD6155?style=flat-square) ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square)

- Prompt Distribution Learning [[pdf]](https://arxiv.org/pdf/2205.03340.pdf)

  `CVPR 2022`

- Conditional Prompt Learning for Vision-Language Models [[pdf]](https://arxiv.org/pdf/2203.05557.pdf) [[code]](https://github.com/KaiyangZhou/CoOp)

  `CVPR 2022`  ![](https://img.shields.io/badge/CoCoOP-CD6155?style=flat-square) ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square)

- DenseCLIP: Language-Guided Dense Prediction with Context-Aware Prompting	[[pdf]](https://arxiv.org/pdf/2112.01518.pdf) [[code]](https://github.com/raoyongming/denseclip) 

  `CVPR 2022` ![](https://img.shields.io/badge/detection-759CBC?style=flat-square) ![](https://img.shields.io/badge/segmentation-759CBC?style=flat-square) 

- Learning to Prompt for Continual Learning [[pdf]](https://arxiv.org/abs/2112.08654) [[code]](https://github.com/google-research/l2p)

  `CVPR 2022`  ![](https://img.shields.io/badge/continue--learning-BC9575?style=flat-square)

- Bridge-Prompt: Towards Ordinal Action Understanding in Instructional Videos [[pdf]](https://arxiv.org/pdf/2203.14104.pdf) [[code]](https://github.com/ttlmh/Bridge-Prompt) 

  `CVPR 2022` ![](https://img.shields.io/badge/action--recognition-759CBC?style=flat-square) ![](https://img.shields.io/badge/action--segmentation-759CBC?style=flat-square)

- PointCLIP: Point Cloud Understanding by CLIP [[pdf]](https://arxiv.org/pdf/2112.02413.pdf) [[code]](https://github.com/ZrrSkywalker/PointCLIP)

  `CVPR 2022`  ![](https://img.shields.io/badge/point--cloud-759CBC?style=flat-square)

- VL-Adapter: Parameter-Efficient Transfer Learning for Vision-and-Language Tasks [[pdf]](https://arxiv.org/pdf/2112.06825.pdf) [[code]](https://github.com/ylsung/VL_adapter)
  
    `CVPR 2022`  ![](https://img.shields.io/badge/VQA-759CBC?style=flat-square) ![](https://img.shields.io/badge/VideoQA-759CBC?style=flat-square) ![](https://img.shields.io/badge/captioning-759CBC?style=flat-square)
    
- Can Language Understand Depth? [[pdf]](https://arxiv.org/pdf/2207.01077.pdf) [[code]](https://github.com/Adonis-galaxy/DepthCLIP)

  `ACM MM 2022` ![](https://img.shields.io/badge/depthclip-CD6155?style=flat-square)  ![](https://img.shields.io/badge/depth--estimation-759CBC?)
  
<bar>

- **Colorful Prompt Tuning for Pre-trained Vision-Language Models** [[pdf]](https://arxiv.org/abs/2109.11797) 

  `Arxiv 2021/08` ![](https://img.shields.io/badge/CPT-CD6155?style=flat-square) ![](https://img.shields.io/badge/grounding-759CBC?style=flat-square) 


- ActionCLIP: A New Paradigm for Video Action Recognition [[pdf]](https://arxiv.org/abs/2109.08472) [[code]](https://github.com/sallymmx/ActionCLIP)

  `Arxiv 2021/09` ![](https://img.shields.io/badge/action--recognition-759CBC?style=flat-square)

- CLIP-Adapter: Better Vision-Language Models with Feature Adapters [[pdf]](https://arxiv.org/abs/2110.04544) [[code]](https://github.com/gaopengcuhk/clip-adapter)

  `Arxiv 2021/10` ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square)

- Tip-Adapter: Training-free CLIP-Adapter for Better Vision-Language Modeling [[pdf]](https://arxiv.org/pdf/2111.03930.pdf) [[code]](https://github.com/gaopengcuhk/tip-adapter)

  `Arxiv 2021/11`  ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square)

- Amortized Prompt: Lightweight Fine-Tuning for CLIP in Domain Generalization [[pdf]](https://arxiv.org/abs/2111.12853)

  `Arxiv 2021/11` ![](https://img.shields.io/badge/domain--generalization-BC9575?style=flat-square)

- Prompting Visual-Language Models for Efficient Video Understanding [[pdf]](https://arxiv.org/abs/2112.04478) [[code]](https://github.com/ju-chen/Efficient-Prompt)

  `Arxiv 2021/12` ![task](https://img.shields.io/badge/action--recognition-759CBC?style=flat-square) ![task](https://img.shields.io/badge/action--localization-759CBC?style=flat-square) ![task](https://img.shields.io/badge/retrieval-759CBC?style=flat-square)

- Unsupervised Prompt Learning for Vision-Language Models [[pdf]](https://arxiv.org/pdf/2204.03649.pdf) [[code]](https://github.com/tonyhuang2022/UPL)

  `Arxiv 2022/04` ![](https://img.shields.io/badge/UPL-CD6155?style=flat-square) ![task](https://img.shields.io/badge/image--classification-759CBC?style=flat-square) ![](https://img.shields.io/badge/unsupervised-BC9575?style=flat-square)

- Prompt-aligned Gradient for Prompt Tuning [[pdf]](https://arxiv.org/abs/2205.14865) [[code]](https://github.com/BeierZhu/Prompt-align)

  `Arxiv 2022/05` ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square)


- Parameter-Efficient Image-to-Video Transfer Learning [[pdf]](https://arxiv.org/pdf/2206.13559.pdf)

  `Arxiv 2022/06`  ![](https://img.shields.io/badge/ST--adapter-CD6155?style=flat-square) ![task](https://img.shields.io/badge/action--recognition-759CBC?style=flat-square)

- DualCoOp: Fast Adaptation to Multi-Label Recognition with Limited Annotations [[pdf]](https://arxiv.org/abs/2206.09541)

  `Arxiv 2022/06` ![task](https://img.shields.io/badge/multilabel--recognition-759CBC?style=flat-square)

- Rethinking the Openness of CLIP [[pdf]](https://arxiv.org/abs/2206.01986)

  `Arxiv 2022/06` ![](https://img.shields.io/badge/REPE-CD6155?style=flat-square)

- OrdinalCLIP: Learning Rank Prompts for Language-Guided Ordinal Regression [[pdf]](https://arxiv.org/abs/2206.02338)

  `Arxiv 2022/06`


### Language-Interactable Prompt
Language-interactable prompter develops few/**zero-shot** capabilities by prompting one/several independent foundational models (VLMs, LMs, VMs, etc.) with the language interface. 

- **Multimodal Few-Shot Learning with Frozen Language Models** [[pdf]](https://arxiv.org/abs/2106.13884)

  `NIPS 2021` ![](https://img.shields.io/badge/VQA-759CBC?)
  
- An Empirical Study of GPT-3 for Few-Shot Knowledge-Based VQA [[pdf]](https://arxiv.org/pdf/2109.05014.pdf) [[code]](https://github.com/microsoft/PICa) 

  `AAAI 2022` ![](https://img.shields.io/badge/VQA-759CBC?)

- A Good Prompt Is Worth Millions of Parameters? Low-resource Prompt-based Learning for Vision-Language Models	[[pdf]](https://arxiv.org/abs/2110.08484)

  `ACL 2022`  ![](https://img.shields.io/badge/VQA-759CBC?)  ![](https://img.shields.io/badge/captioning-759CBC?)

- VisualGPT: Data-efficient Adaptation of Pretrained Language Models for Image Captioning [[pdf]](https://arxiv.org/pdf/2102.10407.pdf) [[code]](https://github.com/Vision-CAIR/VisualGPT)

  `CVPR 2022` ![](https://img.shields.io/badge/captioning-759CBC?)

<bar>
  
- ClipCap: CLIP Prefix for Image Captioning	[[pdf]](https://arxiv.org/abs/2111.09734) [[code]](https://github.com/rmokady/CLIP_prefix_caption)

  `Arxiv 2021/11` ![](https://img.shields.io/badge/captioning-759CBC?)

- Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language [[pdf]](https://arxiv.org/pdf/2204.00598.pdf) [[code]](https://socraticmodels.github.io/#code)

  `Arxiv 2022/04` ![](https://img.shields.io/badge/captioning-759CBC?style=flat-square) ![](https://img.shields.io/badge/retrieval-759CBC?style=flat-square) ![](https://img.shields.io/badge/visual--dialog-759CBC?style=flat-square) 

- Flamingo: a Visual Language Model for Few-Shot Learning [[pdf]](https://arxiv.org/abs/2204.14198) 

  `Arxiv 2022/04` ![](https://img.shields.io/badge/image--classification-759CBC?) ![](https://img.shields.io/badge/VQA-759CBC?) ![](https://img.shields.io/badge/captioning-759CBC?)

- Language Models Can See: Plugging Visual Controls in Text Generation [[pdf]](https://arxiv.org/pdf/2205.02655.pdf) [[code]](https://github.com/yxuansu/MAGIC)

  `Arxiv 2022/05` ![](https://img.shields.io/badge/MAGIC-CD6155?style=flat-square) ![](https://img.shields.io/badge/captioning-759CBC?)

- Zero-Shot Video Question Answering via Frozen Bidirectional Language Models [[pdf]](https://arxiv.org/pdf/2206.08155.pdf) 

  `Arxiv 2022/06` ![](https://img.shields.io/badge/VideoQA-759CBC?)


## Application of Prompt
This section contains awesome papers using the prompt module as tools, like papers using prompts for pretraining or specific applications.

- Unifying Vision-and-Language Tasks via Text Generation [[pdf]](https://arxiv.org/abs/2102.02779) [[code]](https://github.com/j-min/T5)

  `ICML 2021`

- StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery	 [[pdf]](https://arxiv.org/abs/2103.17249) [[code]](https://github.com/orpatashnik/StyleCLIP)

  `ICCV 2021`

- Grounded Language-Image Pre-training [[pdf]](https://arxiv.org/pdf/2112.03857.pdf) [[code]](https://github.com/microsoft/GLIP)

  `CVPR 2022`

- Align and Prompt: Video-and-Language Pre-training with Entity Prompts [[pdf]](https://arxiv.org/abs/2112.09583) [[code]](https://github.com/salesforce/ALPRO)

  `CVPR 2022`

- GroupViT: Semantic Segmentation Emerges from Text Supervision [[pdf]](https://arxiv.org/pdf/2202.11094.pdf) [[code]](https://jerryxu.net/GroupViT/)

  `CVPR 2022`

<bar>

- Unified Multimodal Pretraining and Prompt-based Tuning for Vision-Language Understanding and Generation [[pdf]](https://arxiv.org/abs/2112.05587)
  `Arxiv 2021/12`

## Other Resources 
* [PromptPapers](https://github.com/thunlp/PromptPapers): A comprehensive curated list for prompting papers (mainly in natural language processing)
* Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing	[[pdf]](https://arxiv.org/abs/2107.13586)
  `Arxiv 2021/07`


