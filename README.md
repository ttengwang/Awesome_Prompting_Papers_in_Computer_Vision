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
- **Learning to Prompt for Continual Learning** [[pdf]](https://arxiv.org/abs/2112.08654) [[code]](https://github.com/google-research/l2p)

  `CVPR 2022`  ![](https://img.shields.io/badge/continual--learning-759CBC?style=flat-square)

- **Visual Prompt Tuning** [[pdf]](https://arxiv.org/pdf/2203.12119.pdf) [[code]](https://github.com/KMnP/vpt)
   
  `ECCV 2022` ![](https://img.shields.io/badge/VPT-CD6155?style=flat-square) ![](https://img.shields.io/badge/Image--Classification-759CBC?style=flat-square) 

- **Exploring Visual Prompts for Adapting Large-Scale Models** [[pdf]](https://arxiv.org/pdf/2203.17274.pdf) [[code]](https://github.com/hjbahng/visual_prompting)

  `arXiv 2022/03` ![](https://img.shields.io/badge/Image--Classification-759CBC?style=flat-square) 

- DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning [[pdf]](https://arxiv.org/pdf/2204.04799.pdf) [[code]](https://github.com/google-research/l2p)

  `ECCV 2022` ![](https://img.shields.io/badge/continual--learning-759CBC?style=flat-square)
  
- AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition [[pdf]](https://arxiv.org/abs/2205.13535) [[code]](https://github.com/ShoufaChen/AdaptFormer)
  
  `NeurIPS 2022` ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square) ![](https://img.shields.io/badge/action--recognition-759CBC?style=flat-square)
  
- Vision Transformer Adapter for Dense Predictions [[pdf]](https://arxiv.org/pdf/2205.08534.pdf) [[code]](https://github.com/czczup/ViT-Adapter)
  
  `arXiv 2022/05` ![](https://img.shields.io/badge/object--detection-759CBC?style=flat-square) ![](https://img.shields.io/badge/instance--segmentaion-759CBC?style=flat-square)

- Neural Prompt Search [[pdf]](https://arxiv.org/abs/2206.04673) [[code]](https://github.com/Davidzhangyuanhan/NOAH)

  `arXiv 2022/06` ![](https://img.shields.io/badge/NOAH-CD6155?style=flat-square) ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square) ![](https://img.shields.io/badge/NAS-BC9575?style=flat-square)

- Convolutional Bypasses Are Better Vision Transformer Adapters [[pdf]](https://arxiv.org/abs/2207.07039) [[code]](https://github.com/JieShibo/PETL-ViT)

  `arXiv 2022/07`  ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square) 

- Conv-Adapter: Exploring Parameter Efficient Transfer Learning for ConvNets [[pdf]](https://arxiv.org/pdf/2208.07463.pdf) 

  `arXiv 2022/08`  ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square) 

- Prompt Vision Transformer for Domain Generalization [[pdf]](https://arxiv.org/pdf/2208.08914.pdf) 

  `arXiv 2022/08`  ![](https://img.shields.io/badge/domain--generalization-759CBC?style=flat-square) 

- Visual Prompting via Image Inpainting [[pdf]](https://arxiv.org/pdf/2209.00647.pdf) [[code]](https://yossigandelsman.github.io/visual_prompt/)

  `NeurIPS 2022`  ![](https://img.shields.io/badge/image--to--image--tasks-759CBC?style=flat-square)  ![](https://img.shields.io/badge/in--context--learning-BC9575?style=flat-square)
  
- Visual Prompt Tuning for Test-time Domain Adaptation [[pdf]](https://arxiv.org/abs/2210.04831)

  `arXiv 2022/10`  ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square) 
  
- Visual Prompting for Adversarial Robustness [[pdf]](https://arxiv.org/abs/2210.06284)

  `arXiv 2022/10`  ![](https://img.shields.io/badge/adversarial--robustness-759CBC?style=flat-square) 

- Prompt Generation Networks for Efficient Adaptation of Frozen Vision Transformers [[pdf]](https://arxiv.org/abs/2210.06466) [[code]](https://github.com/jochemloedeman/PGN)

  `arXiv 2022/10`  ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square) 

- Scaling & Shifting Your Features: A New Baseline for Efficient Model Tuning [[pdf]](https://arxiv.org/abs/2210.08823) [[code]](https://github.com/dongzelian/SSF)

  `NeurIPS 2022` ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square)
  
- Towards a Unified View on Visual Parameter-Efficient Transfer Learning [[pdf]](https://arxiv.org/abs/2210.00788) [[code]](https://github.com/bruceyo/V-PETL)
  
  `arXiv 2022/10` ![](https://img.shields.io/badge/action--recognition-759CBC?style=flat-square)
  
- Multitask Vision-Language Prompt Tuning [[pdf]](https://arxiv.org/abs/2211.11720) [[code]](https://github.com/sIncerass/MVLPT)

  `arXiv 2022/11` ![](https://img.shields.io/badge/MVLPT-CD6155?style=flat-square) ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square)  ![](https://img.shields.io/badge/multitask--learning-BC9575?style=flat-square)


### Vision-Language Prompt

- **Learning Transferable Visual Models From Natural Language Supervision** [[pdf]](https://arxiv.org/abs/2103.00020) [[code]](https://github.com/OpenAI/CLIP) 

  `ICML 2021` ![](https://img.shields.io/badge/CLIP-CD6155?style=flat-square) ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square)

- **Learning to Prompt for Vision-Language Models** [[pdf]](https://arxiv.org/abs/2109.01134) [[code]](https://github.com/KaiyangZhou/CoOp)

  `IJCV 2022`  ![](https://img.shields.io/badge/CoOP-CD6155?style=flat-square) ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square)

- Prompt Distribution Learning [[pdf]](https://arxiv.org/pdf/2205.03340.pdf)

  `CVPR 2022` ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square)

- Conditional Prompt Learning for Vision-Language Models [[pdf]](https://arxiv.org/pdf/2203.05557.pdf) [[code]](https://github.com/KaiyangZhou/CoOp)

  `CVPR 2022`  ![](https://img.shields.io/badge/CoCoOP-CD6155?style=flat-square) ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square)

- DenseCLIP: Language-Guided Dense Prediction with Context-Aware Prompting	[[pdf]](https://arxiv.org/pdf/2112.01518.pdf) [[code]](https://github.com/raoyongming/denseclip) 

  `CVPR 2022` ![](https://img.shields.io/badge/detection-759CBC?style=flat-square) ![](https://img.shields.io/badge/segmentation-759CBC?style=flat-square) 

- Bridge-Prompt: Towards Ordinal Action Understanding in Instructional Videos [[pdf]](https://arxiv.org/pdf/2203.14104.pdf) [[code]](https://github.com/ttlmh/Bridge-Prompt) 

  `CVPR 2022` ![](https://img.shields.io/badge/action--recognition-759CBC?style=flat-square) ![](https://img.shields.io/badge/action--segmentation-759CBC?style=flat-square)

- PointCLIP: Point Cloud Understanding by CLIP [[pdf]](https://arxiv.org/pdf/2112.02413.pdf) [[code]](https://github.com/ZrrSkywalker/PointCLIP)

  `CVPR 2022`  ![](https://img.shields.io/badge/point--cloud-759CBC?style=flat-square)

- VL-Adapter: Parameter-Efficient Transfer Learning for Vision-and-Language Tasks [[pdf]](https://arxiv.org/pdf/2112.06825.pdf) [[code]](https://github.com/ylsung/VL_adapter)
  
    `CVPR 2022`  ![](https://img.shields.io/badge/VQA-759CBC?style=flat-square) ![](https://img.shields.io/badge/VideoQA-759CBC?style=flat-square) ![](https://img.shields.io/badge/captioning-759CBC?style=flat-square)
    
- Can Language Understand Depth? [[pdf]](https://arxiv.org/pdf/2207.01077.pdf) [[code]](https://github.com/Adonis-galaxy/DepthCLIP)

  `ACM MM 2022` ![](https://img.shields.io/badge/depthclip-CD6155?style=flat-square)  ![](https://img.shields.io/badge/depth--estimation-759CBC?)

- Prompting for Multi-Modal Tracking [[pdf]](https://arxiv.org/abs/2207.14571)

  `ACM MM 2022` ![](https://img.shields.io/badge/object--tracking-CD6155?style=flat-square)

- Expanding Language-Image Pretrained Models for General Video Recognition [[pdf]](https://arxiv.org/abs/2208.02816) [[code]](https://aka.ms/X-CLIP)

  `ECCV 2022` ![](https://img.shields.io/badge/action--recognition-759CBC?style=flat-square)
  
- Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification [[pdf]](https://arxiv.org/pdf/2207.09519.pdf) [[code]](https://github.com/gaopengcuhk/tip-adapter)

  `ECCV 2022` ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square)
  
<bar>

- **Colorful Prompt Tuning for Pre-trained Vision-Language Models** [[pdf]](https://arxiv.org/abs/2109.11797) 

  `arXiv 2021/08` ![](https://img.shields.io/badge/CPT-CD6155?style=flat-square) ![](https://img.shields.io/badge/grounding-759CBC?style=flat-square) 


- ActionCLIP: A New Paradigm for Video Action Recognition [[pdf]](https://arxiv.org/abs/2109.08472) [[code]](https://github.com/sallymmx/ActionCLIP)

  `arXiv 2021/09` ![](https://img.shields.io/badge/action--recognition-759CBC?style=flat-square)

- CLIP-Adapter: Better Vision-Language Models with Feature Adapters [[pdf]](https://arxiv.org/abs/2110.04544) [[code]](https://github.com/gaopengcuhk/clip-adapter)

  `arXiv 2021/10` ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square)

- Amortized Prompt: Lightweight Fine-Tuning for CLIP in Domain Generalization [[pdf]](https://arxiv.org/abs/2111.12853)

  `arXiv 2021/11` ![](https://img.shields.io/badge/domain--generalization-BC9575?style=flat-square)

- Prompting Visual-Language Models for Efficient Video Understanding [[pdf]](https://arxiv.org/abs/2112.04478) [[code]](https://github.com/ju-chen/Efficient-Prompt)

  `arXiv 2021/12` ![task](https://img.shields.io/badge/action--recognition-759CBC?style=flat-square) ![task](https://img.shields.io/badge/action--localization-759CBC?style=flat-square) ![task](https://img.shields.io/badge/retrieval-759CBC?style=flat-square)

- Unsupervised Prompt Learning for Vision-Language Models [[pdf]](https://arxiv.org/pdf/2204.03649.pdf) [[code]](https://github.com/tonyhuang2022/UPL)

  `arXiv 2022/04` ![](https://img.shields.io/badge/UPL-CD6155?style=flat-square) ![task](https://img.shields.io/badge/image--classification-759CBC?style=flat-square) ![](https://img.shields.io/badge/unsupervised-BC9575?style=flat-square)

- Prompt-aligned Gradient for Prompt Tuning [[pdf]](https://arxiv.org/abs/2205.14865) [[code]](https://github.com/BeierZhu/Prompt-align)

  `arXiv 2022/05` ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square)


- Parameter-Efficient Image-to-Video Transfer Learning [[pdf]](https://arxiv.org/pdf/2206.13559.pdf)

  `arXiv 2022/06`  ![](https://img.shields.io/badge/ST--adapter-CD6155?style=flat-square) ![task](https://img.shields.io/badge/action--recognition-759CBC?style=flat-square)

- DualCoOp: Fast Adaptation to Multi-Label Recognition with Limited Annotations [[pdf]](https://arxiv.org/abs/2206.09541)

  `arXiv 2022/06` ![task](https://img.shields.io/badge/multilabel--recognition-759CBC?style=flat-square)

- Rethinking the Openness of CLIP [[pdf]](https://arxiv.org/abs/2206.01986)

  `arXiv 2022/06` ![](https://img.shields.io/badge/REPE-CD6155?style=flat-square)

- OrdinalCLIP: Learning Rank Prompts for Language-Guided Ordinal Regression [[pdf]](https://arxiv.org/abs/2206.02338)

  `NeurIPS 2022` ![](https://img.shields.io/badge/ordinal--regression-759CBC?style=flat-square)

- Prompt Tuning for Generative Multimodal Pretrained Models [[pdf]](https://arxiv.org/abs/2208.02532) [[code]](https://github.com/OFA-Sys/OFA)

  `arXiv 2022/06` ![](https://img.shields.io/badge/VQA-759CBC?style=flat-square) ![](https://img.shields.io/badge/captioning-759CBC?style=flat-square)  ![](https://img.shields.io/badge/referring--expression-759CBC?style=flat-square)  ![](https://img.shields.io/badge/visual--entailment-759CBC?style=flat-square) 
  
- Prompt Tuning with Soft Context Sharing for Vision-Language Models [[pdf]](https://arxiv.org/pdf/2208.13474.pdf)

  `arXiv 2022/08` ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square)  ![](https://img.shields.io/badge/multi--task--learning-759CBC?style=flat-square) 
 
- Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models [[pdf]](https://arxiv.org/pdf/2209.07511.pdf) [[code]](https://azshue.github.io/TPT/)

  `NeurIPS 2022` ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square) ![](https://img.shields.io/badge/reasoning-759CBC?style=flat-square) 
  
- CPL: Counterfactual Prompt Learning for Vision and Language Models [[pdf]](https://arxiv.org/abs/2210.10362) [[code]](https://github.com/eric-ai-lab/CPL)

  `arXiv 2022/10`  ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square) ![](https://img.shields.io/badge/retrieval-759CBC?style=flat-square) 
![](https://img.shields.io/badge/VQA-759CBC?style=flat-square) 


- Understanding and Mitigating Overfitting in Prompt Tuning for Vision-Language Models [[pdf]](https://arxiv.org/abs/2211.02219) [[code]](https://github.com/machengcheng2016/Subspace-Prompt-Learning)

  `arXiv 2022/10` ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square) ![](https://img.shields.io/badge/object--detection-759CBC?style=flat-square) 
![](https://img.shields.io/badge/semantic--segmentation-759CBC?style=flat-square) 


- Unified Vision and Language Prompt Learning [[pdf]](https://arxiv.org/abs/2210.07225)

  `arXiv 2022/10` ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square)
  
- MaPLe: Multi-modal Prompt Learning [[pdf]](https://arxiv.org/abs/2210.03117) [[code]](https://github.com/muzairkhattak/multimodal-prompt-learning)

  `arXiv 2022/10` ![](https://img.shields.io/badge/image--classification-759CBC?style=flat-square)
  

- Multi-Prompt Alignment for Multi-source Unsupervised Domain Adaptation [[pdf]](https://arxiv.org/abs/2209.15210)

  `arXiv 2022/10` ![](https://img.shields.io/badge/domain--adaptation-759CBC?style=flat-square)



### Language-Interactable Prompt
Language-interactable prompter develops few/**zero-shot** capabilities by prompting one/several independent foundational models (VLMs, LMs, VMs, etc.) with the language interface. 

- **Multimodal Few-Shot Learning with Frozen Language Models** [[pdf]](https://arxiv.org/abs/2106.13884)

  `NeurIPS 2021` ![](https://img.shields.io/badge/VQA-759CBC?)
  
- An Empirical Study of GPT-3 for Few-Shot Knowledge-Based VQA [[pdf]](https://arxiv.org/pdf/2109.05014.pdf) [[code]](https://github.com/microsoft/PICa) 

  `AAAI 2022` ![](https://img.shields.io/badge/VQA-759CBC?)

- A Good Prompt Is Worth Millions of Parameters? Low-resource Prompt-based Learning for Vision-Language Models	[[pdf]](https://arxiv.org/abs/2110.08484)

  `ACL 2022`  ![](https://img.shields.io/badge/VQA-759CBC?)  ![](https://img.shields.io/badge/captioning-759CBC?)

- VisualGPT: Data-efficient Adaptation of Pretrained Language Models for Image Captioning [[pdf]](https://arxiv.org/pdf/2102.10407.pdf) [[code]](https://github.com/Vision-CAIR/VisualGPT)

  `CVPR 2022` ![](https://img.shields.io/badge/captioning-759CBC?)

<bar>
  
- ClipCap: CLIP Prefix for Image Captioning	[[pdf]](https://arxiv.org/abs/2111.09734) [[code]](https://github.com/rmokady/CLIP_prefix_caption)

  `arXiv 2021/11` ![](https://img.shields.io/badge/captioning-759CBC?)

- Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language [[pdf]](https://arxiv.org/pdf/2204.00598.pdf) [[code]](https://socraticmodels.github.io/#code)

  `arXiv 2022/04` ![](https://img.shields.io/badge/captioning-759CBC?style=flat-square) ![](https://img.shields.io/badge/retrieval-759CBC?style=flat-square) ![](https://img.shields.io/badge/visual--dialog-759CBC?style=flat-square) 

- Flamingo: a Visual Language Model for Few-Shot Learning [[pdf]](https://arxiv.org/abs/2204.14198) 

  `arXiv 2022/04` ![](https://img.shields.io/badge/image--classification-759CBC?) ![](https://img.shields.io/badge/VQA-759CBC?) ![](https://img.shields.io/badge/captioning-759CBC?)

- Language Models Can See: Plugging Visual Controls in Text Generation [[pdf]](https://arxiv.org/pdf/2205.02655.pdf) [[code]](https://github.com/yxuansu/MAGIC)

  `arXiv 2022/05` ![](https://img.shields.io/badge/MAGIC-CD6155?style=flat-square) ![](https://img.shields.io/badge/captioning-759CBC?)

- Zero-Shot Video Question Answering via Frozen Bidirectional Language Models [[pdf]](https://arxiv.org/pdf/2206.08155.pdf) 

  `arXiv 2022/06` ![](https://img.shields.io/badge/VideoQA-759CBC?)

- Visual Clues: Bridging Vision and Language Foundations for Image Paragraph Captioning [[pdf]](https://arxiv.org/pdf/2206.01843.pdf) 
  
  `arXiv 2022/06` ![](https://img.shields.io/badge/captioning-759CBC?)


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

- Unified Multimodal Pretraining and Prompt-based Tuning for Vision-Language Understanding and Generation [[pdf]](https://arxiv.org/abs/2112.05587)
  
   `arXiv 2021/12`

- Discovering Bugs in Vision Models using Off-the-shelf Image Generation and Captioning [[pdf]](https://arxiv.org/pdf/2208.08831.pdf)

  `arXiv 2022/08` ![](https://img.shields.io/badge/Dataset--Creation-759CBC?)


## Other Resources 
* [PromptPapers](https://github.com/thunlp/PromptPapers): A comprehensive curated list for prompting papers (mainly in natural language processing)
* Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing	[[pdf]](https://arxiv.org/abs/2107.13586)
  `arXiv 2021/07`


