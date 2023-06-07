# Awesome Prompting Papers in Computer Vision
A curated list of prompt-based papers in computer vision and vision-language learning. 


- [Awesome Prompting Papers in Computer Vision](#awesome-prompting-papers-in-computer-vision)
    - [Keywords](#keywords)
  - [Vision Prompt](#vision-prompt)
  - [Vision-Language Prompt](#vision-language-prompt)
    - [Language-Interactable Prompt](#language-interactable-prompt)
    - [Vision-Language Instruction Tuning](#vision-language-instruction-tuning)
  - [More Resources](#more-resources)

### Keywords
* Task tag, e.g., ![](https://img.shields.io/badge/object--detection-759CBC?style=flat-square) ![](https://img.shields.io/badge/VQA-759CBC?style=flat-square)
* Abbreviation tag, e.g., ![](https://img.shields.io/badge/CLIP-CD6155?style=flat-square)
* Characteristic tag: Some characteristic makes this paper unique, e.g., ![](https://img.shields.io/badge/NAS-BC9575?style=flat-square) ![](https://img.shields.io/badge/unsupervised-BC9575?style=flat-square)
* **Bold font**: We highlight some pilot work that may contribute to the prevalence of visual prompting.


## Vision Prompt
This section collects papers prompting pretrained vision foundation models (e.g., ViT) for parameter-efficient adaptation.

- **Learning to Prompt for Continual Learning** [[paper]](https://arxiv.org/abs/2112.08654) [[code]](https://github.com/google-research/l2p)

  `CVPR 2022`  ![](https://img.shields.io/badge/continual--learning-759CBC?style=flat-square)

- **Visual Prompt Tuning** [[paper]](https://arxiv.org/pdf/2203.12119.pdf) [[code]](https://github.com/KMnP/vpt)
   
  `ECCV 2022` ![](https://img.shields.io/badge/VPT-CD6155?style=flat-square)  
 
- DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning [[paper]](https://arxiv.org/pdf/2204.04799.pdf) [[code]](https://github.com/google-research/l2p)

  `ECCV 2022` ![](https://img.shields.io/badge/continual--learning-759CBC?style=flat-square)
  
- AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition [[paper]](https://arxiv.org/abs/2205.13535) [[code]](https://github.com/ShoufaChen/AdaptFormer)
  
  `NeurIPS 2022`  ![](https://img.shields.io/badge/action--recognition-759CBC?style=flat-square)

- Scaling & Shifting Your Features: A New Baseline for Efficient Model Tuning [[paper]](https://arxiv.org/abs/2210.08823) [[code]](https://github.com/dongzelian/SSF)

  `NeurIPS 2022` 

- P2P: Tuning Pre-trained Image Models for Point Cloud Analysis with Point-to-Pixel Prompting [[paper]](https://arxiv.org/abs/2208.02812) [[code]](https://github.com/wangzy22/P2P)

  `NeurIPS 2022` ![](https://img.shields.io/badge/3D--point--cloud--tasks-759CBC?) 

- Generative Visual Prompt: Unifying Distributional Control of Pre-Trained Generative Models [[paper]](https://arxiv.org/abs/2209.06970) [[code]](https://github.com/ChenWu98/Generative-Visual-Prompt)

  `NeurIPS 2022` ![](https://img.shields.io/badge/image--generation-759CBC?) 

- Visual Prompting via Image Inpainting [[paper]]() [[code]](https://github.com/amirbar/visual_prompting)

  `NeurIPS 2022` ![](https://img.shields.io/badge/visual--in--context--learning-759CBC?) ![](https://img.shields.io/badge/image--generation-759CBC?) 

- Decorate the Newcomers: Visual Domain Prompt for Continual Test Time Adaptation [[paper]](https://arxiv.org/abs/2212.04145)

  `AAAI 2023` 

- LPT: Long-tailed Prompt Tuning for Image Classification [[paper]](https://openreview.net/forum?id=8pOVAeo8ie) 

  `ICLR 2023` 

- Diversity-Aware Meta Visual Prompting [[paper]](https://arxiv.org/abs/2303.08138) [[code]](https://github.com/shikiw/DAM-VP)

  `CVPR 2023` 

- Semantic Prompt for Few-Shot Image Recognition [[paper]](https://arxiv.org/abs/2303.14123) 

  `CVPR 2023` ![](https://img.shields.io/badge/few--shot--learning-759CBC?) 


- Visual Prompt Tuning for Generative Transfer Learning [[paper]](https://arxiv.org/abs/2210.00990) [[code]](https://github.com/google-research/generative_transfer)

  `CVPR 2023` ![](https://img.shields.io/badge/image--generative--tasks-759CBC?) 

- CORA: Adapting CLIP for Open-Vocabulary Detection with Region Prompting and Anchor Pre-Matching [[paper]](https://arxiv.org/abs/2303.13076) [[code]](https://github.com/tgxs002/CORA)

  `CVPR 2023` ![](https://img.shields.io/badge/open--vocabulary--detection-759CBC?) 

- Images Speak in Images: A Generalist Painter for In-Context Visual Learning [[paper]](https://arxiv.org/abs/2212.02499) [[code]](https://arxiv.org/abs/2212.02499)

  `CVPR 2023` ![](https://img.shields.io/badge/image--generation-759CBC?)  ![](https://img.shields.io/badge/in--context--learning-759CBC?)

- PIVOT: Prompting for Video Continual Learning [[paper]](https://arxiv.org/abs/2212.04842)

  `CVPR 2023` ![](https://img.shields.io/badge/continual--learning-759CBC?) 

- Learning Expressive Prompting With Residuals for Vision Transformers [[paper]](https://arxiv.org/abs/2303.15591)

  `CVPR 2023` ![](https://img.shields.io/badge/semantic--segmentation-759CBC?) 

- BlackVIP: Black-Box Visual Prompting for Robust Transfer Learning [[paper]](https://arxiv.org/abs/2303.14773) [[code]](https://github.com/changdaeoh/BlackVIP)

  `CVPR 2023` ![](https://img.shields.io/badge/black--box--optimization-759CBC?) 

- Visual Prompt Multi-Modal Tracking [[paper]](https://arxiv.org/abs/2303.10826) [[code]](https://github.com/jiawen-zhu/ViPT)

  `CVPR 2023` ![](https://img.shields.io/badge/object--training-759CBC?) 

- A-La-Carte Prompt Tuning (APT): Combining Distinct Data Via Composable Prompting [[paper]](https://arxiv.org/abs/2302.07994) 

  `CVPR 2023` ![](https://img.shields.io/badge/continual--learning-759CBC?) 

- Understanding and Improving Visual Prompting: A Label-Mapping Perspective [[paper]](https://arxiv.org/abs/2211.11635) [[code]](https://github.com/OPTML-Group/ILM-VP)

  `CVPR 2023` 

- Visual Query Tuning: Towards Effective Usage of Intermediate Representations for Parameter and Memory Efficient Transfer Learning [[paper]](https://arxiv.org/abs/2212.03220) [[code]](https://github.com/andytu28/VQT)

  `CVPR 2023` 

- Explicit Visual Prompting for Low-Level Structure Segmentations low-level segmentation [[paper]](https://arxiv.org/abs/2303.10883) [[code]](https://github.com/NiFangBaAGe/Explicit-Visual-Prompt)

  `CVPR 2023` ![](https://img.shields.io/badge/low--level--segmentation-759CBC?) 

- Understanding and Improving Visual Prompting: A Label-Mapping Perspective [[paper]](https://arxiv.org/abs/2211.11635) [[code]](https://github.com/OPTML-Group/ILM-VP)

  `CVPR 2023` 

**ArXiv Papers**

- **Exploring Visual Prompts for Adapting Large-Scale Models** [[paper]](https://arxiv.org/pdf/2203.17274.pdf) [[code]](https://github.com/hjbahng/visual_prompting)

  `arXiv 2022/03` 

- Vision Transformer Adapter for Dense Predictions [[paper]](https://arxiv.org/pdf/2205.08534.pdf) [[code]](https://github.com/czczup/ViT-Adapter)
  
  `arXiv 2022/05` ![](https://img.shields.io/badge/object--detection-759CBC?style=flat-square) ![](https://img.shields.io/badge/instance--segmentaion-759CBC?style=flat-square)

- Neural Prompt Search [[paper]](https://arxiv.org/abs/2206.04673) [[code]](https://github.com/Davidzhangyuanhan/NOAH)

  `arXiv 2022/06` ![](https://img.shields.io/badge/NOAH-CD6155?style=flat-square)  ![](https://img.shields.io/badge/NAS-BC9575?style=flat-square)

- Convolutional Bypasses Are Better Vision Transformer Adapters [[paper]](https://arxiv.org/abs/2207.07039) [[code]](https://github.com/JieShibo/PETL-ViT)

  `arXiv 2022/07`   

- Conv-Adapter: Exploring Parameter Efficient Transfer Learning for ConvNets [[paper]](https://arxiv.org/pdf/2208.07463.pdf) 

  `arXiv 2022/08`   

- Prompt Vision Transformer for Domain Generalization [[paper]](https://arxiv.org/pdf/2208.08914.pdf) 

  `arXiv 2022/08`  ![](https://img.shields.io/badge/domain--generalization-759CBC?style=flat-square) 

- Prompt-Matched Semantic Segmentation [[paper]](https://arxiv.org/abs/2208.10159) 

  `arXiv 2022/08`  ![](https://img.shields.io/badge/segmentation-759CBC?style=flat-square) 
  
- Visual Prompt Tuning for Test-time Domain Adaptation [[paper]](https://arxiv.org/abs/2210.04831)

  `arXiv 2022/10`   
  
- Visual Prompting for Adversarial Robustness [[paper]](https://arxiv.org/abs/2210.06284)

  `arXiv 2022/10`  ![](https://img.shields.io/badge/adversarial--robustness-759CBC?style=flat-square) 

- Prompt Generation Networks for Efficient Adaptation of Frozen Vision Transformers [[paper]](https://arxiv.org/abs/2210.06466) [[code]](https://github.com/jochemloedeman/PGN)

  `arXiv 2022/10`   

- Towards a Unified View on Visual Parameter-Efficient Transfer Learning [[paper]](https://arxiv.org/abs/2210.00788) [[code]](https://github.com/bruceyo/V-PETL)
  
  `arXiv 2022/10` ![](https://img.shields.io/badge/action--recognition-759CBC?style=flat-square)
  
- Multitask Vision-Language Prompt Tuning [[paper]](https://arxiv.org/abs/2211.11720) [[code]](https://github.com/sIncerass/MVLPT)

  `arXiv 2022/11` ![](https://img.shields.io/badge/MVLPT-CD6155?style=flat-square)   ![](https://img.shields.io/badge/multitask--learning-BC9575?style=flat-square)


## Vision-Language Prompt
This section collects papers prompting pretrained vision-language foundation models (e.g., CLIP) for parameter-efficient adaptation.

- **Learning Transferable Visual Models From Natural Language Supervision** [[paper]](https://arxiv.org/abs/2103.00020) [[code]](https://github.com/OpenAI/CLIP) 

  `ICML 2021` ![](https://img.shields.io/badge/CLIP-CD6155?style=flat-square) 

- **Learning to Prompt for Vision-Language Models** [[paper]](https://arxiv.org/abs/2109.01134) [[code]](https://github.com/KaiyangZhou/CoOp)

  `IJCV 2022`  ![](https://img.shields.io/badge/CoOP-CD6155?style=flat-square) 

- Prompt Distribution Learning [[paper]](https://arxiv.org/pdf/2205.03340.pdf)

  `CVPR 2022` 

- Conditional Prompt Learning for Vision-Language Models [[paper]](https://arxiv.org/pdf/2203.05557.pdf) [[code]](https://github.com/KaiyangZhou/CoOp)

  `CVPR 2022`  ![](https://img.shields.io/badge/CoCoOP-CD6155?style=flat-square) 

- DenseCLIP: Language-Guided Dense Prediction with Context-Aware Prompting	[[paper]](https://arxiv.org/pdf/2112.01518.pdf) [[code]](https://github.com/raoyongming/denseclip) 

  `CVPR 2022` ![](https://img.shields.io/badge/detection-759CBC?style=flat-square) ![](https://img.shields.io/badge/segmentation-759CBC?style=flat-square) 

- Bridge-Prompt: Towards Ordinal Action Understanding in Instructional Videos [[paper]](https://arxiv.org/pdf/2203.14104.pdf) [[code]](https://github.com/ttlmh/Bridge-Prompt) 

  `CVPR 2022` ![](https://img.shields.io/badge/action--recognition-759CBC?style=flat-square) ![](https://img.shields.io/badge/action--segmentation-759CBC?style=flat-square)

- PointCLIP: Point Cloud Understanding by CLIP [[paper]](https://arxiv.org/pdf/2112.02413.pdf) [[code]](https://github.com/ZrrSkywalker/PointCLIP)

  `CVPR 2022`  ![](https://img.shields.io/badge/point--cloud-759CBC?style=flat-square)

- VL-Adapter: Parameter-Efficient Transfer Learning for Vision-and-Language Tasks [[paper]](https://arxiv.org/pdf/2112.06825.pdf) [[code]](https://github.com/ylsung/VL_adapter)
  
    `CVPR 2022`  ![](https://img.shields.io/badge/VQA-759CBC?style=flat-square) ![](https://img.shields.io/badge/VideoQA-759CBC?style=flat-square) ![](https://img.shields.io/badge/captioning-759CBC?style=flat-square)

- A Good Prompt Is Worth Millions of Parameters? Low-resource Prompt-based Learning for Vision-Language Models	[[paper]](https://arxiv.org/abs/2110.08484)

  `ACL 2022`  ![](https://img.shields.io/badge/VQA-759CBC?)  ![](https://img.shields.io/badge/captioning-759CBC?)

- Can Language Understand Depth? [[paper]](https://arxiv.org/pdf/2207.01077.pdf) [[code]](https://github.com/Adonis-galaxy/DepthCLIP)

  `ACM MM 2022` ![](https://img.shields.io/badge/depthclip-CD6155?style=flat-square)  ![](https://img.shields.io/badge/depth--estimation-759CBC?)

- Expanding Language-Image Pretrained Models for General Video Recognition [[paper]](https://arxiv.org/abs/2208.02816) [[code]](https://aka.ms/X-CLIP)

  `ECCV 2022` ![](https://img.shields.io/badge/action--recognition-759CBC?style=flat-square)
  
- Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification [[paper]](https://arxiv.org/pdf/2207.09519.pdf) [[code]](https://github.com/gaopengcuhk/tip-adapter)

  `ECCV 2022` 
  
- OrdinalCLIP: Learning Rank Prompts for Language-Guided Ordinal Regression [[paper]](https://arxiv.org/abs/2206.02338)

  `NeurIPS 2022` ![](https://img.shields.io/badge/ordinal--regression-759CBC?style=flat-square)

- Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models [[paper]](https://arxiv.org/pdf/2209.07511.pdf) [[code]](https://azshue.github.io/TPT/)

  `NeurIPS 2022` 
 


- Learning to Decompose Visual Features with Latent Textual Prompts [[paper]](https://openreview.net/forum?id=wtcud6HroZr)

  `ICLR 2023` 

- PLOT: Prompt Learning with Optimal Transport for Vision-Language Models [[paper]](https://openreview.net/forum?id=zqwryBoXYnh) [[code]](https://github.com/CHENGY12/PLOT)

  `ICLR 2023` 

- Visual-Language Prompt Tuning with Knowledge-guided Context Optimization [[paper]](https://arxiv.org/abs/2303.13283) [[code]](https://github.com/htyao89/KgCoOp)

  `CVPR 2023` ![](https://img.shields.io/badge/image--classification-759CBC?)

- Open-Set Fine-Grained Retrieval Via Prompting Vision-Language Evaluator [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Open-Set_Fine-Grained_Retrieval_via_Prompting_Vision-Language_Evaluator_CVPR_2023_paper.pdf) 

  `CVPR 2023` ![](https://img.shields.io/badge/open--set--retrieval-759CBC?) 

- Multimodal Prompting With Missing Modalities for Visual Recognition [[paper]](https://arxiv.org/abs/2303.03369) [[code]](https://github.com/YiLunLee/Missing_aware_prompts)

  `CVPR 2023` 

- Efficient Multimodal Fusion Via Interactive Prompting [[paper]](https://arxiv.org/abs/2304.06306)

  `CVPR 2023` ![](https://img.shields.io/badge/multimodal--classification-759CBC?) 

- Hierarchical Prompt Learning for Multi-Task Learning [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Hierarchical_Prompt_Learning_for_Multi-Task_Learning_CVPR_2023_paper.pdf) [[code]](https://github.com/lynlynlyn/hipro)

  `CVPR 2023` ![](https://img.shields.io/badge/multitask--learning-759CBC?) 

- Text-Visual Prompting for Efficient 2D Temporal Video Grounding [[paper]](https://arxiv.org/abs/2303.04995)

  `CVPR 2023` ![](https://img.shields.io/badge/video--grounding-759CBC?) 

- VoP: Text-Video Co-operative Prompt Tuning for Cross-Modal Retrieval [[paper]](https://arxiv.org/abs/2211.12764) [[code]](https://github.com/bighuang624/VoP)

  `CVPR 2023` ![](https://img.shields.io/badge/text--video--retrieval-759CBC?) 

- MaPLe: Multi-modal Prompt Learning [[paper]](https://arxiv.org/abs/2210.03117) [[code]](https://github.com/muzairkhattak/multimodal-prompt-learning)


  `CVPR 2023` 

- Texts as Images in Prompt Tuning for Multi-Label Image Recognition [[paper]](https://arxiv.org/abs/2211.12739) [[code]](https://github.com/guozix/TaI-DPT)

  `CVPR 2023` ![](https://img.shields.io/badge/multi--label--recognition-759CBC?) 


- Vita-CLIP: Video and Text Adaptive CLIP Via Multimodal Prompting [[paper]](https://arxiv.org/abs/2304.03307) [[code]](https://github.com/TalalWasim/Vita-CLIP)

  `CVPR 2023` ![](https://img.shields.io/badge/action--recognition-759CBC?) 


- LASP: Text-to-Text Optimization for Language-Aware Soft Prompting of Vision & Language Models [[paper]](https://arxiv.org/abs/2210.01115) [[code]](https://www.adrianbulat.com/lasp)

  `CVPR 2023` 

- $\pi$-Tuning: Transferring Multimodal Foundation Models with Optimal Multi-task Interpolation [[paper]](https://arxiv.org/abs/2304.14381) [[code]](https://github.com/TencentARC/pi-Tuning)

  `ICML 2023` 

- POUF: Prompt-oriented unsupervised fine-tuning for large pre-trained models [[paper]](https://arxiv.org/abs/2305.00350) [[code]](https://github.com/korawat-tanwisuth/POUF)

  `ICML 2023` 
  
- Rethinking the Openness of CLIP [[paper]](https://arxiv.org/abs/2206.01986) [[code]](https://github.com/lancopku/clip-openness)

  `ACL 2023` ![](https://img.shields.io/badge/REPE-CD6155?style=flat-square)
  
<bar>

**ArXiv Papers**

- **Colorful Prompt Tuning for Pre-trained Vision-Language Models** [[paper]](https://arxiv.org/abs/2109.11797) 

  `arXiv 2021/08` ![](https://img.shields.io/badge/CPT-CD6155?style=flat-square) ![](https://img.shields.io/badge/grounding-759CBC?style=flat-square) 


- ActionCLIP: A New Paradigm for Video Action Recognition [[paper]](https://arxiv.org/abs/2109.08472) [[code]](https://github.com/sallymmx/ActionCLIP)

  `arXiv 2021/09` ![](https://img.shields.io/badge/action--recognition-759CBC?style=flat-square)

- CLIP-Adapter: Better Vision-Language Models with Feature Adapters [[paper]](https://arxiv.org/abs/2110.04544) [[code]](https://github.com/gaopengcuhk/clip-adapter)

  `arXiv 2021/10` 

- Amortized Prompt: Lightweight Fine-Tuning for CLIP in Domain Generalization [[paper]](https://arxiv.org/abs/2111.12853)

  `arXiv 2021/11` ![](https://img.shields.io/badge/domain--generalization-BC9575?style=flat-square)

- Prompting Visual-Language Models for Efficient Video Understanding [[paper]](https://arxiv.org/abs/2112.04478) [[code]](https://github.com/ju-chen/Efficient-Prompt)

  `arXiv 2021/12` ![task](https://img.shields.io/badge/action--recognition-759CBC?style=flat-square) ![task](https://img.shields.io/badge/action--localization-759CBC?style=flat-square) ![task](https://img.shields.io/badge/retrieval-759CBC?style=flat-square)

- Unsupervised Prompt Learning for Vision-Language Models [[paper]](https://arxiv.org/pdf/2204.03649.pdf) [[code]](https://github.com/tonyhuang2022/UPL)

  `arXiv 2022/04` ![](https://img.shields.io/badge/UPL-CD6155?style=flat-square) ![](https://img.shields.io/badge/unsupervised-BC9575?style=flat-square)

- Prompt-aligned Gradient for Prompt Tuning [[paper]](https://arxiv.org/abs/2205.14865) [[code]](https://github.com/BeierZhu/Prompt-align)

  `arXiv 2022/05` 


- Parameter-Efficient Image-to-Video Transfer Learning [[paper]](https://arxiv.org/pdf/2206.13559.pdf)

  `arXiv 2022/06`  ![](https://img.shields.io/badge/ST--adapter-CD6155?style=flat-square) ![task](https://img.shields.io/badge/action--recognition-759CBC?style=flat-square)

- DualCoOp: Fast Adaptation to Multi-Label Recognition with Limited Annotations [[paper]](https://arxiv.org/abs/2206.09541)

  `arXiv 2022/06` ![task](https://img.shields.io/badge/multilabel--recognition-759CBC?style=flat-square)

- Prompt Tuning for Generative Multimodal Pretrained Models [[paper]](https://arxiv.org/abs/2208.02532) [[code]](https://github.com/OFA-Sys/OFA)

  `arXiv 2022/06` ![](https://img.shields.io/badge/VQA-759CBC?style=flat-square) ![](https://img.shields.io/badge/captioning-759CBC?style=flat-square)  ![](https://img.shields.io/badge/referring--expression-759CBC?style=flat-square)  ![](https://img.shields.io/badge/visual--entailment-759CBC?style=flat-square) 
  
- Prompt Tuning with Soft Context Sharing for Vision-Language Models [[paper]](https://arxiv.org/pdf/2208.13474.pdf)

  `arXiv 2022/08`   ![](https://img.shields.io/badge/multi--task--learning-759CBC?style=flat-square) 
 

  
- CPL: Counterfactual Prompt Learning for Vision and Language Models [[paper]](https://arxiv.org/abs/2210.10362) [[code]](https://github.com/eric-ai-lab/CPL)

  `arXiv 2022/10`   ![](https://img.shields.io/badge/retrieval-759CBC?style=flat-square) 
![](https://img.shields.io/badge/VQA-759CBC?style=flat-square) 


- Understanding and Mitigating Overfitting in Prompt Tuning for Vision-Language Models [[paper]](https://arxiv.org/abs/2211.02219) [[code]](https://github.com/machengcheng2016/Subspace-Prompt-Learning)

  `arXiv 2022/10`  ![](https://img.shields.io/badge/object--detection-759CBC?style=flat-square) 
![](https://img.shields.io/badge/semantic--segmentation-759CBC?style=flat-square) 


- Unified Vision and Language Prompt Learning [[paper]](https://arxiv.org/abs/2210.07225)

  `arXiv 2022/10` 
    

- Multi-Prompt Alignment for Multi-source Unsupervised Domain Adaptation [[paper]](https://arxiv.org/abs/2209.15210)

  `arXiv 2022/10` ![](https://img.shields.io/badge/domain--adaptation-759CBC?style=flat-square)

- Prompt Pre-Training with Twenty-Thousand Classes for Open-Vocabulary Visual Recognition [[paper]](https://arxiv.org/abs/2304.04704) [[code]](https://github.com/amazon-science/prompt-pretraining)
    
  `arXiv 2023/04`
    
### Language-Interactable Prompt
Language-interactable prompter develops zero/few-shot capabilities by prompting **several independent foundational models** (VLMs, LLMs, VMs, etc.) with the language interface. One of the most attractive applications is [multimodal chatbot](https://github.com/zjr2000/Awesome-Multimodal-Assistant).


- **Multimodal Few-Shot Learning with Frozen Language Models** [[paper]](https://arxiv.org/abs/2106.13884)

  `NeurIPS 2021` ![](https://img.shields.io/badge/VQA-759CBC?)

- An Empirical Study of GPT-3 for Few-Shot Knowledge-Based VQA [[paper]](https://arxiv.org/pdf/2109.05014.pdf) [[code]](https://github.com/microsoft/PICa) 

  `AAAI 2022` ![](https://img.shields.io/badge/VQA-759CBC?)

- VisualGPT: Data-efficient Adaptation of Pretrained Language Models for Image Captioning [[paper]](https://arxiv.org/pdf/2102.10407.pdf) [[code]](https://github.com/Vision-CAIR/VisualGPT)

  `CVPR 2022` ![](https://img.shields.io/badge/captioning-759CBC?)

- **Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language** [[paper]](https://arxiv.org/pdf/2204.00598.pdf) [[code]](https://socraticmodels.github.io/#code)

  `ICLR 2023` ![](https://img.shields.io/badge/captioning-759CBC?style=flat-square) ![](https://img.shields.io/badge/retrieval-759CBC?style=flat-square) ![](https://img.shields.io/badge/visual--dialog-759CBC?style=flat-square) 

<bar>

**Arxiv Papers**
- **Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models** [[paper]](https://arxiv.org/abs/2303.04671) [[code]](https://github.com/microsoft/TaskMatrix) [[demo]](https://huggingface.co/spaces/microsoft/visual_chatgpt) 
`arXiv 2023/03`  ![](https://img.shields.io/badge/Visual--ChatGPT-CD6155?style=flat-square) ![](https://img.shields.io/badge/multimodal--chatbot-759CBC?) ![](https://img.shields.io/badge/LLMs-(chatGPT)-759CBC?) 

- **Chameleon: Plug-and-play compositional reasoning with large language models** [[paper]](https://arxiv.org/abs/2304.09842) [[code]](https://github.com/lupantech/chameleon-llm)
 `arXiv 2023/04` ![](https://img.shields.io/badge/Chameleon-CD6155?style=flat-square) ![](https://img.shields.io/badge/multimodal--chatbot-759CBC?) ![](https://img.shields.io/badge/LLMs-(GPT4)-759CBC?) 

- ClipCap: CLIP Prefix for Image Captioning	[[paper]](https://arxiv.org/abs/2111.09734) [[code]](https://github.com/rmokady/CLIP_prefix_caption)

  `arXiv 2021/11` ![](https://img.shields.io/badge/captioning-759CBC?)

- Flamingo: a Visual Language Model for Few-Shot Learning [[paper]](https://arxiv.org/abs/2204.14198) 

  `arXiv 2022/04` ![](https://img.shields.io/badge/VQA-759CBC?) ![](https://img.shields.io/badge/captioning-759CBC?)

- Language Models Can See: Plugging Visual Controls in Text Generation [[paper]](https://arxiv.org/pdf/2205.02655.pdf) [[code]](https://github.com/yxuansu/MAGIC)

  `arXiv 2022/05` ![](https://img.shields.io/badge/MAGIC-CD6155?style=flat-square) ![](https://img.shields.io/badge/captioning-759CBC?)

- Zero-Shot Video Question Answering via Frozen Bidirectional Language Models [[paper]](https://arxiv.org/pdf/2206.08155.pdf) 

  `arXiv 2022/06` ![](https://img.shields.io/badge/VideoQA-759CBC?)

- Visual Clues: Bridging Vision and Language Foundations for Image Paragraph Captioning [[paper]](https://arxiv.org/pdf/2206.01843.pdf) 
  
  `arXiv 2022/06` ![](https://img.shields.io/badge/captioning-759CBC?)


### Vision-Language Instruction Tuning

The goal of vision-language instruction tuning is to train a model that can effectively understand  instructions for general-purpose multimodal tasks. 

- Visual Instruction Tuning [[paper]](https://arxiv.org/abs/2304.08485) [[code]](https://github.com/haotian-liu/LLaVA) [[demo]](https://llava.hliu.cc/)

  `arXiv 2023/04` 

- MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models [[paper]](https://arxiv.org/abs/2304.10592) [[code]](https://github.com/Vision-CAIR/MiniGPT-4) [[demo]](minigpt-4.github.io)

  `arXiv 2023/04` 

- Otter: A Multi-Modal Model with In-Context Instruction Tuning [[paper]](https://arxiv.org/abs/2305.03726) [[code]](https://github.com/Luodian/Otter) [[demo]](otter.cliangyu.com/)

  `arXiv 2023/05` 

- MultiModal-GPT: A Vision and Language Model for Dialogue with Humans [[paper]](https://arxiv.org/abs/2305.04790) [[code]](https://github.com/open-mmlab/Multimodal-GPT)

  `arXiv 2023/05` 

- InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning [[paper]](https://arxiv.org/abs/2305.06500) [[code]](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip) 

  `arXiv 2023/05` 




## More Resources 
* [PromptPapers](https://github.com/thunlp/PromptPapers): A comprehensive curated list for prompting papers (mainly in natural language processing)
* [Awesome Multimodal Assistant](https://github.com/zjr2000/Awesome-Multimodal-Assistant): a curated list for vision-language instruction tuning and LLM-based chatbot.


