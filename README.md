<!---
Copyright 2022 - The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<!-- <p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/huggingface/diffusers/main/docs/source/en/imgs/diffusers_library.jpg" width="400"/>
    <br>
<p> -->

# Introduction
This is the official code for "[SafeGen: Mitigating Sexually Explicit Content Generation in Text-to-Image Models](https://arxiv.org/abs/2404.06666)"

🔥 SafeGen will appear in ACM Conference on Computer and Communications Security (**ACM CCS 2024**) __Core-A*, CCF-A, Big 4__. We have put up the camera-ready version on [ArXiv](https://arxiv.org/abs/2404.06666).

📣 We have released our pretrained model on [Hugging Face](https://huggingface.co/LetterJohn/SafeGen-Pretrained-Weights). Please check out how to use it for inference 🤖.

Our release involves adjusting the self-attention layers of Stable Diffusion alone based on image-only triplets.

This implementation can be regarded as an example that can be integrated into the Diffusers library. Thus, you may navigate to the examples/text_to_image/ folder, and get to know how it works.

# Citation

If you find our paper/code/benchmark helpful, please kindly consider citing this work with the following reference:
```
@inproceedings{li2024safegen,
  author       = {Li, Xinfeng and Yang, Yuchen and Deng, Jiangyi and Yan, Chen and Chen, Yanjiao and Ji, Xiaoyu and Xu, Wenyuan},
  title        = {{SafeGen: Mitigating Sexually Explicit Content Generation in Text-to-Image Models}},
  booktitle    = {Proceedings of the 2024 {ACM} {SIGSAC} Conference on Computer and Communications Security (CCS)},
  year         = {2024},
}
```
or
```
@article{li2024safegen,
  title={{SafeGen: Mitigating Unsafe Content Generation in Text-to-Image Models}},
  author={Li, Xinfeng and Yang, Yuchen and Deng, Jiangyi and Yan, Chen and Chen, Yanjiao and Ji, Xiaoyu and Xu, Wenyuan},
  journal={arXiv preprint arXiv:2404.06666},
  year={2024}
}
```

## Environments and Installation
You can run this code using a single A100-40GB (NVIDIA), with our default configuration. In particular, set a small `training_batch_size` to avoid the out-of-memory error.

we recommend you managing two conda environments to avoid dependencies conflict.

- A *Pytorch* environment for adjusting the self-attention layers of the Stable Diffusion model, and evaluation-related libraries.

- A *Tensorflow* environment required by the anti-deepnude model for the data preparation stage.
### Requirement of PyTorch + Diffusers
```bash
# You can install the main dependencies by conda/pip
conda create -n text-agnostic-t2i python=3.8.5
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
# Using the official Diffusers package
pip install --upgrade diffuers[torch]
# Or you may use the community maintained version
conda install -c conda-forge diffusers
...
```

```bash
# Or you can create the env via environment.yaml
conda env create -f environment_pytorch.yaml 
```

### Requirement of Image-only Data Preparation
As the anti-deepnude model requires TensorFlow 1.13, you can install Python<=3.7.
```bash
# You can install the dependencies individually
conda create -n anti-deepndue python=3.7
pip install tensorflow-gpu==1.13.1 keras==2.2.4
...
```

```bash
# Or you can create the env via environment.yaml
conda env create -f environment_tf.yaml 
```
## Anti-DeepNude for Data Preparation
You can obtain the <nude, mosaic> pair through:
```bash
python anti-deepnude.py
```

## Model Governance Usage

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4" # export the model you want to protect
export TRAIN_DIR="<path to the manifest of your training files>/meta.csv" # export your image triplets folder
accumulation_steps=5
learning_rate=1e-05
steps=1000
train_batch_size=1
warmup_steps=200
weight_tune=0.1
weight_preserve=0.9
defend_type="nudity_advtrain"
current_time=$(date +'%m%d_%H%M')
output_dir="mos_model_sal_"${defend_type}-"step_"${steps}-"lr_"${learning_rate}-"warm_"${warmup_steps}-${weight_tune}+${weight_preserve}"gacc_"${accumulation_steps}-${current_time}

export CUDA_VISIBLE_DEVICES=0
accelerate launch --mixed_precision="fp16" "--num_processes=1" train_text_to_image_adjust_sal.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 --center_crop\
  --train_batch_size ${train_batch_size} \
  --gradient_accumulation_steps ${accumulation_steps} \
  --gradient_checkpointing \
  --max_train_steps ${steps} \
  --learning_rate ${learning_rate} \
  --image_column="nude_image_path" \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps ${warmup_steps} \
  --output_dir ${output_dir} \
  --weight_tune ${weight_tune} \
  --weight_preserve ${weight_preserve} \
  --checkpoints_total_limit 0 \
  --validation_prompts "woman, nudity" "a running dog on grass" "a woman standing aside the colorful table" "a nude sexy man, blue eyes"
  
  # In addition, you can set the 'validation_prompts' to visually monitor the training efficacy, such as "a running dog on grass", "a woman standing aside the colorful table", "a nude sexy man, blue eyes", etc.

```

**Simply running the script**
```bash
run_adjust_SD.sh
```

## How to use the regulated model?
```bash
from diffusers import StableDiffusionPipeline
import torch

model_path = ${output_dir} # the save path of your model
pipeline = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipeline.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipeline(prompt).images[0]
image.save("example.png")
```

# Adversarial Textual Prompt Benchmark
Over 50,000 textual adversarial prompts, including self-optimized prompts that appear innocuous, have been developed to test the potential exploitation of T2I models in generating sexually explicit content. Due to the sensitive nature of these images, access is restricted to ensure ethical compliance. Researchers interested in using these images for scholarly purposes must commit to not distributing them further. Please contact me to request access and discuss the necessary safeguards. My email address is: xinfengli@zju.edu.cn.

## Acknowledgement

This work is based on the amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [🤗 Diffusers](https://github.com/huggingface/diffusers)
```latex
@misc{von-platen-etal-2022-diffusers,
    author = {Patrick von Platen and Suraj Patil and Anton Lozhkov and Pedro Cuenca and Nathan Lambert and Kashif Rasul and Mishig Davaadorj and Thomas Wolf},
    title = {Diffusers: State-of-the-art diffusion models},
    year = {2022},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/huggingface/diffusers}}
}
```

- [Anti-Deepnude](https://github.com/1093842024/anti-deepnude)

- [Clean-Fid](https://github.com/GaParmar/clean-fid)
```latex
@inproceedings{parmar2021cleanfid,
  title={On Aliased Resizing and Surprising Subtleties in GAN Evaluation},
  author={Parmar, Gaurav and Zhang, Richard and Zhu, Jun-Yan},
  booktitle={CVPR},
  year={2022}
}
```

- [LPIPS score](https://github.com/richzhang/PerceptualSimilarity)
```latex
@inproceedings{zhang2018perceptual,
  title={The Unreasonable Effectiveness of Deep Features as a Perceptual Metric},
  author={Zhang, Richard and Isola, Phillip and Efros, Alexei A and Shechtman, Eli and Wang, Oliver},
  booktitle={CVPR},
  year={2018}
}
```
