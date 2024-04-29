# CorrespondentDream: Enhancing 3D Fidelity of Text-to-3D using Cross-View Correspondences (CVPR'24)
Seungwook Kim, Kejie Li, Xueqing Deng, Yichun Shi, Minsu Cho, Peng Wang

| [Project Page](https://wookiekim.github.io/CorrespondentDream/) | [Paper](https://arxiv.org/abs/2404.10603) |

![correspondentdream-teaser](./CorrespondentDream_qual.jpg)

## Installation

### Install threestudio

**This part is the same as original threestudio. Skip it if you already have installed the environment.**

See [installation.md](docs/installation.md) for additional information, including installation via Docker.

- You must have an NVIDIA graphics card with at least 20GB VRAM and have [CUDA](https://developer.nvidia.com/cuda-downloads) installed.
- Install `Python >= 3.8`.
- (Optional, Recommended) Create a virtual environment:

```sh
python3 -m virtualenv venv
. venv/bin/activate

# Newer pip versions, e.g. pip-23.x, can be much faster than old versions, e.g. pip-20.x.
# For instance, it caches the wheels of git packages to avoid unnecessarily rebuilding them later.
python3 -m pip install --upgrade pip
```

- Install `PyTorch >= 1.12`. We have tested on `torch1.12.1+cu113` and `torch2.0.0+cu118`, but other versions should also work fine.
- NOTE: We found that it is important to use the version of xformers that is aligned with the version of PyTorch. Refer to [xformers](https://github.com/facebookresearch/xformers) to check for compatibility.

```sh
# torch1.12.1+cu113
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# or torch2.0.0+cu118
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

or:

```sh
# xformer 0.0.25 is compatible with pytorch 2.2.1
pip3 install torch==2.0.0 torchvision --index-url https://download.pytorch.org/whl/cu118
pip3 install -U xformers==0.0.17
```

- (Optional, Recommended) Install ninja to speed up the compilation of CUDA extensions:

```sh
pip install ninja
```

- Install dependencies:

```sh
pip install -r requirements.txt
```

### Install MVDream and update the UNet to retrieve features to establish correspondences
[MVDream](https://github.com/bytedance/MVDream) multi-view diffusion model, which is used as the prior in CorrespondentDream, is provided in a different codebase. Install it by:

```sh
git clone https://github.com/bytedance/MVDream extern/MVDream
pip install -e extern/MVDream 
```

Then, replace the original UNet code of MVDream so that we can retrieve the features from the upsampling layers of the UNet, which are used to establish cross-view correspondences:

```sh
cp openaimodel.py extern/MVDream/mvdream/ldm/modules/diffusionmodules/openaimodel.py
```


## Quickstart

While the original MVDream provides two configurations (one using soft-shading and one without), CorrespondentDream opts to always use shading with normal computation, so that the corrected 3D infidelities are better visible.
We provide two configurations for CorrespondentDream, one with our proposed CFG scheduling (refer to supplementary) for improved smoothness while preserving the details, and one without.
Note that the results in the main paper did not use CFG scheduling.

Because we use a lowered resolution size of 128x128, our code can be run on a V100 GPU in most cases to compute normal. In order to use a higher resolution of 256x256 as MVDream, it would need an A100 GPU in most cases.

```sh
# CorrespondentDream without CFG scheduling
# TODO: fill this part
python launch.py --config configs/corr-mvdream-sd21-shading.yaml --train --gpu 0 system.prompt_processor.prompt="An astronaut riding a horse"

# CorrespondentDream with CFG scheduling
python launch.py --config configs/corr-mvdream-sd21-shading-cfg-scheduling.yaml --train --gpu 0 system.prompt_processor.prompt="An astronaut riding a horse"
```

### Resume from checkpoints

If you want to resume from a checkpoint, do:

```sh
# resume training from the last checkpoint, you may replace last.ckpt with any other checkpoints
python launch.py --config path/to/trial/dir/configs/parsed.yaml --train --gpu 0 resume=path/to/trial/dir/ckpts/last.ckpt
# if the training has completed, you can still continue training for a longer time by setting trainer.max_steps
python launch.py --config path/to/trial/dir/configs/parsed.yaml --train --gpu 0 resume=path/to/trial/dir/ckpts/last.ckpt trainer.max_steps=20000
# you can also perform testing using resumed checkpoints
python launch.py --config path/to/trial/dir/configs/parsed.yaml --test --gpu 0 resume=path/to/trial/dir/ckpts/last.ckpt
# note that the above commands use parsed configuration files from previous trials
# which will continue using the same trial directory
# if you want to save to a new trial directory, replace parsed.yaml with raw.yaml in the command

# only load weights from saved checkpoint but dont resume training (i.e. dont load optimizer state):
python launch.py --config path/to/trial/dir/configs/parsed.yaml --train --gpu 0 system.weights=path/to/trial/dir/ckpts/last.ckpt
```

## Tips
- **Preview**. Generating 3D content with SDS would a take a lot of time. So we suggest to use the 2D multi-view image generation [MVDream](https://github.com/bytedance/MVDream) to test if the model can really understand the text before using it for 3D generation.
- **Compatibility of CorrespondentDream with text prompt**. CorrespondentDream may not always be effective for all types of different prompts, particularly given too homogeneous or shiny surfaces. It will usually not make the output worse, so it's worth a try at a cost of slightly increased time.
- **Rescale Factor**. We introducte rescale adjustment from [Shanchuan et al.](https://arxiv.org/abs/2305.08891) to alleviate the texture over-saturation from large CFG guidance. However, in some cases, we find it to cause floating noises in the generated scene and consequently OOM issue. Therefore we reduce the rescale factor from 0.7 in original paper to 0.5. However, if you still encounter such a problem, please try to further reduce `system.guidance.recon_std_rescale=0.3`.

## Credits

This code is built on the [MVDream](https://github.com/bytedance/MVDream), [MVDream-threestudio](https://github.com/bytedance/MVDream-threestudio), and [threestudio-project](https://github.com/threestudio-project/threestudio). 
Big thanks to all the projects and their open-sourced code!

## Citing

If you find MVDream helpful, please consider citing:

```
@article{kim2024correspondentdream,
  title={Enhancing 3D Fidelity of Text-to-3D using Cross-View Correspondences},
  author={Kim, Seungwook and Li, Kejie and Deng, Xueqing and Shi, Yichun and Cho, Minsu and Wang, Peng},
  journal={arXiv preprint arXiv:2404.10603},
  year={2024}
}
```
