### Fine-tuning workflow

Our team has incorporated the [axolotl](https://github.com/OpenAccess-AI-Collective/axolotl?tab=readme-ov-file#train) tool designed to streamline the fine-tuning process.

Given the docker with enabled CUDA runtime you can run the fine-tuning process like so

```
docker run --privileged --gpus "all" --shm-size 10g --rm -it --name axolotl --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd):/home/ -w /home/workspace winglian/axolotl:main-py3.10-cu118-2.0.1 accelerate launch -m axolotl.cli.train <mistral/gemma>.yaml
```

Please note: It is imperative to tailor the .yaml configuration files to align with your specific hardware setup and development environment, such as the WandB project settings.