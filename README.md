# from-bytes-to-borsh
Artifacts and workflows for "From Bytes to Borsch: Fine-Tuning Gemma and Mistral for Ukrainian Language Representation" paper


### Prerequisites

To fully initialize the repo please run

```
git submodule update --init --recursive
```

### Folder structure

`unlp-2024-shared-task` contains the files from the UNLP competition. 

`fine-tuning` contains workflows and configs used to fine-tune LLMs to participate in "UNLP-2024" challenge.

`benchmarks` contains scripts and workflows on how we evaluated the fine-tuned LLMs.

`notebooks` contains python notebooks.

# Citation

```
@misc{frombytestoborsch,
      title={From Bytes to Borsch: Fine-Tuning Gemma and Mistral for the Ukrainian Language Representation}, 
      author={Artur Kiulian and Anton Polishko and Mykola Khandoga and Oryna Chubych and Jack Connor and Raghav Ravishankar and Adarsh Shirawalmath},
      year={2024},
      eprint={2404.09138},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2404.09138}, 
}
```
