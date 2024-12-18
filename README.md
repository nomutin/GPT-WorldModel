# GPT-Style World Model

![python](https://img.shields.io/badge/python-3.10-blue)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/keio-crl/GPT-WorldModel/actions/workflows/ci.yaml/badge.svg)](https://github.com/keio-crl/GPT-WorldModel/actions/workflows/ci.yaml)

## 🚀 Usage

### Installation

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

### Training

Use [lightningCLI](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html).

```bash
uv run scripts/cli.py fit \
    --config config/path_to_config.yaml \
    --trainer.devices [x,y]
```

## 📕 References

### VQ-VAE

- [NVIDIA/Cosmos-Tokenizer](https://github.com/NVIDIA/Cosmos-Tokenizer)

    ```bibtex
    @misc{CosmosTokenizer,
      author = {NVIDIA},
      title = {NVIDIA/Cosmos-Tokenizer},
      year = {2023},
      howpublished = {\url{https://github.com/NVIDIA/Cosmos-Tokenizer}},
    }
    ```

### World Model

- [VideoGPT: Video Generation using VQ-VAE and Transformers](https://wilsonyan.com/videogpt/index.html)

    ```bibtex
    @inproceedings{yan2023videogpt,
      title={VideoGPT: Video Generation using VQ-VAE and Transformers},
      author={Yan, Wilson and Li, Yuan and Zhang, Yu and Wang, Yizhou and Chen, Xilin and Liu, Ziwei and Lu, Jiwen and Zhou, Ming and Yang, Jian},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      pages={10014--10023},
      year={2023}
    }

- [Genie: Generative Interactive Environments](https://sites.google.com/view/genie-2024/home)

    ```bibtex
    @article{bruce2024genie,
      title={Genie: Generative Interactive Environments},
      author={Bruce, Jake and Dennis, Michael and Edwards, Ashley and Parker-Holder, Jack and Shi, Yuge and Hughes, Edward and Lai, Matthew and Mavalankar, Aditi and Steigerwald, Richie and Apps, Chris and others},
      journal={arXiv preprint arXiv:2402.15391},
      year={2024}
    }
    ```

- [1x-technologies/1xgpt](https://github.com/1x-technologies/1xgpt)

    ```bibtex
    @misc{1X_Technologies_1X_World_Model_2024,
        author = {{1X Technologies}},
        month = jun,
        title = {{1X World Model Challenge}},
        year = {2024}
    }
    ```
