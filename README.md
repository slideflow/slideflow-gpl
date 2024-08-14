![slideflow logo](https://github.com/jamesdolezal/slideflow/raw/master/docs-source/pytorch_sphinx_theme/images/slideflow-banner.png)

[![Python application](https://github.com/slideflow/slideflow-gpl/actions/workflows/python-app.yml/badge.svg?branch=master)](https://github.com/slideflow/slideflow-gpl/actions/workflows/python-app.yml)
[![PyPI version](https://badge.fury.io/py/slideflow-gpl.svg)](https://badge.fury.io/py/slideflow-gpl)
| [ArXiv](https://arxiv.org/abs/2304.04142) | [Docs](https://slideflow.dev) | [Cite](#reference)


**Slideflow-GPL brings additional digital pathology deep learning tools to Slideflow, under the GPL-3 license.**

Slideflow is designed to provide an accessible, easy-to-use interface for developing state-of-the-art pathology models. While the core Slideflow package integrates with a wide range of cutting-edge methods and models, the variability in licensing practices necessitates that some functionality is distributed through separate add-on packages. **Slideflow-GPL** extends Slideflow with additional tools available under the GPL-3 license, ensuring that the core package remains as open and permissive as possible.

## Requirements
- Python >= 3.8
- [Slideflow](https://github.com/jamesdolezal/slideflow) >= 3.0
- [PyTorch](https://pytorch.org/) >= 1.9

## Installation
Slideflow-GPL is easily installed via PyPI and will automatically integrate with Slideflow.

```
pip install slideflow-gpl
```

## Features
- **RetCCL**, a pretrained feature extractor ([GitHub](https://github.com/Xiyue-Wang/RetCCL) | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S1361841522002730))
- **CTransPath**, a pretrained feature extractor ([GitHub](https://github.com/Xiyue-Wang/TransPath) | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S1361841522002043))
- **CLAM**, a multiple-instance learning (MIL) model architecture ([GitHub](https://github.com/mahmoodlab/CLAM) | [Paper](https://www.nature.com/articles/s41551-020-00682-w))

#### RetCCL & CTransPath

The RetCCL and CTransPath feature extractors are accessible using the [same interface](https://slideflow.dev/mil/#generating-features) all pretrained extractors utilize in Slideflow.

```python
import slideflow as sf

retccl = sf.build_feature_extractor('retccl')
```

Please see the [Slideflow documentation](https://slideflow.dev/mil/#generating-features) for additional information on how feature extractors can be deployed and used. 

#### CLAM

The CLAM architectures, `CLAM_SB`, `CLAM_SB`, `MIL_fc`, and `MIL_fc_mc` will be automatically available upon installation, and can be specified using the same `mil_config()` interface used for other MIL models in Slideflow.

```python
import slideflow as sf
import slideflow.mil

config = sf.mil.mil_config('clam_mb', epochs=20, lr=1e-4)
```

Please see the [Slideflow docs](https://slideflow.dev/mil/) for more information on MIL models.

#### CLAM - Legacy Trainer

The legacy CLAM trainer, which has been superseded by Slideflow's FastAI trainer, can still be used if desired. To use this trainer instead of the default FastAI framework, set the argument `trainer='legacy_clam'`:

```python
config = sf.mil.mil_config('clam_mb', trainer='legacy_clam', ...)
```

## License
This code is made available under the GPLv3 License.

All three features made available in this repository - RetCCL, CTransPath, and CLAM - are licensed under GPLv3. However, please be aware that authors have stated their intent for these models to be used for non-commercial, academic purposes ([1](https://github.com/Xiyue-Wang/RetCCL), [2](https://github.com/Xiyue-Wang/TransPath), [3](https://github.com/mahmoodlab/CLAM)). 

## Reference
If you find our work useful for your research, or if you use parts of this code, please consider citing as follows:

Dolezal, J.M., Kochanny, S., Dyer, E. et al. Slideflow: deep learning for digital histopathology with real-time whole-slide visualization. BMC Bioinformatics 25, 134 (2024). https://doi.org/10.1186/s12859-024-05758-x

```
@Article{Dolezal2024,
    author={Dolezal, James M. and Kochanny, Sara and Dyer, Emma and Ramesh, Siddhi and Srisuwananukorn, Andrew and Sacco, Matteo and Howard, Frederick M. and Li, Anran and Mohan, Prajval and Pearson, Alexander T.},
    title={Slideflow: deep learning for digital histopathology with real-time whole-slide visualization},
    journal={BMC Bioinformatics},
    year={2024},
    month={Mar},
    day={27},
    volume={25},
    number={1},
    pages={134},
    doi={10.1186/s12859-024-05758-x},
    url={https://doi.org/10.1186/s12859-024-05758-x}
}
```
