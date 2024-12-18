# DifFlexMM

![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue?logo=python&logoColor=ecf0f1&labelColor=34495e)
[![Python tests](https://github.com/bertoldi-collab/DifFlexMM/actions/workflows/python_tests.yml/badge.svg)](https://github.com/bertoldi-collab/DifFlexMM/actions/workflows/python_tests.yml)
[![Paper](https://img.shields.io/badge/Paper-10.1038/s41563--024--02008--6-blue?logoColor=ecf0f1&labelColor=34495e)](https://doi.org/10.1038/s41563-024-02008-6)
[![DOI](https://img.shields.io/badge/Data-10.5281/zenodo.12823471-blue?logo=zenodo&logoColor=ecf0f1&labelColor=34495e)](https://doi.org/10.5281/zenodo.12823471)
[![arXiv](https://img.shields.io/badge/arXiv-2403.08078-b31b1b?logo=arXiv&logoColor=arXiv&labelColor=34495e)](https://arxiv.org/abs/2403.08078)
[![GitHub license](https://img.shields.io/github/license/bertoldi-collab/DifFlexMM?labelColor=34495e)](https://github.com/bertoldi-collab/DifFlexMM/blob/main/LICENSE)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fbertoldi-collab%2FDifFlexMM&count_bg=%2327AE60&title_bg=%2334495E&icon=github.svg&icon_color=%23E7E7E7&title=Hits&edge_flat=false)](https://hits.seeyoufarm.com)

**Dif**ferentiable **Flex**ible **M**echanical **M**etamaterials

https://github.com/user-attachments/assets/8e966e96-13d4-439b-a0cb-d7dff115e7e1

## 🌅 Why DifFlexMM?

🚀 The goal of this framework is to push the boundary of designable dynamic behaviors achievable by nonlinear mechanical metamaterials.

🤖 Through the power of differentiable simulations, the framework removes the typical limitations of metamaterial design e.g. periodicity, intuition-based design, manual tuning, single-task design, etc., and enables the automated discovery of non-periodic, multi-functional, and reprogrammable metamaterials in the nonlinear dynamic regime.

## 🚁 Overview

This repository provides a fully differentiable dynamic design framework for 2D flexible mechanical metamaterials i.e. a network of rigid units connected by flexible ligaments.
The main physical ingredients of the model are:

- 🥌 Rigid-body kinematics of the units.
- 🎈 Elastic ligaments modeled by customizable energy functions.
- 💥 Energy-based contact interactions between rigid units.

With these ingredients, flexible mechanical metamaterials define a rich space of nonlinear dynamic behaviors that can be navigated by the framework.

🔭 From a high-level perspective, the framework facilitates the construction of the mapping between design parameters and the desired behavior of the metamaterial system.

![Code mapping](docs/code_mapping.svg)

By leverging [JAX](https://github.com/google/jax), this complex mapping is implemented in a differentiable fashion, thus allowing gradients to flow through the entire dynamic simulation.
In particular, differentiability is provided with respect to:

- Geometric paramaters: arbitrary parametrizations can be defined in the [`geometry`](difflexmm/geometry.py) module.
- Ligament paramaters: energy functions can be defined in the [`energy`](difflexmm/energy.py) module.
- Damping parameters: linear viscous damping as defined in the [`loading`](difflexmm/loading.py) module.
- Driving parameters: arbitrary driving functions can be applied to any degree of freedom.
- Loading parameters: arbitrary loading functions can be applied to any degree of freedom.
- Initial conditions: initial positions and velocities of the system.
- and any other paramater present in the [`ControlParams`](difflexmm/utils.py#L145-L163) data structure.

The main entry point of the simulator is the [`setup_dynamic_solver(...)`](difflexmm/dynamics.py#L60) in the [`dynamics`](difflexmm/dynamics.py) module.
This function takes all the fixed mappings (geometry, energy, loading, etc.) and returns a differentiable function that simulates the metamaterial dynamics.
Arbitrary forward problems can be defined by chaining this simulator function with any desired objective function.

## 📜 Paper

This repository contains all the code developed for the paper:

> [G. Bordiga, E. Medina, S. Jafarzadeh, C. Boesch, R. P. Adams, V. Tournat, K. Bertoldi. Automated discovery of reprogrammable nonlinear dynamic metamaterials. _Nature Materials_. (2024).](https://doi.org/10.1038/s41563-024-02008-6)

## 🎯 Solved design problems

The framework has been used to design a variety of mechanical metamaterials with different functionalities.

|  | Task/tasks description | Notebooks | Data 💾 | Video |
| --- | --- | --- | --- | --- |
| 🌟 | Focusing energy at a single target location | [Quads](notebooks/quads_focusing_3dp_pla_shims.ipynb), [Kagome](notebooks/kagome_focusing_3dp_pla_shims.ipynb) | [Quads](data/quads_focusing_3dp_pla_shims), [Kagome](data/kagome_focusing_3dp_pla_shims) | [Quads](https://github.com/bertoldi-collab/DifFlexMM/assets/16863374/ff76f0bc-463d-49c4-83bb-278f301af246), [Kagome](https://github.com/bertoldi-collab/DifFlexMM/assets/16863374/537a6e32-c62d-4fdc-8a9d-e4762fda8a21) |
| ️🗡️ | Splitting energy between different target locations | [Quads](notebooks/quads_energy_splitting_3dp_pla_shims.ipynb) | [Quads](data/quads_energy_splitting_3dp_pla_shims) | [Quads](https://github.com/user-attachments/assets/7b8ea46b-ccde-4f3f-8c04-e4f9015d62e5) |
| ✨ | Focusing multiple inputs at the same target location | [Quads](notebooks/quads_focusing_multi_input_3dp_pla_shims.ipynb) | [Quads](data/quads_focusing_multi_input_3dp_pla_shims) | [Quads](https://github.com/bertoldi-collab/DifFlexMM/assets/16863374/fda885c3-ffd6-4b67-a19e-ad59d5f52a96) |
| ️💫 | Reprogramming focusing target via static pre-compression | [Quads](notebooks/quads_focusing_switching_static_tuning_3dp_pla_shims.ipynb) | [Quads](data/quads_focusing_switching_static_tuning_3dp_pla_shims) | [Quads](https://github.com/bertoldi-collab/DifFlexMM/assets/16863374/5fa5cd61-f7dc-44b4-824c-6929818e7755) |
| 🌟🛡️ | Switching between focusing and protection task | [Quads](notebooks/quads_focusing_vs_protection_static_tuning_3dp_pla_shims.ipynb) | [Quads](data/quads_focusing_vs_protection_static_tuning_3dp_pla_shims) | [Quads](https://github.com/bertoldi-collab/DifFlexMM/assets/16863374/ad4f9811-e623-4867-af42-c36ee31bcfbb) |
| 🌀 | Nonlinear motion conversion | [Quads](notebooks/quads_spin_3dp_pla_shims.ipynb) | [Quads](data/quads_spin_3dp_pla_shims) | [Quads](https://github.com/bertoldi-collab/DifFlexMM/assets/16863374/9aa2bbc9-cbe0-4896-8c83-ce67f2c61af3) |

💾 All data generated or used for the paper can be downloaded from [![DOI](https://img.shields.io/badge/Data-10.5281/zenodo.12823471-blue?logo=zenodo&logoColor=ecf0f1&labelColor=34495e)](https://doi.org/10.5281/zenodo.12823471).
To access and visualize the data:

- Extract `data.zip` in the root directory of the repository.
- Load and visualize optimization results for all the solved problems using the provided [notebooks](notebooks).
- Load and visualize experimental data using the notebooks in the [exp](exp) folder.

## ⬇️ Installation

### DifFlexMM only

Assuming you have access to the repo and ssh keys are set up in your GitHub account, you can install the package with

```bash
pip install git+ssh://git@github.com/bertoldi-collab/DifFlexMM.git
```

### DifFlexMM with examples of solved design problems

Clone the repository, `cd` into the `DifFlexMM` folder, and install with

```bash
pip install -e .
```

## 🤝 Contributing

<details>
<summary><b>Expand here</b></summary>

The dependency management of the project is done via [poetry](https://python-poetry.org/docs/).

To get started:

- Install [poetry](https://python-poetry.org/docs/).
- Clone the repository.
- `cd` into the root directory and run `poetry install`. This will create the poetry environment with all the necessary dependencies.
- If you are using vscode, search for `venv path` in the settings and paste `~/.cache/pypoetry/virtualenvs` in the `venv path` field. Then select the poetry enviroment as python enviroment for the project.

</details>

## 📝 Citation

If you use this code in your research or anywhere, please cite the paper:

```bibtex
@article{bordiga_2024,
    title   = {Automated Discovery of Reprogrammable Nonlinear Dynamic Metamaterials},
    author  = {Bordiga, Giovanni and Medina, Eder and Jafarzadeh, Sina and B{\"o}sch, Cyrill and Adams, Ryan P. and Tournat, Vincent and Bertoldi, Katia},
    year    = {2024},
    journal = {Nature Materials},
    volume  = {23},
    number  = {11},
    pages   = {1486--1494},
    issn    = {1476-4660},
    doi     = {10.1038/s41563-024-02008-6},
}
```
