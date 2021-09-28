# Femur Segmentation

A simple pipeline for the segmentation of femur CT scan based on Graph Cut.

| **Authors**  | **Project** |  **Build Status** | **License** | **Code Quality** |
|:------------:|:-----------:|:-----------------:|:-----------:|:----------------:|
| [**R.Biondi**](https://github.com/RiccardoBiondi) <br/> [**D. Dall'Olio**](https://github.com/DanieleDallOlio)| **FemurSegmentation** | [![Ubuntu CI](https://github.com/RiccardoBiondi/FemurSegmentation/workflows/Ubuntu%20CI/badge.svg)](https://github.com/RiccardoBiondi/FemurSegmentation/actions/workflows/ubuntu.yml) <br/> [![Windows CI](https://github.com/RiccardoBiondi/FemurSegmentation/workflows/Windows%20CI/badge.svg)](https://github.com/RiccardoBiondi/FemurSegmentation/actions/workflows/windows.yml) |[![license](https://img.shields.io/github/license/mashape/apistatus.svg)]()|**Codacy** [![Codacy Badge](https://app.codacy.com/project/badge/Grade/f07936f011b64e95b5e2bdcd7b8bc61f)](https://www.codacy.com/gh/RiccardoBiondi/FemurSegmentation/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=RiccardoBiondi/FemurSegmentation&amp;utm_campaign=Badge_Grade) <br/> **Codebeat** [![codebeat badge](https://codebeat.co/badges/a0131d28-4075-456c-9b69-7f3bac3f3d42)](https://codebeat.co/projects/github-com-riccardobiondi-femursegmentation-master)|

**Appveyor** [![Build status](https://ci.appveyor.com/api/projects/status/o4yt5atxpnje9c9i?svg=true)](https://ci.appveyor.com/project/RiccardoBiondi/femursegmentation)

## Table of Contents

  1. [Introduction](#Introdouction)
  2. [Usage](#Usage)
  3. [Contribute](#Contribute)
  4. [License](#Lincense)
  5. [Authors](#Authors)
  6. [Acknowledgments](#Acknowledgments)
  7. [Citation](#Citation)
  8. [References](#Refereces)

## Introduction

## Prerequisites

Supported python versions: ![Python version](https://img.shields.io/badge/python-3.6.*|3.7.*|3.8.*|3.9.*-blue.svg)

Supported c++ compilers: ![g++ compiler](https://img.shields.io/badge/g++-4.8|4.9|5.*|6.*|7.*|8.*|9.*|10.*-orange.svg)
![clang compiler](https://img.shields.io/badge/clang-7.*|8.*|9.*-red.svg)
![MinGW compiler](https://img.shields.io/badge/MinGW-3.*|4.*-green.svg)

### Installation

Clone the repository:

```console
git clone https://github.com/RiccardoBiondi/FemurSegmentation
cd FemurSegmentation
```

If you are using conda, create and activate the environment:

```console
conda env create -f itk_env.yaml
conda env activate itk
```

Or, if you are using `pip`, install the required packages:

```console
python -m pip install -r requirements.txt
```

Now you are ready to build the package:

```console
python setup.py develop --user
```



### Testing

We have provide a test routine in [test](./test) directory. This routine use:
  - pytest >= 3.0.7

  - hypothesis >= 4.13.0

Please install these packages to perform the test.
You can run the full set of test with:

```console
  python -m pytest
```


## Getting Started

Now we will see how to perform the automated and semi-automated segmentation.

Make sure to add `/FemurSegmentation/lib` to your python library before running.

On Ubuntu like os:
```console
export PYTHONPATH=$PYTHONPATH:~/FemurSegmentation/lib/
```

or for windows users(from PowerShell):
```console
$env:PYTHONPATH="C:\path\to\FemurSegmentation\lib\"
```

### Prepare Data

This script will process one leg at a time. Firstly we have to split the whole CT scan into left and right leg. *split_image* script will perform this step:

```console
  python split_image.py --input='path/to/input/image' --output='./output/file/name'
```

You can also provide the ground truth segmentation to format the labels for each leg:

```console
  python split_image.py --input='/path/to/input/image' --output='/path/to/output/file/name' --label='/path/to/image/labels'
```

### Automated Segmentation

Now you can segment one leg at time by running:

```console
python run_automated_segmentation.py --input='/path/to/input/file/ --output='/path/to/output/file.nrrd'
```

### Semi-Automated Segmentation

To run the semi automated segmentation, you have to provide the hard constrains.
To do so, you have to manually label six or seven slices of the whole scan.

Label Legend:
  - 0: No label
  - 1: Femur
  - 2: Background (Soft tissues, non-femur bones)

**notes** Be careful on the hip-joint region

To achieve this purpose, you can use 3DSlicer sofrtware.

**note** The hard constrain mask must have the same size of the original image.

Now you are ready to perform the segmentation:

```console
python run_semiautomated_segmentation.py --input='/path/to/input/file/ --output='/path/to/output/file.nrrd' --init='/path/to/init/hard/constrains'
```

## Contribute

Any contribution is more than welcome. Just fill an [issue](./.github/ISSUE_TEMPLATE.md) or a [pull request](./.github/PULL_REQUEST_TEMPLATE.md) and we will check ASAP!

See [here](https://github.com/RiccardoBiondi/FemurSegmentation/blob/master/CONTRIBUTING.md) for further informations about how to contribute with this project.


## License

## Authors

* **Riccardo Biondi** [git](https://github.com/RiccardoBiondi), [unibo](https://www.unibo.it/sitoweb/riccardo.biondi7)

* **Daniele Dall'Olio** [git](https://github.com/DanieleDallOlio), [unibo](https://www.unibo.it/sitoweb/daniele.dallolio)

* **Nico Curti** [git](https://github.com/Nico-Curti), [unibo](https://www.unibo.it/sitoweb/nico.curti2)

* **Gastone Castellni** [unibo](https://www.unibo.it/sitoweb/gastone.castellani)

## Acknowledgments



## Citation

```BibTeX
@misc{FemurSegmentation,
  author = {Biondi, Riccardo and Dall'Olio, Daniele and Curti, Nico and Castellani, Gastone},
  title = {A graph Cut Approach for Femur Segmentation},
  year = {2021},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/RiccardoBiondi/FemurSegmentation}},
}
```


## References

<a id="1">[1]</a>
Krčah, M., Székely, G., Blanc, R.
Fully automatic and fast segmentation of the femur bone from 3D-CT images with no shape prior
2011 IEEE International Symposium on Biomedical Imaging: From Nano to Macro, Chicago, IL, 2011, pp. 2087-2090. [doi](https://doi.org/10.1109/ISBI.2011.5872823)


<a id="2">[2]</a>
Bryce A. Besler, Andrew S. Michalski, Michael T. Kuczynski, Aleena Abid, Nils D. Forkert, Steven K. Boyd
Bone and joint enhancement filtering: Application to proximal femur segmentation from uncalibrated computed tomography datasets,
Medical Image Analysis
2021 Medical Image Analysis, Volume 67 ISSN 1361-8415. [doi](https://doi.org/10.1016/j.media.2020.101887)
