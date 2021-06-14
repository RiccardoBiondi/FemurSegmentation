# Femur Segmentation

A simple pipeline for the segmentation of femur CT scan segmentation based on Graph Cut.

| **Authors**  | **Project** |  **Build Status** | **License** | **Code Quality** |
|:------------:|:-----------:|:-----------------:|:-----------:|:----------------:|
| [**R.Biondi**](https://github.com/RiccardoBiondi) | **FemurSegmentation** | **Windows**: **Linux**: [![Build Status](https://travis-ci.com/RiccardoBiondi/FemurSegmentation.svg?token=YRvqSXwHasrnEcL9EuWP&branch=master)](https://travis-ci.com/RiccardoBiondi/FemurSegmentation) |[![license](https://img.shields.io/github/license/mashape/apistatus.svg)]()|**Codacy**  **Codebeat** |

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

## Usage

Supported python versions: ![Python version](https://img.shields.io/badge/python-3.6.*|3.7.*|3.8.*-blue.svg)

Supported c++ compilers: ![g++ compiler](https://img.shields.io/badge/g++-7.*|8.*|9.*-orange.svg)
![clang compiler](https://img.shields.io/badge/clang-3.*|4.*-red.svg)
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

Any contribution is more than welcome. Just fill an [issue](./.github/ISSUE_TEMPLATE.md) or a [pull request](./.github/PULL_REQUEST_TEMPLATE.md) and we will check ASAP!

See [here](https://github.com/RiccardoBiondi/FemurSegmentation/blob/master/CONTRIBUTING.md) for further informations about how to contribute with this project.

### Segmentation

Make sure to add `/FemurSegmentation/lib` to your python library before running. On Ubuntu like os:
```console
export PYTHONPATH=$PYTHONPATH:~/FemurSegmentation/lib/
```

or for windows users:
```console
set PYTHONPATH=PYTHONPATH;"C:\path\to\FemurSegmentation\lib\"
```
To run the unsupervised segmentation:
```console
python segment_femur.py --input='/path/to/input/file/ --output='/path/to/output/file.nrrd'
```

where --input require a path to a file (like `filename.nrrd`) or to a folder containing a single DICOM series.

### Evaluation

## Contribute

## License

## Authors

* **Riccardo Biondi** [git](https://github.com/RiccardoBiondi)
* **Daniele Dall'Olio** [git](https://github.com/DanieleDallOlio), [unibo](https://www.unibo.it/sitoweb/daniele.dallolio)
* **Nico Curti** [git](https://github.com/Nico-Curti), [unibo](https://www.unibo.it/sitoweb/nico.curti2)

## Acknowledgments

## Citation

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
