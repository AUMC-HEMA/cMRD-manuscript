# Code from: "Computational measurable residual disease assessment in acute myeloid leukemia using mixture models"

## Scripts

This repository uses both R (flow cytometry pre-processing) and Python (modeling) together. The code is structured as follows:

|            Script           |                                   Function                                  |
|:---------------------------:|:---------------------------------------------------------------------------:|
|      01-preprocess.Rmd      | Pre-processing of flow cytometry data (BLAST110, LAIP29 and RBM18 datasets) |
|       02-GMMclf.ipynb       |                     Blast prediction models + benchmark                     |
|         03-WBC.ipynb        |                            WBC prediction models                            |
|       04a-refGMM.ipynb      |                         Reference GMM model training                        |
| 04b-refGMM-annotation.ipynb |  Generation of model component CSV file for FCS annotation                  |
|   04c-refGMM-labeling.Rmd   |             Construction of annotated FCS files                             |
|      05-benchmark.ipynb     |                     Benchmark of cMRD methods                               |
|       06a-timing-R.Rmd      |                             Timing of R scripts                             |
|   06b-timing-Python.ipynb   |                           Timing of Python scripts                          |

## Data

Datasets are available from Zenodo: https://zenodo.org/records/11046402
