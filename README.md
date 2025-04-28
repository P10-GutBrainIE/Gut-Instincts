# Gut-Instincts
This repo contains the submission of the group **Gut-Instincts** in the [GutBrainIE CLEF 2025 challenge](https://hereditary.dei.unipd.it/challenges/gutbrainie/2025/), part of the BioASQ CLEF Lab 2025. The challenge focuses on extracting structured information from biomedical abstracts related to the gut microbiota and its connections with Parkinson's disease and mental health. The goal is to develop Information Extraction (IE) systems to support experts in understanding the gut-brain interplay.

The challenge is divided into two main subtasks:
1. Named Entity Recognition (NER): Identifying and classifying specific text spans into predefined categories.
2. Relation Extraction (RE): Determining if a particular relationship between two categories holds.

## Table of Contents
- [Reproducibility](#reproducibility)
  - [Notes](#notes)
- [Setup](#setup)
  - [Virtual Environment](#virtual-environment)
  - [Installing Dependencies](#installing-dependencies)
- [License](#license)

## Reproducibility
To reproduce our results, follow these steps:
1. **Download the Data:** 

The official challenge data is not included in this repository. Download the data and place the data in the `data/` directory, preserving the original folder structure.

2. **Prepare the Environment**

Follow the guide in [Setup](#setup) to create an environment, activate the envirnoment, and install all dependencies.

3. **Train the Models**

Run the training for the NER model:
```bash

```
Run the training for the RE model:
```bash

```

4. **Evaluate the Models**
After training has finished, run the evaluation:
```bash

```

5. **Reproduce Submission Files**
To generate prediction files in the format required by the challenge for final submission, use:
```bash

```
Check the `submission_results/` directory for the output files.

### Notes
- All training was conducted on a computational cluster with GPU resources. Training on local machines may take significantly longer or may not be feasible depending on hardware.
- If you encounter issues with missing packages, ensure your environment matches the versions specified in `pyproject.toml`.

## Setup

### Virtual Environment

It is recommended to use a virtual environment to avoid dependency conflicts.

**Windows:**
```bash
python -m venv env
env\Scripts\activate
```

**Linux/MacOS:**
```bash
python3 -m venv env
source env/bin/activate
```

To deactivate the environment:
```bash
deactivate
```

### Installing Dependencies

Install the necessary dependencies as specified in `pyproject.toml`:
```bash
pip install -e .
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
