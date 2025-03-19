# Gut-Instincts
This repo contains the submission of the group Gut-Instincts in the [GutBrainIE CLEF 2025 challenge](https://hereditary.dei.unipd.it/challenges/gutbrainie/2025/), part of the BioASQ CLEF Lab 2025. The challenge focuses on extracting structured information from biomedical abstracts related to the gut microbiota and its connections with Parkinson's disease and mental health. The goal is to develop Information Extraction (IE) systems to support experts in understanding the gut-brain interplay.

The challenge is divided into two main subtasks:
1. Named Entity Recognition (NER): Identifying and classifying specific text spans into predefined categories.
2. Relation Extraction (RE): Determining if a particular relationship between two categories holds.

The training data includes highly curated datasets manually annotated by experts and students, as well as automatically generated annotations. The test set consists of titles and abstracts from the gold and platinum collections.

## Table of Contents
- [Setup](#setup)
  - [Virtual Environment](#virtual-environment)
  - [Installing Dependencies](#installing-dependencies)
  - [Code Formatting](#code-formatting)
- [Structure](#structure)
- [License](#license)

## Setup
### Virtual Environment
You can optionally create a virtual environment before installing any dependencies. This helps keep the project dependencies isolated and avoids conflicts with other projects.

On Windows, to create the virtual environment, run:
```
python -m venv env
```
On Linux:
```
python3 -m venv env
```
On Windows, use the following command to activate the environment:
```
env\Scripts\activate
```
On Linux:
```
source env/bin/activate
```
To deactivate the environment, use the following command:
```
deactivate
```
### Installing Dependencies
To install the necessary dependencies for the project, run:
```
pip install -r requirements.txt
```
### Code Formatting
The project is set up to use Ruff as the formatter and linter. Install the Ruff extension in VS Code, and use the shorcut `Alt + Shift + F` to run the formatter in a specfiic file.

## Structure

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
