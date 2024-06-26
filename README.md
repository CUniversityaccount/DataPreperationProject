# Data Corruption Augmentation and Cleaning, Data prepartion
This is the data preperation project for the course Data Preperation 2023
To generate erroneous in the local notebooks, use the local notebooks defined in the example and analyse directory.

## Setup
### Generating erroneous data
To install the necassary tools make an local environment in the repository through
```bash
python -m venv ./venv
```

Activitate the local environment following the instructions defined by the python organization (https://docs.python.org/3/library/venv.html)

After activating the local environment, install the dependencies for the erroneous augmentation of data through the following command:
```bash
pip install -r requirements
```

### Data Cleaning
For the data cleaning part use conda as package manager and do the following commands:
```bash
conda create -n DP python=3.9 -y
conda install -n DP ipykernel --update-deps --force-reinstall -y
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### Analysis
For the analysis part, use this command to install the model package:
```
python -m spacy download en_core_web_md
```
