# In china
use
```
export HF_ENDPOINT=https://hf-mirror.com/
```
then login

# Python
```
python -m venv .venv

# activate
source .env/bin/activate
# deactivate
deactivate

# install dependency
python3 -m pip install transformers datasets evaluate accelerate
python3 -m pip install torch
```

# Conda
conda env create --file hf-env.yml
conda env update –file hf-env.yml

conda my_env export > environment.yml
conda env create -f environment.yml
conda activate my_env

Install dependencies:
conda config --append channels conda-forge
conda install -n hf-env gradio

conda list | grep <package>  seach package name


Install hf jupiter notebook dependencies:
conda install -c huggingface transformers -n hf-env

