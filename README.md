### Steps to recreate environment ###
1. Ensure poetry is installed on your machine ([instructions](https://python-poetry.org/docs/#installing-with-pipx)).
2. Ensure CUDA is installed on your machine. See torch version in pyproject.toml for CUDA version requirement.
3. Install dependencies using `poetry install -vv`.
4. Activate the environment using `poetry shell`. \
(If using zsh, you may need to apply
[this workaround](https://github.com/python-poetry/poetry-plugin-shell/issues/9) to 
activate the environment)