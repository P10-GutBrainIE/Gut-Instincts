# nlp_tutorials
## Setup
### Virtual Environment
You can optionally create a virtual environment (https://bitly.cx/VHGf4)  before installing any dependencies. This helps keep the project dependencies isolated and avoids conflicts with other projects. To create the virtual environment, run:
```
python -m venv env
```
On Windows, use the following command to activate the environment:
```
env\Scripts\activate
```
To deactivate the environment, use the following command:
```
deactivate
```
### Installing Dependencies
To install the necessary dependencies for the project (as specified in pyproject.toml), run:
```
pip install .
```
### Code Formatting
The project is set up to use Ruff as the formatter. Install the Ruff extension in VS Code, and use the shorcut `Alt + Shift + F` to run the formatter in a specfiic file.
