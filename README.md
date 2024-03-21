# NeuralNetwork

## Table of Content
1. [Summary](#Summary)
2. [Setting-Up](#Setting-Up)
   1. [Pyenv](#Pyenv)
   2. [Pipenv](#Pipenv)

## Summary
In this repository, we implement an infrastructure for training neural networks.

## Setting-Up
In this project, we use [Pyenv](https://github.com/pyenv/pyenv) for controlling Python versions, and 
[Pipenv](https://github.com/pypa/pipenv) as a package manager. You can see how to install and use these 2 tools below.
### Pyenv
To install pyenv, first we need to assign dependencies. We use the following command for this:
```shell
$ sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl
```
Then we define an environment variable to specify the desired pyenv version:
```shell
export PYENV_GIT_TAG=v2.3.16
```
Now it's time to install pyenv:
```shell
curl https://pyenv.run | bash
```
Then, we add pyenv to the load path with the following commands:
```shell
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
echo 'eval "$(pyenv init -)"' >> ~/.profile
```
Then, you need to reload your shell:
```shell
bash
```
The current python version is mentioned in Pipfile in python_full_version attribute.
With the following commands, the current python version for the project will be installed and used:
```shell
pyenv install <python-full-version>
pyenv global <python-full-version>
```
### Pipenv
First, we install pipenv:
```shell
pip install pipenv==2023.10.24
```
Then we navigate to the root of the project and run the following command to create the project's environment:
```shell
pipenv shell
```
To install packages, we need to run the following command in the project's root directory. 
```shell
pipenv install
```
Additionally, to add a new package to the project, we should run the following command:
```shell
pipenv install <package-name>==<package-version>
```

