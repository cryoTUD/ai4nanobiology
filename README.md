# AI4Nanobiology
![NB4170 banner](./docs/NB4170_banner.png)

## Welcome to the AI44Nanobiology course! 

Here you will find the Jupyter Notebooks used in the course NB4170 – AI4Nanobiology. 

## How to use 

1) Local computer 
There are several options that you can use to access the notebooks and run them. The file environment.yml contains the dependencies that you need to install to run the notebooks in your local machine. If you have conda installed, you can create a new environment with the following command: 

```bash
conda env create -f environment.yml
```
This will create a new environment called `ai4bio` which you can use to run the notebooks. 

2) Binder
Alternately, you can also use Binder to run the notebooks without installing anything. Simply click the badge below to launch an instance of JupyterLab in your browser, where you can access and run the notebooks.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cryoTUD/ai4nanobiology/HEAD)

Note that if you choose to use Binder, you do have to save the notebooks to your computer to keep the changes that you made. Binder will not save them automatically. So if you wish to keep the changes that you made, please download them before you close the Binder session. 

3) DelftBlue
We have also set up a DelftBlue environment for the course. If you are enrolled in the course on Brightspace, you will get acccess to the DelftBlue environment.  

[Open OnDemand](https://login.delftblue.tudelft.nl) is probably the easiest way to access DelftBlue, but you may have to wait in a queue to get access to the environment. 

Using the Cluster Shell Access (Clusters > DelftBlue Cluster Shell Access)
- Setup the environment: 
```bash
source /projects/nb4170/setup_nb4170.sh
```

Then go to Interactive Apps > JupyterLab and start a new session. Set the runtime to 2 hours. Under additional module type 
```bash
cuda, miniforge3
```
For container file, select PyTorch

Then click Launch. This will take a few minutes to start. Once it starts, you can run Jupyter Notebooks from the course directory. 

