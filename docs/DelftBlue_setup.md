# AI4Nanobiology: Setting things up in DelftBlue
## Part 1: Getting access

#### 1. You can request access for all students by raising a topdesk call [here](https://tudelft.topdesk.net/tas/public/ssp/content/serviceflow?unid=58cebc4f3e9b4e67b87ddd2b54bf1666)
- Purpose:
  
  ```bash
  Students to access DelftBlue for the duration of a master course (NB4170) in Q4
  ```
- Resources:

  ```bash
  Access to gpu-a100-small (or gpu-a100) and compute nodes. Compute nodes used for interactive jupyter notebook execution. GPUs for training a neural network using python scripts
  ```
  
- Student list: Export from Brightspace. Go to Course Admin > Classlist > Select all > Export as CSV
Upload the student list along with the request. All the students who are in the list will automatically get access to DelftBlue. This will create a new account ```education-as-courses-nb4170``` that can be used to access DelftBlue and submit jobs. 

> [!TIP]
> Upload the student list atleast one week before the start of the course. It typically takes 2-3 business days for admins to process the request and add everyone to the group. Mention that in the course description so students do not enroll at the last moment. If that still happens (expect that) then you can send another list with the new students added to it in the same request. Plan week 1 so that it can be run on Binder.

> [!CAUTION]
> External students will need to go through some extra step (same holds true for instructors, and TA's from outside TU Delft). The following additional steps are required to ensure they can also access DelftBlue

 <details>
<summary>Request for external users to access DelftBlue</summary>
   
- In 2026, we had to raise generic topdesk [request](https://tudelft.topdesk.net/tas/public/ssp/content/serviceflow?unid=d4bd7e8e403b44feb221daa17d1aad61) with the following text: 
   
```bash
Please add NetID xxxxxxx to AD group Apps-OpenVPN-UD-HPCGuest
```

There maybe [new ways](https://tudelft.topdesk.net/tas/public/ssp/content/serviceflow?unid=1ce40a77-96d2-4bca-87ab-f0bcf694c5e7) to request it. So please check with DelftBlue admins about that. 

- External users install OpenVPN on their device. Instructions to install OpenVPN for [Windows](https://filelist.tudelft.nl/ICT%20Handleidingen/3%20-%20Content%20OUD%20%28studentenportal%29/Handleidingen/Open%20VPN/OpenVPN_Externals_Windows1.pdf) and [Mac](https://filelist.tudelft.nl/ICT%20Handleidingen/3%20-%20Content%20OUD%20%28studentenportal%29/Handleidingen/Open%20VPN/OpenVPN_Externals_Mac2.pdf) are given. 

- Users need to connect through OpenVPN and not eduVPN. Disconnect any other VPN that maybe running. Note that this seems to be required even when connecting through eduroam. 
</details>

#### 2. Access DelftBlue through [Open OnDemand](https://login.delftblue.tudelft.nl/). Instructions for open the cluster shell access and JupyterLab can be found [here](./accessing_delftblue_instructions.pdf).

<details>
  <summary>Accessing DelftBlue Desktop</summary>
  <img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/0103d49c-f052-47af-b612-3e2d833d900c" />
</details>

#### 3. Submit jobs 
For student projects, students submit SLURM scripts to make use of GPUs. They have access to all partition. Open OnDemand has web interface to submit jobs through web browser. You can find that [here](./submit_jobs_delftblue_openondemand.pdf). 

## Part 2: Setting up DelftBlue environments
<details>
 <summary>Folder structure</summary>
<img width="800" height="250" alt="image" src="https://github.com/user-attachments/assets/223c8a42-da17-412c-af91-8ef5290df183" />
</details>

- Shared project directory at: ```/projects/nb4170/``` which contained setup file, conda environments, code and data. Students have read access to this folder.
- Create a shared conda environment that can be accessed by all students. Use the [ai4bio](../environment.yml) environment file from the repo. Navigate to the folder containing the environment file and run
  ```bash
  module load miniforge3
  conda env create -f /path/to/environment.yml -p /projects/nb4170/envs/ai4bio
  ```

- Append the shared environment folder to the conda config so it can be accessed from anywhere. 
  ```bash
  conda config --append envs_dirs /projects/nb4170/envs
  ```
  This edits your ```~/.condarc``` file. You only need to do it once. For students, it is convenient if this line is added to the setup script.

- Register your environment using ```ipykernel``` so that it remains accessible by JupypterLab through compute/gpu nodes. ```ai4bio``` environment already has this package installed, but if you create a new environment then you need to include this package in it and run the following command using the new environment to access that. 

  ```bash
  python -m ipykernel install --user --name ai4bio --display-name "Python (ai4bio)"
  ```
> [!TIP]
> Students requested having a larger shared folder for their group projects. So in future it maybe nicer to have another shared folder: ```/projects/nb4170_projects``` only for student projects. You can have higher storage space for this, upto 1 TB.
  
#### Setup file 

We had one setup file stored at: ```/projects/nb4170/setup_nb4170.sh```. The setup file can be very useful to organise environment, code, and data. For the class of 2026, we let students copy each week's course material from the shared project directory. They ran the setup file at the start of each class through *Cluster Shell Access* on Open OnDemand web interface. This is a very flexible approach and allows you to adapt the course content each week. 

> [!IMPORTANT]
> As of 2026, the compute nodes do not have internet access. So if your course material requires connection to the internet you need to first download it yourself and store it at the shared folder. The setup file can be used to copy that material to the student's home directory.

