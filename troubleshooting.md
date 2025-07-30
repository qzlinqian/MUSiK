# Troubleshooting
## Installation
**Be sure to use Python 3.10 to avoid dependency conflicts.**
### For submodule k-wave-python
##### Use as submodule

The k-wave-python package is a submodule here, so when you clone this repository, you can use

```shell
git clone --recursive --remote-submodules git@github.com:norway99/MUSiK.git
```
or if you have already cloned the MUSik repo, use this
```shell
git submodule update --init --recursive --remote
```
Remember to add the **remote** flag, because the k-wave package is updating frequently, and there are deprecated dependencies that may cause installation failure. 

Be sure to stay in **stable** versions (check git log and roll back to a commit with "**release**" tag).

```shell
git log # then find a stable commit id xxxx
git reset --hard xxxx # reset to the stable commit
```

##### Or pip install

Alternatively, to stick on stable release versions, you can just run pip install instead of using the submodule

```
pip install k-wave-python
```
## Running for the first time

**k-wave needs to download some binaries at the first run.** 

It may cause issue when you are using cloud computating resources whose computation nodes do not have internet access (like SuperCloud).
I used SuperCloud and there is a ["download" partition](https://mit-supercloud.github.io/supercloud-docs/using-the-download-partition/) with data transfer nodes.
So my strategy was to use this "download" partition and jupyter notebook in turn to download the binaries and target uninstalled dependencies.

## Running on old systems

I have migrated from SuperCloud to Engaging. Engaging's computation nodes have internet access, so you don't need to worry about the binaries, but it do have other issues due to its outdated system.

In short word, you have to use `singularity` to install a container and use it to run all the code. (See [Build Singularity Images](https://orcd-docs.mit.edu/software/apptainer/#build-singularity-images).) Also, keep k-wave-python as a submodule because it tries to modify the binaries everytime.

#### Create a container with singularity in --sandbox

You can use any other docker images you find useful ([DockerHub](https://hub.docker.com)).

```shell
singularity build --sandbox my-image docker://nvidia/cuda:12.9.1-cudnn-runtime-ubuntu22.04
```

#### Install your libraries in fakeroot shell

Be sure to log in a head node with `fakeroot`. In Engaging: 

```shell
ssh <user>@orcd-login004.mit.edu
```

Then start container shell with the flags `--writable --fakeroot` and install everything you need.

```shell
singularity shell --writable --fakeroot my-image
Apptainer> apt-get update
Apptainer> pip install open3d
Apptainer> apt install xxx
```

#### And here is my submit.sh file:

```shell
#!/bin/bash

#SBATCH -p mit_normal_gpu --gres=gpu:h100:2
#SBATCH -c 32

module load apptainer/1.1.9   # load singularity

singularity exec --nv \  # use GPU
  --bind /home/xxx/ultrasound/MUSiK:/workspace \ # bind workspace
  /home/xxx/my-image/ \ 
  python /workspace/demos/kidney/my_test.py # find python file under workspace
```



| Issue                 | Error message                                                | Solutions                                                    |
| --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| gcc outdated          | /lib64/libm.so.6: version `GLIBC_2.29' not found (required by kwave/bin/linux/\*\*\*\*) | Pull an image with Ubuntu 20.04+ with `--sandbox` and install required libraries with `singularity shell --writable --fakeroot my-image` flag. |
| Binaries not writable | Something like \*\*\*/kwave/linux/bin/\*\*\*\* not writable  | Don't pip install k-wave-python (or you have to reset binaries paths). Use it as git submodule in your workspace. |
| k-wave related        | anything reporting error inside k-wave package               | Check you k-wave version and roll back to a stable version   |
| libsz.so.2 not found  | error while loading shared libraries: libsz.so.2: cannot open shared object file: No such file or directory | run `apt-get update && apt-get install -y libsz2` with your `--writable --fakeroot` shell |
| Path issues           | xxx not found or cannot open file xxx                        | Check paths in shell file and **python file**. The path is reset to root when executing using singularity. |
