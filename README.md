# ICAR : Interaction-Centric Automatic Layout Generation for AR Elements
We present ICAR, a novel Interaction Centric automatic layout generation framework for AR elements. Our key idea is to use the critical role of human interaction in layout generation for
AR elements.
![NO TEASER][./images/teaser.png]

## Description

This repository contains the viewpoint estimation, layout generation and data processing code used for the experiments in `ICAR`.

## Installation 
To install the necessary dependencies run the following command:
```shell
    pip install -r requirements.txt
```
## Get Started
、、、
git clone --recursive https://github.com/AaronJackson/vrn.git
cd src
、、、
## Scenes Preparation
## Viewpoint Estimation
we provide an examples of how to estimate viewpoint for target object  `0` in scene `S1_E1`. 
```
python ./population.py --config ../cfg_files/contact_semantics.yaml --checkpoint_path $POSA_dir/trained_models/contact_semantics.pt --pkl_file_path $POSA_dir/POSA_rp_poses/rp_ethan_posed_012_0_0.pkl --scene_name S1_E1 --render True --viz True --target_object 0 --show_init_pos True --use_cuda True --use_semantics 1
```
This will open a window showing all the cuiling initial positions for the specified `pkl` file. Then a window showing the final results will be opened.

It also render the results and save them.

If you don't have a screen, you can turn off the visualization `--viz 0`.

If you don't have CUDA installed then you can add this flag `--use_cuda 0`. This applies to all commands in this repository.

## Layout Generation
we provide an examples of how to generate layout for AR elements in Scenario `1` for scene `S1_E1` with the viewpoint estimated from posed_body `rp_ethan_posed_012_0_0_00_00`. 
```
python ./layout.py --scene_name S1_E1 --body_name rp_ethan_posed_012_0_0_00_00 --scenario 1
```

## Structure of list of AR elements 

