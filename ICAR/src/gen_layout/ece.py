import torch
import torch.optim as Optim
import numpy as np
import open3d as o3d
import os.path as osp
import math
import torch.linalg
import pdb
import pickle
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import datetime
import sys
sys.path.append("D:\code\python\POSA\src")
import layout
from gen_human import eulerangles
from gen_human import data_utils 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy.ma as ma


def compute_elbow_pos(hand_pos,r_shoulder,body_orient,proarm_len=0.38,forearm_len=0.46,ang_step = math.pi/16):
    '''
    hand_pos   : 手指位置
    r_shoulder : 右边肩关节的位置
    body_orient: 人体朝向
    proarm_len : 大臂长度
    forearm_len: 小臂长度
    ang_step   : 肘关节位置采样间隔(度)
    '''
    #将手的坐标旋转至肩关节坐标系下
    #在世界坐标系下，默认人与坐标轴对齐，面向y轴负方向：
    axis = np.cross(np.array([.0, -1.0, .0]), body_orient) # 旋转轴
    axis_norm = axis/np.linalg.norm(axis)
    angle = np.arccos(np.dot(np.array([.0, -1.0, 0.0]), body_orient)) #旋转角
    rot_mat = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_norm * angle)
    
    hand_pos_trans = hand_pos - r_shoulder
    hand_pos_rot = torch.matmul(torch.tensor(rot_mat,dtype=torch.float32).t(),hand_pos_trans.reshape(3,1)).reshape(1,3)
    
    # 计算肘关节位置
    # 如果hand_pos在身体前侧且超出手臂长度范围，肘关节固定在肩-手方向大臂长度处  (1,3)
    # 如果hand_pos在身体前侧且在手臂长度范围内，按照肘关节平面计算肘关节位置  (N,3)
    # 如果hand_pos在身体后侧，则没有可能的肘关节位置，
    
    if torch.norm(hand_pos_rot,dim=1) > (proarm_len + forearm_len):
        elbow_pos = hand_pos_rot / torch.norm(hand_pos_rot) * proarm_len
    else :
        hand_dir = (hand_pos_rot)/torch.norm(hand_pos_rot,dim=1) #[1,3]
        y_dir = torch.tensor([.0,1.0,.0],dtype=torch.float32).unsqueeze(0) #需要修改 POSA坐标系下，重力坐标方向为 (0,0,-1)
        u = -y_dir + F.cosine_similarity(y_dir,hand_dir) * hand_dir  #[1,3]
        u = u/torch.norm(u,dim=1)
        v = torch.cross(hand_dir,u) #[1,3]
    
        cos_beta = (proarm_len**2 + torch.norm(hand_pos_rot)**2 - forearm_len **2)/(2 * proarm_len * torch.norm(hand_pos_rot)) #(1,)
        sin_beta = torch.sqrt(1 - cos_beta**2) #(1,)
    
        center = cos_beta * proarm_len * hand_dir #(1,3)
        radius = sin_beta * proarm_len  #(1,)
    
        # 计算肘关节位置
        angles = -torch.arange(0, math.pi * 3/4, ang_step, dtype=torch.float32)
        cos_values = torch.cos(angles).unsqueeze(1) #(36,)
        sin_values = torch.sin(angles).unsqueeze(1)
        elbow_pos = radius * (cos_values * u + sin_values * v) + center
        
    return hand_pos_rot, elbow_pos

def compute_ECE(hand_pos,proarm_len=0.2817,forearm_len=0.2689,palm_len=0.0862,hand_len=0.1899,weight=73.0,gender='M',ang_step = math.pi/16):
    '''
    假定手部坐标是世界肩膀坐标系下的坐标，z轴朝上，-y轴为朝向方向
    hand_pos   : 手指位置
    proarm_len : 大臂长度
    forearm_len: 小臂长度
    palm_len   : 手掌心长度
    hand_len   : 全手长度
    weight     : 体重
    gender     : 性别
    ang_step   : 肘关节位置采样间隔(度)
    '''
    # 计算肘关节位置
    # 如果hand_pos太远，肘关节固定在肩-手方向大臂长度处          (1,3)
    # 如果hand_pos在手臂长度范围内，按照肘关节平面计算肘关节位置  (N,3)
    hand_pos = hand_pos.reshape(1,3)
    if torch.norm(hand_pos,dim=1) >= (proarm_len + forearm_len + hand_len):
        elbow_pos = hand_pos / torch.norm(hand_pos,dim=1) * proarm_len 
    else :
        hand_dir = (hand_pos)/torch.norm(hand_pos,dim=1) #[1,3]
        # y_dir = torch.tensor([.0,1.0,.0],dtype=torch.float32).unsqueeze(0) #需要修改 POSA坐标系下，重力坐标方向为 (0,0,-1)
        # u = -y_dir + F.cosine_similarity(y_dir,hand_dir) * hand_dir  #[1,3]
        z_dir = torch.tensor([.0,.0,1.0],dtype=torch.float32).unsqueeze(0) #需要修改 POSA坐标系下，重力坐标方向为 (0,0,-1)
        u = -z_dir + F.cosine_similarity(z_dir,hand_dir) * hand_dir  #[1,3]
        u = u/torch.norm(u,dim=1)
        # v = torch.cross(hand_dir,u) #[1,3]
        v = torch.cross(u,hand_dir)
    
        cos_beta = (proarm_len**2 + torch.norm(hand_pos)**2 - (forearm_len + hand_len) **2)/(2 * proarm_len * torch.norm(hand_pos)) #(1,)
        sin_beta = torch.sqrt(1 - cos_beta**2) #(1,)
    
        center = cos_beta * proarm_len * hand_dir #(1,3)
        radius = sin_beta * proarm_len  #(1,)
    
        # 计算肘关节位置
        # angles = -torch.arange(0, math.pi * 3/4, ang_step, dtype=torch.float32)
        angles = torch.arange(math.pi * 1/8, math.pi * 2/4, ang_step, dtype=torch.float32)
        cos_values = torch.cos(angles).unsqueeze(1) #(36,)
        sin_values = torch.sin(angles).unsqueeze(1)
        elbow_pos = radius * (cos_values * u + sin_values * v) + center
    
    #计算质心位置
    elbow_dir = (elbow_pos)/(torch.norm(elbow_pos,dim=1).unsqueeze(1))
    elbow2hand_dir = (hand_pos - elbow_pos) / torch.norm(hand_pos - elbow_pos,dim=1).unsqueeze(1)
    
    if gender == 'M':
        proarm_mass = weight * 0.0271
        forearm_mass = weight * 0.0162
        hand_mass = weight * 0.0061
        
        proarm_com = elbow_dir * proarm_len * 0.5772
        forearm_com = elbow_pos + elbow2hand_dir * forearm_len * 0.4574
        hand_com = elbow_pos + elbow2hand_dir * (forearm_len + palm_len * 0.79)
        
    if gender == "F":
        proarm_mass = weight * 0.0255
        forearm_mass = weight * 0.0138
        hand_mass = weight * 0.0056
        
        proarm_com = elbow_dir * proarm_len * 0.5754
        forearm_com = elbow_pos + elbow2hand_dir * forearm_len * 0.4559
        hand_com = elbow_pos + elbow2hand_dir * (forearm_len + palm_len * 0.7474)    
        
    arm_mass = proarm_mass + forearm_mass + hand_mass
    
    if torch.norm(hand_pos,dim=1) <= (proarm_len + forearm_len + hand_len):
        arm_com = proarm_com * (proarm_mass / arm_mass) + forearm_com * (forearm_mass / arm_mass) + hand_com * (hand_mass / arm_mass)
    else:
        #如果交互位置超出手臂长度，修改计算重心的方式
        #计算临界条件
        if gender == 'M':
            bound_com = (proarm_len * 0.5772) * (proarm_mass / arm_mass) + (proarm_len + forearm_len * 0.4574) * (forearm_mass / arm_mass) + (proarm_len + forearm_len + palm_len * 0.79) * (hand_mass / arm_mass)
        if gender == 'F':
            bound_com = (proarm_len * 0.5754) * (proarm_mass / arm_mass) + (proarm_len + forearm_len * 0.4559) * (forearm_mass / arm_mass) + (proarm_len + forearm_len + palm_len * 0.7474) * (hand_mass / arm_mass)
        
        
        arm_len = proarm_len + forearm_len + hand_len
        arm_com = hand_pos * (bound_com / arm_len)
    
    r = arm_com
    mg = arm_mass * torch.tensor([.0,.0,-9.8],dtype=torch.float32).unsqueeze(0)
    # torque_shoulder = torch.cross(r,mg.expand_as(r)) 
    # torque_shoulder = torch.norm(torque_shoulder,dim=1)/101.6  
    torque_shoulder_cos = 1 - F.cosine_similarity(r,mg.expand_as(r),dim=-1)
    # if gender == 'M': 
    #     torque_shoulder = torch.norm(mg) * torque_shoulder_cos * torch.norm(r,dim=1)/101.6
    # if gender == 'F':
    #     torque_shoulder = torch.norm(mg) * torque_shoulder_cos * torch.norm(r,dim=1)/87.2
    if gender == 'M': 
        torque_shoulder = torch.norm(mg) * torque_shoulder_cos * torch.norm(r,dim=1)/(101.6 * (proarm_len + forearm_len + hand_len))
    if gender == 'F':
        torque_shoulder = torch.norm(mg) * torque_shoulder_cos * torch.norm(r,dim=1)/(87.2 * (proarm_len + forearm_len + hand_len))
    torque_min = torque_shoulder.min() 

    return torque_min
