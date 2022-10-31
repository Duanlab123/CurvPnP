from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import cv2

def Dis(u):

    # -----------------------------------------------------------------
    # compute discrete curvatures in a 3*3 region of center point u
    # -----------------------------------------------------------------

    # ---------------------------
    #      initialization
    # ---------------------------
    H, W = u.shape[:2]
    
    dist = torch.zeros((H, W, 8))
    # -------------------------------------------------------------------
    # define 8 normal directions in 3*3 region of center point u(:,:)
    # -------------------------------------------------------------------

    u1 = torch.cat((u[H-1:, :], u[:H-1, :]),0)     
    u2 = torch.cat((u[1:, :], u[:1, :]),0)        
    u3 = torch.cat((u[:, W-1:], u[:, :W-1]),1)    
    u4 = torch.cat((u[:, 1:] ,u[:, :1]),1)  
    u5 = torch.cat((u1[:,W-1:], u1[:, :W-1]),1)  
    u6 = torch.cat((u2[:, 1:], u2[:, :1]),1)  
    u7 = torch.cat((u1[:, 1:], u1[:, :1]),1)  
    u8 = torch.cat((u2[:,W-1:], u2[:, :W-1]),1)  

    # -----------------------------------------------------------------
    # find 8 proximal points in 3*3 region near to the center point
    # -----------------------------------------------------------------

    u01 = (u1 + u) / 2
    u02 = (u2 + u) / 2
    u03 = (u3 + u) / 2
    u04 = (u4 + u) / 2
    u05 = (u5 + u3 + u1 + u) / 4
    u06 = (u6 + u4 + u2 + u) / 4
    u07 = (u7 + u4 + u1 + u) / 4
    u08 = (u8 + u3 + u2 + u) / 4

    # --------------------------------------------------------------------
    # compute 8 normal curvatures in 3*3 region of center point u(:,:)
    # --------------------------------------------------------------------

    dist[:, :, 0] = ((2 * u - u3 - u4) / torch.sqrt((2 * u1 - u3 - u4) ** 2 + (u3 - u4) ** 2 + 4)) / ((u01 - u) ** 2 + 1)

    dist[:, :, 1] = ((u3 + u4 - 2 * u) / torch.sqrt((2 * u2 - u3 - u4) ** 2 + (u4 - u3) ** 2 + 4)) / ((u02 - u) ** 2 + 1)

    dist[:, :, 2] = ((u1 + u2 - 2 * u) / torch.sqrt((u1 + u2 - 2 * u3) ** 2 + (u2 - u1) ** 2 + 4)) / ((u03 - u) ** 2 + 1)

    dist[:, :, 3] = ((2 * u - u1 - u2) / torch.sqrt((u1 + u2 - 2 * u4) ** 2 + (u1 - u2) ** 2 + 4)) / ((u04 - u) ** 2 + 1)

    dist[:, :, 4] = ((u5 + u7 + u8 - u1 - u3 - u) / torch.sqrt((u8 - u5) ** 2 + (u7 - u5) ** 2 + 4)) / ((u05 - u) ** 2 + 2)

    dist[:, :, 5] = ((u + u2 + u4 - u6 - u7 - u8) / torch.sqrt((u7 - u6) ** 2 + (u8 - u6) ** 2 + 4)) / ((u06 - u) ** 2 + 2)

    dist[:, :, 6] = ((u + u1 + u4 - u5 - u6 - u7) / torch.sqrt((u7 - u6) ** 2 + (u5 - u7) ** 2 + 4)) / ((u07 - u) ** 2 + 2)

    dist[:, :, 7] = ((u5 + u6 + u8 - u2 - u3 - u) / torch.sqrt((u8 - u5) ** 2 + (u6 - u8) ** 2 + 4)) / ((u08 - u) ** 2 + 2)

    # ----------------------------------------------------------------------
    # find max and min normal curvatures in 3*3 region of center point u
    # ----------------------------------------------------------------------
    row, col = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    tmp = abs(dist)
    ind1 = torch.argmin(tmp, axis=2)
    ind2 = torch.argmax(tmp, axis=2)

    # -----------------------------------------------
    # compute gaussian curvature in every point u
    # -----------------------------------------------
    m = dist[row, col, ind1] * dist[row, col, ind2]  # gaussian curvature

    return m
