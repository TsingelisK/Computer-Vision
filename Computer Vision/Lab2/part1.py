# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 20:57:22 2021

@author: Panagiotis Gardelis  AM : 03117006
         Konstantinos Spathis AM : 03117048
          
"""

import cv2 
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
import sys
from matplotlib.patches import Circle
import scipy 
from scipy.io import loadmat
from scipy import signal 
from scipy import stats 
from scipy.stats import multivariate_normal
import scipy.io
import scipy.misc
import PIL
import skimage.measure
from scipy.ndimage import map_coordinates
from math import ceil 
from math import floor 
from scipy.ndimage import convolve
from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter
#from scipy.signal import convolve
from cv21_lab2_2_utils_without_tk import read_video
from cv21_lab2_2_utils_without_tk import show_detection
from cv21_lab2_2_utils_without_tk import orientation_histogram
from cv21_lab1_part2_utils import interest_points_visualization, disk_strel
from cv_lab2_meros1o import lk, gaussCreator, Acreator, justGETinterestPoints

"""####################################"""
"""############# MEROS 2o #############"""
"""####################################"""

#def to show image
def PlotarismaIMG(eikona, titlos):
    plt.imshow(eikona) 
    plt.title(titlos)
    plt.show()
    return None

#def to show image in grayscale
def grayIMG(eikona, titlos):
    plt.imshow(eikona, cmap = 'gray') 
    plt.title(titlos)
    plt.show()
    return None

"""####################################"""
"""############## ASK 2.1 #############"""
"""################################ARXI"""

boxerPath = 'cv21_lab2_part2_material/data/boxing/'
walkerPath = 'cv21_lab2_part2_material/data/walking'
runnerPath = 'cv21_lab2_part2_material/data/running/'

def ReaDviDeo(path,FileName,ariFra):
    viDeo =  read_video(path+FileName,ariFra,0)
    return viDeo

#FN = File Name && p6w --> person 4 walking
FNp4w = 'person05_walking_d2_uncomp.avi'
videoP4W = ReaDviDeo(walkerPath,FNp4w,20)

print("Check walker video")
print(videoP4W.shape)
print(" ")

#FN = File Name && p6w --> person 1 running
FNp1r = 'person01_running_d1_uncomp.avi'
videoP1R = ReaDviDeo(runnerPath,FNp1r,20)

print("Check runner video")
print(videoP1R.shape)
print(" ")

#FN = File Name && p6w --> person 1 boxing
FNp1b = 'person01_boxing_d2_uncomp.avi'
videoP1B = ReaDviDeo(boxerPath,FNp1b,20)

print("Check boxer video")
print(videoP1B.shape)
print(" ")

def getAframe(ToVideo,framePos):
    Frame = ToVideo[:,:,framePos]
    Frame = np.array(Frame)
    return Frame

frameP1R = getAframe(videoP1R,12)
#print(videoP1R.shape)
#print(videoP1R)
#grayIMG(frameP1R, "The frame of the video")

"""#------- PUT ALL FRAMES IN A LIST ---------#"""

def listaApoFramesCreator(EnaVideo,PlhthosFrames):
    listaApoFrames = []
    for i in range(0, PlhthosFrames):
        Helper_Frame = getAframe(EnaVideo,i)
        listaApoFrames.append(Helper_Frame)
    return listaApoFrames

FramesP1R = []
FramesP1R = listaApoFramesCreator(videoP1R,20)

""" ##JUST CHECKING IF EVERYTHING IS OKAY##
for i in range(0,len(FramesP1R)):
    grayIMG(FramesP1R[i], "The "+str(i+1)+"-osto frame of the video")
"""

"""------------------------------------"""
"""__________ Harris - Stephnes _______"""
"""--------------------------------ARXI"""

def gaussCreator1D(scale):
    n = int(np.ceil(3*scale)*2) + 1
    gauss1D = cv2.getGaussianKernel(n, scale)
    return gauss1D

def gaussCreator3D(scale,kernelWidth):
    gauss1 = gaussCreator1D(scale)
    gauss2 = gaussCreator1D(scale)
    gauss3 = gaussCreator1D(scale)
    gauss1neo, gauss2neo, gauss3neo = np.meshgrid(gauss1,gauss2,gauss3)
    ekth1 = gauss1neo**2 
    ekth2 = gauss2neo**2
    ekth3 = gauss3neo**2
    num = ekth1 + ekth2 + ekth3
    denum = 2*(kernelWidth**2)
    pyrhnas = np.exp(-num/denum)
    return pyrhnas

def GxytCreator(x,y,t,sigma,taf):
    num1 = (x**2) + (y**2)
    denum1 = 2*(sigma**2)
    ekth1 = -(num1/denum1)
    ekth2 = -((t**2)/(2*(taf**2)))
    numerator = np.exp(ekth1+ekth2)
    yporizo1 = (2*np.pi)**3
    yporizo2 = sigma**4
    yporizo3 = taf**2
    denumerator = np.sqrt(yporizo1*yporizo2*yporizo3)
    G_3d = numerator/denumerator
    return G_3d

def Gaussian3D(scale, sigma, taf):
    gauss1 = gaussCreator1D(sigma)
    gauss2 = gaussCreator1D(sigma)
    gauss3 = gaussCreator1D(sigma)
    x, y, t = np.meshgrid(gauss1,gauss2,gauss3)
    pyrhnas = GxytCreator(x,y,t,sigma,taf)
    return pyrhnas

def Gaussian3D_B(eS, sigma, taf):
    gauss1 = gaussCreator1D(sigma)
    gauss2 = gaussCreator1D(sigma)
    gauss3 = gaussCreator1D(sigma)
    x, y, t = np.meshgrid(gauss1,gauss2,gauss3)
    n = int(np.ceil(3*sigma)*2) + 1
    gauss1D = cv2.getGaussianKernel(eS*n, sigma)
    return gauss1D

def Conv(A,B):
    cconv = convolve(A, B, mode='constant', cval=0.0)
    return cconv

#Harris - Stephens - 3D
def HS3D(ToVideo,rho,eS,k,sigma,taf):
    
    #kernelWidth = 1 
    G3D_1 = Gaussian3D(rho, sigma, taf)
    #G3D_1 = GxytCreator(x,y,t,sigma,taf)
    
    #Outs_1 = cv2.filter2D(ToVideo, -1, G3D_1)
    #Outs_1 = np.convolve(ToVideo, G3D_1, mode='same')
    #print(Outs)
    Outs_1 = convolve(ToVideo, G3D_1, mode='constant', cval=0.0)
    
    Lx, Ly, Lt = np.gradient(Outs_1)
    
    Lx2 = Lx**2
    Ly2 = Ly**2
    Lt2 = Lt**2
    LxLy = Lx*Ly
    LxLt = Lx*Lt
    LyLt = Ly*Lt
    
    #Paragwgoi = np.array([[Lx2,LxLy,LxLt],[LxLy,Ly2,LyLt],[LxLt,LyLt,Lt2]])
    
    G3D_2 = Gaussian3D(eS, eS*sigma, eS*taf)
    
    #Tanystes for x
    
    J11 = scipy.signal.convolve(Lx2, G3D_2, mode="same")
    J12 = scipy.signal.convolve(LxLy, G3D_2, mode="same")
    J13 = scipy.signal.convolve(LxLt, G3D_2, mode="same")
    
    #Tanystes for y
    J21 = J12
    J22 = scipy.signal.convolve(Ly2, G3D_2, mode="same")
    J23 = scipy.signal.convolve(LyLt, G3D_2, mode="same")
    
    #Tanystes for z
    J31 = J13
    J32 = J23
    J33 = scipy.signal.convolve(Lt2, G3D_2, mode="same")
    
    #Create the teliko krithrio H
    H1 = + J11*((J22*J33)-(J23**2))
    H2 = - J12*((J21*J33)-(J31*J23))
    H3 = + J13*((J21*J32)-(J31*J22))
    H4 = - k*((J11+J22+J33)**2)
    H = H1 + H2 + H3 + H4
    ns = np.ceil(3*eS*sigma)*2+1
    B_sq = disk_strel(ns)
    HDilation = cv2.dilate(H,B_sq)
    megisto = H.max()
    print("To megisto einai : "+str(megisto))
    
    criterion = cv2.bitwise_and(np.where(H==HDilation,1,0),np.where(H>0.005*megisto,1,0))
    
    Points = []
    for i in range(criterion.shape[0]):
        for j in range(criterion.shape[1]):
            for k in range(criterion.shape[2]):
                if (criterion[i][j][k] == 1 ) :
                    Points.append((j,i,k,eS))
                    
    Points = np.reshape(Points,(len(Points),4))  
    
    return Points 

#Points_HS3D = HS3D(videoP1B,2,2,0.5,4,1.5)    
#print(Points_HS3D)
#show_detection(videoP1B, Points_HS3D, save_path="cv21_lab2_workspace")

"""------------------------------------"""
"""__________ Harris - Stephnes _______"""
"""-------------------------------TELOS"""

"""------------------------------------"""
"""___________ Gabor - Filter _________"""
"""--------------------------------ARXI"""

def gaussCreator2D(scale):
    n = int(np.ceil(3*scale)*2) + 1
    gauss1D = cv2.getGaussianKernel(n, scale)
    gauss2D = gauss1D @ gauss1D.T
    return gauss2D

def NormalNorma(h):
    h = h/np.linalg.norm(h,ord =1)
    return h
    
def HevenCreator(taf): 
    ti = np.arange(-2*taf,2*taf +1, 1 )
    omega = 4/taf
    shn = np.cos(2* np.pi*ti*omega)
    numExp = -(ti**2)
    denumExp = (2* (taf**2))
    expo = np.exp(numExp/denumExp)
    Heven = shn*expo
    Heven = NormalNorma(Heven)
    return Heven

def HoddCreator(taf): 
    ti = np.arange(-2*taf,2*taf +1, 1 )
    omega = 4/taf
    hmi = np.sin(2* np.pi*ti*omega)
    numExp = -(ti**2)
    denumExp = (2* (taf**2))
    expo = np.exp(numExp/denumExp)
    Hodd = hmi*expo
    Hodd = NormalNorma(Hodd)
    return Hodd

def OrismataKrithriou(Ig,h):
    Convo = scipy.ndimage.convolve1d(Ig,h)
    orisma = Convo**2
    return orisma

def KrithrioGabor(Ig, he, ho):
    H = OrismataKrithriou(Ig,he) + OrismataKrithriou(Ig,ho)
    return H

#Gabor - 3D
def Gabor_3D(ToVideo,sigma,taf):
    
    ToVideo = ToVideo.astype(np.float)/255
    
    #gabor creation
    Gs = gaussCreator2D(sigma)
    
    #synelixh
    Ig = cv2.filter2D(ToVideo, -1, Gs)
    
    #Creation of hev and ho
    Heven = HevenCreator(taf)
    Hodd = HoddCreator(taf)
    
    #Creation of criterion
    H = KrithrioGabor(Ig, Heven, Hodd)
    
    #Katwfli
    ns = np.ceil(3*sigma)*2+1
    B_sq = disk_strel(ns)
    HDilation = cv2.dilate(H,B_sq)
    megisto = H.max()
    print("To megisto einai : "+str(megisto))
    
    criterion = cv2.bitwise_and(np.where(H==HDilation,1,0),np.where(H>0.05*megisto,1,0))
    
    Points = []
    for i in range(criterion.shape[0]):
        for j in range(criterion.shape[1]):
            for k in range(criterion.shape[2]):
                if (criterion[i][j][k] == 1 ) :
                    Points.append((j,i,k,sigma))
                    
    Points = np.reshape(Points,(len(Points),4))  
    
    return Points

Points_Gabor = Gabor_3D(videoP1B, 2, 1.5)
print(Points_Gabor)
show_detection(videoP1B, Points_Gabor, save_path="cv21_lab2_workspace")

"""------------------------------------"""
"""___________ Gabor - Filter _________"""
"""-------------------------------TELOS"""

"""####################################"""
"""############## ASK 2.1 #############"""
"""###############################TELOS"""


"""####################################"""
"""############## ASK 2.2 #############"""
"""################################ARXI"""

FramesP1B = []
FramesP1B = listaApoFramesCreator(videoP1B,20)

def KopseThnEikona2(eikona, x, y, height, width):
    kommenh = eikona[y:y+width, x:x+height]
    return kommenh

def HistoPeri(listaMeEikones,nbins,box):
    
    Antigrafa = [listaMeEikones[0].copy()]
    Dx = []
    Dy = []
    for i in range(0,len(listaMeEikones)-1):
        Antigrafa.append(listaMeEikones[i+1].copy())
        features = cv2.goodFeaturesToTrack(Antigrafa[i],25,0.01,10)
        Kommenh1 = KopseThnEikona2(Antigrafa[i], box[0], box[1], box[2], box[3])
        Kommenh2 = KopseThnEikona2(Antigrafa[i+1], box[0], box[1], box[2], box[3])
        dxi, dyi = lk(Kommenh1, Kommenh2, features, 4, 0.01, 0, 0, 50)
        Dx.append(dxi)
        Dy.append(dyi)
    #desc = orientation_histogram(Gx,Gy,nbins,[n m])
    Dxy = np.expand_dims(Dx,axis=1)
    Dxy = np.insert(Dxy,0,Dy,axis=1)
    print(Dxy)
    return None

boxaki = [100, 100, 29, 34]
#HistoPeri(FramesP1B,10, boxaki)

#dx_f1, dy_f1 = lk(faceImg_1, faceImg_2, features_f1, 4, 0.01, 0, 0, 50)



"""####################################"""
"""############## ASK 2.2 #############"""
"""###############################TELOS"""


"""####################################"""
"""############## ASK 2.3 #############"""
"""################################ARXI"""


"""####################################"""
"""############## ASK 2.3 #############"""
"""###############################TELOS"""


