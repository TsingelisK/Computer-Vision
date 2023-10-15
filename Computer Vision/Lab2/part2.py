# -*- coding: utf-8 -*-
"""

Created on Sun May 23 20:23:22 2021

@authors: Panagiotis Gardelis  AM : 03117006
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

"""#-----------------------------------#"""
"""#----------- ASKHSH 1 --------------#"""
"""#-----------------------------------#"""

"""##############################################"""
"""########### ASKHSH 1.1 ###############ARXI"""
"""##############################################"""

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

def CbCrPlotarisma(eikona,titlos):
    Kopia = eikona.copy()
    (Y, Cb, Cr) = cv2.split(Kopia)
    CbCr = cv2.merge(( Cb, Cr))
    PlotarismaIMG(CbCr, titlos)
    return None

#def to read images and plot their og image
def readImage(imageName):
    
    #reading the image 
    IMG = cv2.imread("GreekSignLanguage/"+imageName+".png", cv2.IMREAD_COLOR) 
    #IMG = cv2.imread("villegas.jpg", cv2.IMREAD_COLOR)
    
    #print("Resolution of "+imageName+" is : " , IMG.shape, "pixels")
    
    IMGrgb = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)
    IMGrgb_Help1 = IMGrgb.copy()
    IMGrgb_Help2 = IMGrgb.copy()
    
    #PlotarismaIMG(IMGrgb, "Original Image of "+imageName)
        
    #grayscale
    IMGgray = cv2.cvtColor(IMGrgb_Help1, cv2.COLOR_RGB2GRAY)
    #IMGgray = IMGrgb
    
    #YCrCb
    IMGyrb = cv2.cvtColor(IMGrgb_Help2, cv2.COLOR_RGB2YCrCb)
    
    #normalize from 0-255 to [0,1]
    #metatropi eikonas se float
    IMGrgb = IMGrgb.astype(np.float) / 255
    IMGyrb = IMGyrb.astype(np.float) / 255
    #print("Resolution of "+imageName+" yrb is : " , IMGyrb.shape, "pixels")
    #print("NEW Range : {} - {} ".format(IMGyrb.min(),IMGyrb.max()))

    return IMGyrb, IMGrgb, IMGgray

IMGyCrCb_1,IMGrgb_1 ,IMGgray_1 = readImage("1")
PlotarismaIMG(IMGrgb_1, "RGB Image of image 1")
PlotarismaIMG(IMGyCrCb_1, "YCrCb Image of image 1")
RGB_1 = IMGrgb_1.copy()

"""
Kopia = IMGyCrCb_1.copy()
(Y, Cb, Cr) = cv2.split(Kopia)
CbCr = cv2.merge(( Cb, Cr))
PlotarismaIMG(CbCr, "The CbCr channels of the image")
"""

def meanAndSigma(eikona):
    (Y, Cb, Cr) = cv2.split(eikona)
    
    #flatten the channels
    Cb = Cb.flatten()
    Cr = Cr.flatten()
    meanCb = np.mean(Cb)
    meanCr = np.mean(Cr)
    mean = np.array([meanCb,meanCr])
    print("mean :")
    print(mean)
    print(" ")
    Sigma = np.cov(Cb,Cr)
    print("Syndiakimansi :")
    print(Sigma)
    print(" ")
    return mean, Sigma

mat = loadmat('GreekSignLanguage/skinSamplesRGB.mat')

#get the important stuff into a numpy array
derma = np.array(mat['skinSamplesRGB'])

#convert RGB to YCbCr
dermayrb = cv2.cvtColor(derma, cv2.COLOR_RGB2YCrCb)
dermayrb = dermayrb.astype(np.float) / 255

meanMAT, SigmaMAT = meanAndSigma(dermayrb)

def PskinCalculator(eikona,m,s):
    (Y, Cb, Cr) = cv2.split(eikona)
    CbCr = cv2.merge(( Cb, Cr))
    
    #CbCr = CbCr.astype(np.float) / 255
    Pk = stats.multivariate_normal.pdf(CbCr, mean=m, cov=s)
    return Pk

Pskin_mat = PskinCalculator(derma, meanMAT, SigmaMAT)

#PlotarismaIMG(Pskin_mat, "The possibility in image")

def properCordinates(id_1, l):
    oros1 = l[id_1].bbox[1]
    oros2 = l[id_1].bbox[0]
    oros3 = l[id_1].bbox[3] - l[id_1].bbox[1]
    oros4 = l[id_1].bbox[2] - l[id_1].bbox[0]
    lista = [oros1, oros2, oros3, oros4]
    return lista

def RegionDetector(I,mu,cov):
    
    #get the possibility of skin in image 1 
    PskinIMG = PskinCalculator(I,mu,cov)
    
    #find the max of the Pskin
    print("The max of the Pskin is : "+str(PskinIMG.max()))
    print(" ")
    
    #normalize the possibility
    PskinIMG = PskinIMG.astype(np.float)/PskinIMG.max()
    grayIMG(PskinIMG,"Image Showing the possibility of skin")
    
    #efarmogi katwfliou
    _, normalIMG = cv2.threshold(PskinIMG,0.05,1, cv2.THRESH_BINARY)
    grayIMG(normalIMG,"Binary Image ")
    
    #efarmogi opening me mikro pyrhna
    OpenPyrhnas = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    openingIMG = cv2.morphologyEx(normalIMG, cv2.MORPH_OPEN,OpenPyrhnas)
    grayIMG(openingIMG,"Binary Image after morph open")
    
    #efarmogi closing me megalo pyrhna
    ClosePyrhnas = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23))
    closingIMG = cv2.morphologyEx(openingIMG, cv2.MORPH_CLOSE,ClosePyrhnas)
    grayIMG(closingIMG,"Binary Image after morph close")
    
    #xrhsh label
    labelsIMG,_ = scipy.ndimage.label(closingIMG)
    perioxesSkin = skimage.measure.regionprops(labelsIMG)
    print(" ")
    
    #create the Bounding Boxes 
    Proswpo = properCordinates(0, perioxesSkin)
    print("The coordinates of the Face are : "+str(Proswpo))
    
    AristeroXeri = properCordinates(1, perioxesSkin)
    print("The coordinates of the Left Hand are : "+str(AristeroXeri))
    
    DexiXeri = properCordinates(2, perioxesSkin)
    print("The coordinates of the Right Hand are : "+str(DexiXeri))
    print(" ")
    
    return Proswpo, AristeroXeri, DexiXeri

Proswpo_1, AristeroXeri_1, DexiXeri_1 = RegionDetector(IMGyCrCb_1, meanMAT, SigmaMAT)

def CreateBoxesInIMG(imgRGB, Proswpo, AristeroXeri, DexiXeri):
    #Red color 
    red = (255, 0, 0)
    #Green color  
    green = (0,255,0)
    #Blue color 
    blue = (0,0,255)
    
    #declare a new image to be skin detected --> SD
    IMGwithBOXES = imgRGB
    cv2.rectangle(IMGwithBOXES, (Proswpo[0], Proswpo[1]), (Proswpo[2]+Proswpo[0], Proswpo[3]+Proswpo[1]), red, 3)
    cv2.rectangle(IMGwithBOXES, (AristeroXeri[0], AristeroXeri[1]), (AristeroXeri[2]+AristeroXeri[0], AristeroXeri[3]+AristeroXeri[1]), green, 3)
    cv2.rectangle(IMGwithBOXES, (DexiXeri[0], DexiXeri[1]), (DexiXeri[2]+DexiXeri[0], DexiXeri[3]+DexiXeri[1]), blue, 3)

    return IMGwithBOXES

IMGrgb_1_SD = CreateBoxesInIMG(IMGrgb_1, Proswpo_1, AristeroXeri_1, DexiXeri_1)

PlotarismaIMG(IMGrgb_1_SD, "Skin detection on image 1")

def CombinedBoxesANDregion(I,mu,cov,imgRGB):
    Face, LEFThand, RIGHThand = RegionDetector(I,mu,cov)
    #imgBGR = cv2.cvtColor(I, cv2.COLOR_YCrCb2BGR)
    #imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
    NeaEikona = CreateBoxesInIMG(imgRGB, Face, LEFThand, RIGHThand)
    return NeaEikona

"""##############################################"""
"""########### ASKHSH 1.1 ###############TELOS"""
"""##############################################"""

"""##############################################"""
"""########### ASKHSH 1.2 ###############ΑΡΧΙ"""
"""##############################################"""

def getSyniswstes(skinLista):
    x = skinLista[0]
    y = skinLista[1]
    height = skinLista[2]
    width = skinLista[3]
    return x, y, height, width

#O oros f1 --> face of image No 1
x_f1, y_f1, height_f1, width_f1 = getSyniswstes(Proswpo_1)

#O oros LH1 --> Left Hand of image No 1
x_LH1, y_LH1, height_LH1, width_LH1 = getSyniswstes(AristeroXeri_1)

#O oros RH1 --> Right Hand of image No 1
x_RH1, y_RH1, height_RH1, width_RH1 = getSyniswstes(DexiXeri_1)

"""# For the 2nd image #ARXI"""
IMGyCrCb_2 ,IMGrgb_2 ,IMGgray_2 = readImage("2")
Proswpo_2, AristeroXeri_2, DexiXeri_2 = RegionDetector(IMGyCrCb_2, meanMAT, SigmaMAT)
x_f2, y_f2, height_f2, width_f2 = getSyniswstes(Proswpo_2)
x_LH2, y_LH2, height_LH2, width_LH2 = getSyniswstes(AristeroXeri_2)
x_RH2, y_RH2, height_RH2, width_RH2 = getSyniswstes(DexiXeri_2)
"""# For the 2nd image #TELOS"""

""" TELIKA XREIAZOMAI TOUS PINAKES POU DINONTAI """
listaProswpo = [138, 88, 73, 123]
listaLeftHand =  [47, 243, 71, 66]
listaRightHand = [162, 264, 83, 48]

#isws na mhn mas noiazei to frame tou proswpou alla h 
#thesi tou rectangular tou 
def KopseThnEikona(eikona, x, y, height, width):
    kommenh = eikona[y:y+width, x:x+height, :]
    return kommenh

def readImageNoNormal(imageName):
    
    #reading the image 
    IMG = cv2.imread("GreekSignLanguage/"+imageName+".png", cv2.IMREAD_COLOR) 
    IMGrgb = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)
    IMGrgb_Help1 = IMGrgb.copy()
    IMGrgb_Help2 = IMGrgb.copy()        
    #grayscale
    IMGgray = cv2.cvtColor(IMGrgb_Help1, cv2.COLOR_RGB2GRAY)
    #YCrCb
    IMGyrb = cv2.cvtColor(IMGrgb_Help2, cv2.COLOR_RGB2YCrCb)
    
    return IMGyrb, IMGrgb, IMGgray

def InterestPoints(eikona):
    #print(eikona.shape)
    #eikona = cv2.cvtColor(eikona, cv2.COLOR_BGR2RGB)
    eikona = cv2.cvtColor(eikona, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(eikona,25,0.01,10)
    #corners = np.int0(corners)
    #print("The corners are : ")
    #print(corners)   
    for i in corners:
        x, y = i.ravel()
        Ccenter = (x,y)
        color = (255, 0, 0)
        radius = 5
        thickness = 1
        eikona = cv2.circle(eikona,Ccenter, radius, color, thickness)
             
    return eikona

"""########## VISUALIZE THE INTEREST POINTS ##############
#i --> image , nn --> No Normal, y --> YCrCb , r --> RGB , g --> GRAY
iynn1, irnn1, ignn1 = readImageNoNormal("1")

faceImg_1 = KopseThnEikona(irnn1, x_f1, y_f1, height_f1, width_f1)
PlotarismaIMG(faceImg_1, "Only face of nohmahtria")

#faceImg_gray_1 = KopseThnEikona(Eikones_gray[0], x_f1, y_f1, height_f1, width_f1)
#grayIMG(faceImg_gray_1, "Only gray face of nohmahtria")

#Interest points of image 1 
IPof_1 = InterestPoints(faceImg_1)
grayIMG(IPof_1 , "Interest Points of only face of nohmahtria")
"""

def justGETinterestPoints(eikona):
    eikona = cv2.cvtColor(eikona, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(eikona,25,0.01,10)
    #corners = np.int0(corners)         
    return corners

"""# For the 1nd image #ARXI"""
#i --> image , nn --> No Normal, y --> YCrCb , r --> RGB , g --> GRAY
iynn1, irnn1, ignn1 = readImageNoNormal("1")
faceImg_1 = KopseThnEikona(irnn1, listaProswpo[0], listaProswpo[1], listaProswpo[2], listaProswpo[3])
LeftHandImg_1 = KopseThnEikona(irnn1, listaLeftHand[0], listaLeftHand[1], listaLeftHand[2], listaLeftHand[3])
RightHandImg_1 = KopseThnEikona(irnn1, listaRightHand[0], listaRightHand[1], listaRightHand[2], listaRightHand[3])

"""# For the 1nd image #TELOS"""

"""# For the 2nd image #ARXI"""
iynn2, irnn2, ignn2 = readImageNoNormal("2")
faceImg_2 = KopseThnEikona(irnn2, listaProswpo[0], listaProswpo[1], listaProswpo[2], listaProswpo[3])
LeftHandImg_2 = KopseThnEikona(irnn2, listaLeftHand[0], listaLeftHand[1], listaLeftHand[2], listaLeftHand[3])
RightHandImg_2 = KopseThnEikona(irnn2, listaRightHand[0], listaRightHand[1], listaRightHand[2], listaRightHand[3])

"""# For the 2nd image #TELOS"""

#def to create A 
#WARNING it needs grayscale image
def Acreator(eikona,d_x0, d_y0):
    #eikona =cv2.cvtColor(eikona,cv2.COLOR_RGB2GRAY)
    #eikona = eikona.astype(np.float)/255 
    x0, y0 = np.meshgrid(np.arange(eikona.shape[1]),np.arange(eikona.shape[0]))
    #xNeo = np.ravel(x0+dxy[0])
    #yNeo = np.ravel(y0+dxy[1])
    cord = map_coordinates(eikona, [(y0 + d_y0), (x0 + d_x0)], order=1)
    A1 = np.gradient(cord,axis=1)
    A2 = np.gradient(cord,axis=0)
    #A = np.array([A1,A2])
    return A1, A2

def gaussCreator(scale):
    n = int(np.ceil(3*scale)*2) + 1
    gauss1D = cv2.getGaussianKernel(n, scale)
    gauss2D = gauss1D @ gauss1D.T 
    return gauss2D

def lk(I1, I2, features, rho, epsilon, d_x0, d_y0, repeat):

    #make the images gray 
    I1 = cv2.cvtColor(I1, cv2.COLOR_RGB2GRAY)
    I1 = I1.astype(np.float)/255
    
    I2 = cv2.cvtColor(I2, cv2.COLOR_RGB2GRAY)
    I2 = I2.astype(np.float)/255
   
    
    #split the features
    features1, features2 = cv2.split(features)
    
    #make empty areas
    Teliko_dx = np.zeros(I1.shape)
    Teliko_dy = np.zeros(I2.shape)
    
    #for the first time
    dx = d_x0
    dy = d_y0
    
    for i in range(0,len(features1)):
        y = int(features1[i])
        x = int(features2[i])
        
        for j in range(0,repeat):
            #create the gaussian filter 
            Gr = gaussCreator(rho)
            #create the A
            A1, A2 = Acreator(I1, dx, dy)
            #print("eimai kala eimai sthn "+str(i+1)+"-osth fora")
            x0, y0 = np.meshgrid(np.arange(I1.shape[1]),np.arange(I1.shape[0]))
            #create In-1
            InMeion1 = map_coordinates(I1,[(y0+dy), (x0+dx)],order=1)
            #create the E
            E = I2 - InMeion1 
        
            #create the dependecies of matrix u2x2
            u2x2_a = cv2.filter2D((A1**2), -1, Gr) + epsilon
            u2x2_b = cv2.filter2D((A1*A2), -1, Gr) 
            u2x2_c = cv2.filter2D((A1*A2), -1, Gr) 
            u2x2_d = cv2.filter2D((A2**2), -1, Gr) + epsilon
        
            #calculate the orizousa
            detOFu2x2 = abs(u2x2_a*u2x2_d - u2x2_b*u2x2_c)
        
            #create the inverse u2x2 
            u2x2_11 = u2x2_d / detOFu2x2
            u2x2_12 = - ( u2x2_b / detOFu2x2 )
            u2x2_21 = - ( u2x2_c / detOFu2x2 )
            u2x2_22 = u2x2_a / detOFu2x2
        
            #create the dependecies of matrix u2x1
            u2x1_a = cv2.filter2D((A1*E), -1, Gr) 
            u2x1_b = cv2.filter2D((A2*E), -1, Gr)
        
            #calculate the Ufinal 
            Ufinal_1 = u2x2_11*u2x1_a + u2x2_12*u2x1_b
            Ufinal_2 = u2x2_21*u2x1_a + u2x2_22*u2x1_b 
        
            #get only the features  
            U_1 = Ufinal_1[x,y]
            U_2 = Ufinal_2[x,y]
            
            #FINALLY the dx and dy are
            dx = dx + U_1 
            dy = dy + U_2
        
        Teliko_dx[x,y] = dx
        Teliko_dy[x,y] = dy
        
    return Teliko_dx, Teliko_dy

"""##############################################"""
""" FOR THE FACE OF NOHMATRIA FROM F1 --> F2 """
"""##############################################"""

FfaceImg_1 = faceImg_1.copy()
features_f1 = justGETinterestPoints(FfaceImg_1)
dx_f1, dy_f1 = lk(faceImg_1, faceImg_2, features_f1, 4, 0.01, 0, 0, 50)

PlotarismaIMG(faceImg_1, "OG of face image 1")
PlotarismaIMG(faceImg_2, "Next frame of image 2")

plt.quiver(-dx_f1,-dy_f1,angles='xy',scale=100)
plt.gca().invert_yaxis()
plt.title("Optical Flow for the Face from frame 1 -> 2")
plt.show()

"""##############################################"""
""" FOR THE LEFT HAND OF NOHMATRIA FROM F1 --> F2 """
"""##############################################"""

FLeftHandImg_1 = LeftHandImg_1.copy()
features_lh1 = justGETinterestPoints(FLeftHandImg_1)
dx_lh1, dy_lh1 = lk(LeftHandImg_1, LeftHandImg_2, features_lh1, 4, 0.01, 0, 0, 50)

PlotarismaIMG(LeftHandImg_1, "OG of Left Hand image 1")
PlotarismaIMG(LeftHandImg_2, "Next Left Hand of image 2")

plt.quiver(-dx_lh1,-dy_lh1,angles='xy',scale=100)
plt.gca().invert_yaxis()
plt.title("Optical Flow for the Left Hand from frame 1 -> 2")
plt.show()

"""##############################################"""
"""FOR THE RIGHT HAND OF NOHMATRIA FROM F1 --> F2 """
"""##############################################"""

FRightHandImg_1 = RightHandImg_1.copy()
features_rh1 = justGETinterestPoints(FRightHandImg_1)
dx_rh1, dy_rh1 = lk(RightHandImg_1, RightHandImg_2, features_rh1, 4, 0.01, 0, 0, 50)

PlotarismaIMG(RightHandImg_1, "OG of Right Hand image 1")
PlotarismaIMG(RightHandImg_2, "Next Right Hand of image 2")

plt.quiver(-dx_rh1,-dy_rh1,angles='xy',scale=100)
plt.gca().invert_yaxis()
plt.title("Optical Flow for the Right Hand from frame 1 -> 2")
plt.show()

"""##############################################"""
"""############## ASKHSH 1.2 ###############TELOS"""
"""##############################################"""

"""##############################################"""
"""############### ASKHSH 1.3 ###############ARXI"""
"""##############################################"""

def EnergyCalculator(dx, dy):
    Energy = np.zeros(dx.shape)
    for i in range(0,int(dx.shape[0])):
        for j in range(0,int(dx.shape[1])):
            energy = dx[i][j]*dx[i][j] + dy[i][j]*dy[i][j]
            Energy[i,j] = energy
    return Energy

energy_f1 = EnergyCalculator(dx_f1, dy_f1)
grayIMG(energy_f1," Optical Flow Energy ")
#print(abs(np.mean(energy_f1)))

def displ(d_x, d_y):
    energeia = EnergyCalculator(d_x, d_y)
    #efarmogi krithriou
    step = np.mean(energeia) - energeia.min()
    step1 = 0.9*energeia.max()
    _, binaryEnergeia = cv2.threshold(energeia, step1, 1, cv2.THRESH_BINARY)
    
    dinter_x = []
    dinter_y = []
    grammes,sthles = energeia.shape #shape[0] --> grammes && shape[1] --> sthles
    for i in range(0,int(grammes)):
        for j in range(0,int(sthles)):
            if (binaryEnergeia[i,j]==1):
                dinter_x.append(i)
                dinter_y.append(j)
    #print(dinter_x)
    #print(dinter_y)
    sumX = 0
    for i in range(0, len(dinter_x)):
        sumX = sumX + d_x[dinter_x[i],dinter_y[i]]
    
    meanX = sumX/len(dinter_x)
    
    sumY = 0
    for i in range(0, len(dinter_y)):
        sumY = sumY + d_y[dinter_x[i],dinter_y[i]]
    
    meanY = sumY/len(dinter_y)
    
    displ_x, displ_y = meanX, meanY
    
    return displ_x, displ_y


displ_x_f1, displ_y_f1 = displ(dx_f1, dy_f1)

print(displ_x_f1, displ_y_f1)

def CheckDisp1rect(eikona1, eikona2, lista, dispX, dispY,arithmos):
    Antigrafo1 = eikona1.copy()
    Antigrafo2 = eikona2.copy()
    color = (255,0,255)
    
    cv2.rectangle(Antigrafo1, (lista[0], lista[1]), (lista[2]+lista[0], lista[3]+lista[1]), color, 3)
    PlotarismaIMG(Antigrafo1,"Previous picture "+str(arithmos))
    
    cv2.rectangle(Antigrafo2, (lista[0] - int(dispX), lista[1] - int(dispY)), (lista[2]+lista[0]- int(dispX), lista[3]+lista[1] - int(dispY)), color, 3)
    PlotarismaIMG(Antigrafo2,"NEXT frame "+str(arithmos+1))
    return None

_,RGB_2 ,_ = readImage("2")
#check the face rect
CheckDisp1rect(RGB_1, RGB_2, Proswpo_1, displ_x_f1, displ_y_f1,1)

#check the left hand rect
displ_x_lh1, displ_y_lh1 = displ(dx_lh1, dy_lh1)
CheckDisp1rect(RGB_1, RGB_2, AristeroXeri_1, displ_x_lh1, displ_y_lh1,1)

#check the right hand rect
displ_x_rh1, displ_y_rh1 = displ(dx_rh1, dy_rh1)
CheckDisp1rect(RGB_1, RGB_2, DexiXeri_1, displ_x_rh1, displ_y_rh1,1)

"""##############################################"""
"""############## ASKHSH 1.3 ###############TELOS"""
"""##############################################"""

"""#----LET'S LOAD ALL THE IMAGES INTO A LIST----#"""
Eikones_yCrCb = []
Eikones_rgb = []
Eikones_gray = []
for i in range(1,67):
    IMGyCrCb_HELP,IMGrgb_HELP ,IMGgray_HELP = readImageNoNormal(str(i))
    Eikones_yCrCb.append(IMGyCrCb_HELP)
    Eikones_rgb.append(IMGrgb_HELP)
    Eikones_gray.append(IMGgray_HELP)
    
def VideoChecker(eikona1,eikona2, lista, rho, epsilon, d_x0, d_y0):
    Antigrafo1 = eikona1.copy()
    Antigrafo2 = eikona2.copy()
    Cut_eikona1 = KopseThnEikona(Antigrafo1, lista[0], lista[1], lista[2], lista[3])
    Cut_eikona2 = KopseThnEikona(Antigrafo2, lista[0], lista[1], lista[2], lista[3])
    Feikona1 = Cut_eikona1.copy()
    features_eikona1 = justGETinterestPoints(Feikona1)
    dx, dy = lk(Cut_eikona1, Cut_eikona2, features_eikona1,  rho, epsilon, d_x0, d_y0, 10)
    displ_x, displ_y = displ(dx, dy)
    
    return displ_x, displ_y

def CheckDisp(eikona1, lista, dispX, dispY,arithmos):
    Antigrafo1 = eikona1.copy()
    color = (255,0,255)
    
    cv2.rectangle(Antigrafo1, (lista[0] - ceil(dispX), lista[1] - ceil(dispY)), (lista[2]+lista[0] - ceil(dispX), lista[3]+lista[1] - ceil(dispY)), color, 3)
    PlotarismaIMG(Antigrafo1,"Picture no "+str(arithmos))
    
    return None

def VisualAllrectIMG(IMGlista, CORlista, rho, epsilon, d_x0, d_y0):
    Displaysment_x = []
    Displaysment_y = []
    sum_disp_x = 0 
    sum_disp_y = 0
    for i in range(0,65):
        displ_x_HELP, displ_y_HELP = VideoChecker(IMGlista[i],IMGlista[i+1], CORlista, rho, epsilon, d_x0, d_y0)
        Displaysment_x.append(displ_x_HELP)
        Displaysment_y.append(displ_y_HELP)
        sum_disp_x = sum_disp_x + Displaysment_x[i] 
        sum_disp_y = sum_disp_y + Displaysment_y[i]
        CheckDisp(IMGlista[i], CORlista, sum_disp_x, sum_disp_y,i)

    return None

"""##############################################"""
""" FOR THE FACE OF NOHMATRIA FROM F i --> i+1   """
"""##############################################"""

#VisualAllrectIMG(Eikones_rgb, listaProswpo, 4, 0.01, 0, 0)

"""#################################################"""
""" FOR THE LEFT HAND OF NOHMATRIA FROM F i --> i+1 """
"""#################################################"""

#VisualAllrectIMG(Eikones_rgb, listaLeftHand, 2, 0.1, 0, 0)

"""##################################################"""
""" FOR THE RIGHT HAND OF NOHMATRIA FROM F i --> i+1 """
"""##################################################"""

#VisualAllrectIMG(Eikones_rgb, listaRightHand, 2, 0.1, 0, 0)

def aplosLK(I1, I2, features, rho, epsilon, d_x0, d_y0, repeat):

    #make the images gray 
    #I1 = cv2.cvtColor(I1, cv2.COLOR_RGB2GRAY)
    I1 = I1.astype(np.float)/255
    
    #I2 = cv2.cvtColor(I2, cv2.COLOR_RGB2GRAY)
    I2 = I2.astype(np.float)/255

    #split the features
    if (features is None):
        features1 = [0] 
        features2 = [0]
    else :
        features1, features2 = cv2.split(features)
    
    #make empty areas
    Teliko_dx = np.zeros(I1.shape)
    Teliko_dy = np.zeros(I2.shape)
    
    #for the first time
    dx = d_x0
    dy = d_y0
    
    for i in range(0,len(features1)):
        y = int(features1[i])
        x = int(features2[i])
        
        for j in range(0,repeat):
            #create the gaussian filter 
            Gr = gaussCreator(rho)
            #create the A
            A1, A2 = Acreator(I1, dx, dy)
            #print("eimai kala eimai sthn "+str(i+1)+"-osth fora")
            x0, y0 = np.meshgrid(np.arange(I1.shape[1]),np.arange(I1.shape[0]))
            #create In-1
            InMeion1 = map_coordinates(I1,[(y0+dy), (x0+dx)],order=1)
            #create the E
            E = I2 - InMeion1 
        
            #create the dependecies of matrix u2x2
            u2x2_a = cv2.filter2D((A1**2), -1, Gr) + epsilon
            u2x2_b = cv2.filter2D((A1*A2), -1, Gr) 
            u2x2_c = cv2.filter2D((A1*A2), -1, Gr) 
            u2x2_d = cv2.filter2D((A2**2), -1, Gr) + epsilon
        
            #calculate the orizousa
            detOFu2x2 = abs(u2x2_a*u2x2_d - u2x2_b*u2x2_c)
        
            #create the inverse u2x2 
            u2x2_11 = u2x2_d / detOFu2x2
            u2x2_12 = - ( u2x2_b / detOFu2x2 )
            u2x2_21 = - ( u2x2_c / detOFu2x2 )
            u2x2_22 = u2x2_a / detOFu2x2
        
            #create the dependecies of matrix u2x1
            u2x1_a = cv2.filter2D((A1*E), -1, Gr) 
            u2x1_b = cv2.filter2D((A2*E), -1, Gr)
        
            #calculate the Ufinal 
            Ufinal_1 = u2x2_11*u2x1_a + u2x2_12*u2x1_b
            Ufinal_2 = u2x2_21*u2x1_a + u2x2_22*u2x1_b 
        
            #get only the features  
            U_1 = Ufinal_1[x,y]
            U_2 = Ufinal_2[x,y]
            
            #FINALLY the dx and dy are
            dx = dx + U_1 
            dy = dy + U_2
        
        Teliko_dx[x,y] = dx
        Teliko_dy[x,y] = dy
        
    return Teliko_dx, Teliko_dy

def Pyramides(eikona, rho, scales):
    
    Antigrafo = eikona.copy()
    
    Gr = gaussCreator(rho)
    
    EikonesPyramida = []
    EikonesPyramida.append(Antigrafo)   
    for i in range(0,scales):
        emg_Help = cv2.filter2D(EikonesPyramida[i], -1, Gr)
        emg_Help = cv2.resize(emg_Help, (emg_Help.shape[1]//2, emg_Help.shape[0]//2), cv2.INTER_LINEAR_EXACT)
        EikonesPyramida.append(emg_Help)
    
    return EikonesPyramida
    
def MultiScalesLK(I1, I2, rho, epsilon, d_x0, d_y0, scales):
    
    #make the images gray 
    I1 = cv2.cvtColor(I1, cv2.COLOR_RGB2GRAY)
    #I1 = I1.astype(np.float)/255
    
    I2 = cv2.cvtColor(I2, cv2.COLOR_RGB2GRAY)
    #I2 = I2.astype(np.float)/255
    
    I1pyramida = []
    I1pyramida = Pyramides(I1, rho, scales)
    I2pyramida = []
    I2pyramida = Pyramides(I2, rho, scales)
    
    I1corners = cv2.goodFeaturesToTrack(I1pyramida[scales-1],25,0.01,10) 
    #I1corners = justGETinterestPoints(I1pyramida[scales-1])
    #I1corners = np.int0(I1corners)    
    #print(I1corners)
    
    DX1, DY1 = aplosLK(I1pyramida[scales-1],I2pyramida[scales-1], I1corners, rho, epsilon, d_x0, d_y0, 50)
    
    for i in range(scales-2,-1,-1):
        #print("EIMAI KALA EIMAI STHN FORA ",i)
        dx1 = 2*cv2.resize(DX1 , (I1pyramida[i].shape[1], I1pyramida[i].shape[0]) ,cv2.INTER_LINEAR_EXACT)
        dy1 = 2*cv2.resize(DY1 , (I1pyramida[i].shape[1], I1pyramida[i].shape[0]),cv2.INTER_LINEAR_EXACT)
        I1corners = cv2.goodFeaturesToTrack(I1pyramida[i],25,0.01,10)
        dx1, dy1 = aplosLK(I1pyramida[i],I2pyramida[i],I1corners,rho,epsilon,dx1[0][0],dy1[0][0],10)
    #print(dx1, dy1)
    return dx1, dy1

#Mult_dx_f1, Mult_dy_f1 = MultiScalesLK(img1,img2, 2, 0.1, 0, 0,4)

def VideoChecker2(eikona1, eikona2, lista, rho, epsilon, d_x0, d_y0):
    Antigrafo1 = eikona1.copy()
    Antigrafo2 = eikona2.copy()
    Cut_eikona1 = KopseThnEikona(Antigrafo1, lista[0], lista[1], lista[2], lista[3])
    Cut_eikona2 = KopseThnEikona(Antigrafo2, lista[0], lista[1], lista[2], lista[3])
    Feikona1 = Cut_eikona1.copy()
    #features_eikona1 = justGETinterestPoints(Feikona1)
    dx, dy = MultiScalesLK(Cut_eikona1, Cut_eikona2, rho, epsilon, d_x0, d_y0, 4)
    displ_x, displ_y = displ(dx, dy)
    
    return displ_x, displ_y


def CheckDisp2(eikona1, lista, dispX, dispY,arithmos):
    Antigrafo1 = eikona1.copy()
    color = (255,0,255)
    
    cv2.rectangle(Antigrafo1, (lista[0] - ceil(dispX), lista[1] - ceil(dispY)), (lista[2]+lista[0] - ceil(dispX), lista[3]+lista[1] - ceil(dispY)), color, 3)
    PlotarismaIMG(Antigrafo1,"Picture no "+str(arithmos))
    
    return None

def VisualAllrectIMG2(IMGlista, CORlista, rho, epsilon, d_x0, d_y0):
    Displaysment_x = []
    Displaysment_y = []
    sum_disp_x = 0 
    sum_disp_y = 0
    for i in range(0,65):
        displ_x_HELP, displ_y_HELP = VideoChecker2(IMGlista[i],IMGlista[i+1], CORlista, rho, epsilon, d_x0, d_y0)
        Displaysment_x.append(displ_x_HELP)
        Displaysment_y.append(displ_y_HELP)
        sum_disp_x = sum_disp_x + Displaysment_x[i] 
        sum_disp_y = sum_disp_y + Displaysment_y[i]
        CheckDisp2(IMGlista[i], CORlista, sum_disp_x, sum_disp_y, i+1)

    return None

#VisualAllrectIMG2(Eikones_rgb, listaProswpo, 2, 0.1, 0, 0)

"""##############################################"""
""" FOR THE FACE OF NOHMATRIA FROM F i --> i+1   """
"""##############################################"""

#VisualAllrectIMG2(Eikones_rgb, listaProswpo, 4, 0.1, 0, 0)

"""#################################################"""
""" FOR THE LEFT HAND OF NOHMATRIA FROM F i --> i+1 """
"""#################################################"""

#VisualAllrectIMG2(Eikones_rgb, listaLeftHand, 2, 0.1, 0, 0)

"""##################################################"""
""" FOR THE RIGHT HAND OF NOHMATRIA FROM F i --> i+1 """
"""##################################################"""

#VisualAllrectIMG2(Eikones_rgb, listaRightHand, 4, 0.01, 0, 0)

"""#################################################"""
"""############### TELOS ASKHSHS ###################"""
"""#################################################"""


