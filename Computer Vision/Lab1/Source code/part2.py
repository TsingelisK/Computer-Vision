# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 19:32:12 2021

@authors: Panagiotis Gardelis & Konstantinos Spathis
"""

import cv2 
import matplotlib.pyplot as plt
import numpy as np
from cv21_lab1_part2_utils import interest_points_visualization, disk_strel
#from matplotlib import rcParams

"""############### ASKHSH 2.1 ################"""

"""############### LOADING IMAGES ############"""

#reading the image blood_smear
IMGbloodsmear = cv2.imread("blood_smear.jpg", cv2.IMREAD_COLOR) 
print("Resolution : " , IMGbloodsmear.shape, "pixels")

IMGbloodsmear = cv2.cvtColor(IMGbloodsmear, cv2.COLOR_BGR2RGB)
IMGbloodsmearRGB = IMGbloodsmear

#if no cmap argument given, the default is color image 
plt.imshow(IMGbloodsmear) 
plt.title("Original Image of blood_smear")
plt.show()

IMGbloodsmear = cv2.cvtColor(IMGbloodsmear, cv2.COLOR_RGB2GRAY)

#normalize from 0-255 to [0,1]
#metatropi eikonas se float
IMGbloodsmear = IMGbloodsmear.astype(np.float) / 255
print("NEW Range : {} - {} ".format(IMGbloodsmear.min(),IMGbloodsmear.max()))

#reading the image mars
IMGmars = cv2.imread("mars.png", cv2.IMREAD_COLOR) 
print("Resolution : " , IMGmars.shape, "pixels")

IMGmars = cv2.cvtColor(IMGmars, cv2.COLOR_BGR2RGB)
IMGmarsRGB = IMGmars

#if no cmap argument given, the default is color image 
plt.imshow(IMGmars) 
plt.title("Original Image of mars")
plt.show()

IMGmars = cv2.cvtColor(IMGmars, cv2.COLOR_RGB2GRAY)

#normalize from 0-255 to [0,1]
#metatropi eikonas se float
IMGmars = IMGmars.astype(np.float) / 255
print("NEW Range : {} - {} ".format(IMGmars.min(),IMGmars.max()))

#reading the urban_edges 
IMGurbanedges  = cv2.imread("urban_edges.jpg", cv2.IMREAD_COLOR) 
print("Resolution : " , IMGurbanedges.shape, "pixels")

IMGurbanedges = cv2.cvtColor(IMGurbanedges, cv2.COLOR_BGR2RGB)
IMGurbanedgesRGB = IMGurbanedges
#if no cmap argument given, the default is color image 
plt.imshow(IMGurbanedges) 
plt.title("Original Image of urban_edges")
plt.show()

IMGurbanedges = cv2.cvtColor(IMGurbanedges, cv2.COLOR_RGB2GRAY)

#normalize from 0-255 to [0,1]
#metatropi eikonas se float
IMGurbanedges = IMGurbanedges.astype(np.float) / 255
print("NEW Range : {} - {} ".format(IMGurbanedges.min(),IMGurbanedges.max()))

#noisy urban_edges image with scale = 2
IMGurbanedgesWithNoise = IMGurbanedges + np.random.normal(0, 2, IMGurbanedges.shape)

#plotarisma noisy image urban_edges
plt.imshow(IMGurbanedgesWithNoise, cmap="gray") 
plt.title("Noisy Image of urban_edges")
plt.show()

def aploPlotarismaMeIMSHOW(eikona, Sxolia, onomaEikonas):
    plt.imshow(eikona, cmap = 'gray') 
    plt.title(Sxolia + onomaEikonas)
    plt.show()
    return None

"""############### CREATION OF J TANYSTES ############"""
##The followig functions were created becaused we have 3 images 
##we want to create their tensioner

##Function to create the 2 gaussian filters Gs Gr
def creatorOfGaussian(eisodos):
    n = int(2*np.ceil(3*eisodos)+1)
    gauss1D = cv2.getGaussianKernel(n, eisodos)
    gauss2D = gauss1D @ gauss1D.T 
    return gauss2D

##Function to filter image with Gs
def eikonaMeGaussian(eikona, gauss, kriths):
    #creation of eikona with gaussian with convolution
    eikonaMeGauss = cv2.filter2D(eikona, -1, gauss)
    if(kriths == "IMGurbanedges"):
       onomaEikonas1 = "urban_edges"
    if(kriths == "IMGmars"):
       onomaEikonas1 = "mars"
    if(kriths == "IMGbloodsmear"):
       onomaEikonas1 = "blood_smear"
    #gurise thn eikona me Gauss
    return eikonaMeGauss, onomaEikonas1

##Function to create the multiplied gradients of an image
##so that they can be convolved with the gaussian filter r
def gradientCreator(eikonaMeGayss):
    #creation of the gradient y 
    eikonaGradientY = np.gradient(eikonaMeGayss,axis=1)
    #creation of the gradient x 
    eikonaMeGradientX = np.gradient(eikonaMeGayss,axis=0)
    #multiply gradients
    gradientXY = eikonaMeGradientX * eikonaGradientY
    gradientXX = eikonaMeGradientX * eikonaMeGradientX
    gradientYY = eikonaGradientY * eikonaGradientY
    #gyrise ta gradient sthn exodo
    return gradientXX, gradientXY,gradientYY 

def tanysthsCreator(gauss, gradient):
    #J = cv2.filter2D(gauss, -1, gradient)
    J = cv2.filter2D(gradient, -1, gauss)
    return J

def LAMDA(J1, J2, J3, kriths):
    J1subJ3 = np.subtract(J1, J3)
    oros1 = J1subJ3 * J1subJ3
    oros2 = 4*(J2 * J2)
    #oros1 = J1subJ3**2
    #oros2 = 4*(J2**2)
    yporizo = np.add(oros1,oros2)
    riza = np.sqrt(yporizo)
    if(kriths == 1): #gia thn prosthesh
        lamda = (J1 + J3 + riza)/2
    if(kriths == -1): #gia thn afairesh
        lamda = (J1 + J3 - riza)/2
    return lamda

def creatorOfR(lamdaThetiko,lamdaArnhtiko):
    k=0.05
    oros1 = lamdaThetiko * lamdaArnhtiko
    #oros1 = np.dot(lamdaThetiko,lamdaArnhtiko)
    athroisma = np.add(lamdaThetiko, lamdaArnhtiko)
    oros2 = athroisma * athroisma
    #oros2 = athroisma**2
    R = oros1 -k*oros2
    #R = np.subtract(oros1, k*oros2)
    return R 

def HessianCoeffs(eikona, scale):
    Gscale = creatorOfGaussian(scale) 
    eikonaMeGayss = cv2.filter2D(eikona, -1, Gscale)
    #creation of the gradient y 
    Ly = np.gradient(eikonaMeGayss,axis=1)
    Lyy = np.gradient(Ly,axis=1)
    #creation of the gradient x 
    Lx = np.gradient(eikonaMeGayss,axis=0)
    Lxx = np.gradient(Lx,axis=0)
    #creation of Lxy
    Lxy = np.gradient(Lx,axis=1)
    return Lxx, Lxy, Lyy

"""############### CREATION OF Harris - Stephens ############"""

def Harris(eikona, s, r, onoma, krithsRETURN, krithsHarris):
    Gs = creatorOfGaussian(s) #scale = 2 
    Gr = creatorOfGaussian(r) # r = 2.5
    
    if(krithsHarris == "Stephens"):
        IS,onomaEikonas2 = eikonaMeGaussian(eikona, Gs, onoma)
        gradientXX, gradientXY, gradientYY = gradientCreator(IS)
        J1 = tanysthsCreator(Gr,gradientXX)
        J2 = tanysthsCreator(Gr,gradientXY)
        J3 = tanysthsCreator(Gr,gradientYY)
        lamdaThetiko = LAMDA(J1, J2, J3, 1)
        lamdaArnhtiko = LAMDA(J1, J2, J3, -1)
        R = creatorOfR(lamdaThetiko,lamdaArnhtiko)
        #condition S1
        #sigmaS1 = 1.5
        ns = np.ceil(3*s)*2+1
        B_sq = disk_strel(ns)
        RDilation = cv2.dilate(R,B_sq)
        megisto = R.max()
        thetacorn = 0.005
        eikonaTelikh = cv2.bitwise_and(np.where(R==RDilation,1,0),np.where(R>thetacorn*megisto,1,0))
        numberOfGwnies = 0
        grammes, sthles = eikonaTelikh.shape
        for i in range(0,int(grammes)):
            for j in range(0,int(sthles)):
                if (eikonaTelikh[i, j] == 1):
                    numberOfGwnies = numberOfGwnies +1
                    
    elif(krithsHarris == "Hessian"):
        Lxx, Lxy, Lyy = HessianCoeffs(eikona, s)
        R = Lxx*Lyy -Lxy*Lxy
        megisto = R.max()
        #R = np.linalg.det(Hessian_matrix)
        #condition S1
        #sigmaS1 = 1.5
        ns = np.ceil(3*s)*2+1
        B_sq = disk_strel(ns)
        RDilation = cv2.dilate(R,B_sq)
        #megisto = R.max()
        thetacorn = 0.005
        eikonaTelikh = cv2.bitwise_and(np.where(R==RDilation,1,0),np.where(R>thetacorn*megisto,1,0))
        numberOfGwnies = 0
        grammes, sthles = eikonaTelikh.shape
        for i in range(0,int(grammes)):
            for j in range(0,int(sthles)):
                if (eikonaTelikh[i, j] == 1):
                    numberOfGwnies = numberOfGwnies +1
        
    if(krithsRETURN == 0):
        return eikonaTelikh, numberOfGwnies, R, onomaEikonas2, J1, J2, J3, lamdaThetiko, lamdaArnhtiko
    elif(krithsRETURN == 1):
         return eikonaTelikh, numberOfGwnies
     
    return None

def visualMONOs(eikona, scale):
    correctINPUT = []
    grammes, sthles = eikona.shape
    for i in range(0,int(grammes)):
        for j in range(0,int(sthles)):
            if (eikona[i, j] == 1):
                correctINPUT.append(j)
                correctINPUT.append(i)
                correctINPUT.append(scale)    
        correctIN = np.reshape(correctINPUT,(len(correctINPUT)//3,3)) 
    return correctIN

def plotarismaHarrisStephens(NewR,number, R, onomaEikonas, J1, J2, J3, lamdaThetiko, lamdaArnhtiko):
    #plotarisma tanysth J1
    plt.imshow(J1, cmap = 'gray') 
    plt.title("Tanysths J1 for " + onomaEikonas)
    plt.show()

    #plotarisma tanysth J2
    plt.imshow(J2, cmap = 'gray') 
    plt.title("Tanysths J2 for " + onomaEikonas)
    plt.show()

    #plotarisma tanysth J3
    plt.imshow(J3, cmap = 'gray') 
    plt.title("Tanysths J3 for " + onomaEikonas)
    plt.show()

    #plotarisma lamdaThetikourbanedges
    plt.imshow(lamdaThetiko, cmap="gray") 
    plt.title("Thetiko Lamda for " + onomaEikonas)
    plt.show()

    #plotarisma lamdaThetikourbanedges
    plt.imshow(lamdaArnhtiko, cmap="gray") 
    plt.title("Negative Lamda for " + onomaEikonas)
    plt.show()

    #plotarisma R
    plt.imshow(R, cmap="gray") 
    plt.title("R before edit " + onomaEikonas)
    plt.show()

    #plotarisma R
    plt.imshow(NewR, cmap = 'gray') 
    plt.title("NEW R apo Harris Stephens for " + onomaEikonas)
    plt.show()

    print("To plhthos twn gwniwn endiaforntos ths {} einai: {}".format(onomaEikonas,str(number)))
    return None

"""### For the urban_edges ###"""
NEWRu,numberu, Reikonasu, onomaEikonasu,J1eikonasu, J2eikonasu, J3eikonasu, lamdaThetikoeikonasu, lamdaArnhtikoeikonasu = Harris(IMGurbanedges, 2, 2.5, "IMGurbanedges",0,"Stephens")
plotarismaHarrisStephens(NEWRu,numberu, Reikonasu, onomaEikonasu,J1eikonasu, J2eikonasu, J3eikonasu, lamdaThetikoeikonasu, lamdaArnhtikoeikonasu)
print("To plhthos twn gwniwn endiaforntos ths urban_edges einai: %d" %numberu)
    
"""### For the mars ###"""
NEWRmars,numbermars, Reikonasmars, onomaEikonasmars,J1eikonasmars, J2eikonasmars, J3eikonasmars, lamdaThetikoeikonasmars, lamdaArnhtikoeikonasmars = Harris(IMGmars, 2, 2.5, "IMGmars",0,"Stephens")
#plotarismaHarrisStephens(NEWRmars,numbermars, Reikonasmars, onomaEikonasmars,J1eikonasmars, J2eikonasmars, J3eikonasmars, lamdaThetikoeikonasmars, lamdaArnhtikoeikonasmars)
print("To plhthos twn gwniwn endiaforntos ths mars einai:  %d" %numbermars)
    
"""### For the blood_smears ###"""
NEWRbloodsmear,numberbloodsmear, Reikonasbloodsmear, onomaEikonasbloodsmear,J1eikonasbloodsmear, J2eikonasbloodsmear, J3eikonasbloodsmear, lamdaThetikoeikonasbloodsmear, lamdaArnhtikoeikonasbloodsmear = Harris(IMGbloodsmear, 2, 2.5, "IMGbloodsmear",0,"Stephens")
#plotarismaHarrisStephens(NEWRbloodsmear,numberbloodsmear, Reikonasbloodsmear, onomaEikonasbloodsmear,J1eikonasbloodsmear, J2eikonasbloodsmear, J3eikonasbloodsmear, lamdaThetikoeikonasbloodsmear, lamdaArnhtikoeikonasbloodsmear)
print("To plhthos twn gwniwn endiaforntos ths blood_smears einai: %d" %numberbloodsmear)

#VISUALIZATION OF GWNIES FOR IMAGE URBAN EDGES 
#N3UB = Nx3 array tou urban edges
N3UB = visualMONOs(NEWRu, 2)
interest_points_visualization(IMGurbanedgesRGB, N3UB)
plt.title("Visualization of gwnies for urban edges image for s = 2")
plt.show()

#VISUALIZATION OF GWNIES FOR IMAGE MARS 
#N3M = Nx3 array tou mars
N3M = visualMONOs(NEWRmars, 2)
interest_points_visualization(IMGmarsRGB, N3M)
plt.title("Visualization of gwnies for mars image for s = 2")
plt.show()

#VISUALIZATION OF GWNIES FOR IMAGE BLOOD SMEAR 
#N3BS = Nx3 array tou blood smear
N3BS = visualMONOs(NEWRbloodsmear, 2)
interest_points_visualization(IMGbloodsmearRGB, N3BS)
plt.title("Visualization of gwnies for blood smear image for s = 2")
plt.show()

"""############### ASKHSH 2.2 ################"""

def scaleANDr(N):
    sigma = 1.5
    scale = (sigma**N)*2
    r = (sigma**N)*2.5
    return scale, r

def normalLoGs(eikona, scale):
    Gscale = creatorOfGaussian(scale) 
    eikonaMeGayss = cv2.filter2D(eikona, -1, Gscale)
    #eikonaMeGayss = cv2.Laplacian(eikonaMeGayss,cv2.CV_64FC1,3)
    #creation of the gradient y 
    Ly = np.gradient(eikonaMeGayss,axis=1)
    Lyy = np.gradient(Ly,axis=1)
    #creation of the gradient x 
    Lx = np.gradient(eikonaMeGayss,axis=0)
    Lxx = np.gradient(Lx,axis=0)
    #eikonaLoG = cv2.Laplacian(eikona,cv2.CV_64FC1,3)
    eikonaLoG = np.abs((scale**2)*(np.abs(Lxx + Lyy)))
    return eikonaLoG

def PolyklimatesGwnies(eikona, N, onoma):
    imageLista = [] 
    sLista = []
    rLista = []
    imageLoGLista = []
    for i in range(0,N):
        s, r = scaleANDr(i)
        sLista.append(s)
        rLista.append(r)
        image1, numberOfShmeia = Harris(eikona, s, r, onoma, 1,"Stephens")
        imageLista.append(image1)
        image2 = eikona
        imageLoG = normalLoGs(image2, s)
        imageLoGLista.append(imageLoG)
        #plotarisma
        #plt.imshow(imageLoGLista[i], cmap = 'gray') 
        #plt.title("LoG {} of image {}".format(onoma,(i+1)))
        #plt.show()

    for i in range(0,N):
        if(i == 0):
            krithrio1 = cv2.bitwise_and(np.where(imageLista[i]==1,1,0),np.where(imageLoGLista[i] > imageLoGLista[i+1],1,0))   
            eikonaMeLoG = krithrio1  
        elif(i == 1):
            krithrio2 = cv2.bitwise_and(np.where(imageLista[i]==1,1,0),np.where(imageLoGLista[i] > imageLoGLista[i-1],1,0),np.where(imageLoGLista[i] > imageLoGLista[i+1],1,0))
            eikonaMeLoG = krithrio2
        elif(i == 2):
            krithrio3 = cv2.bitwise_and(np.where(imageLista[i]==1,1,0),np.where(imageLoGLista[i] > imageLoGLista[i-1],1,0),np.where(imageLoGLista[i] > imageLoGLista[i+1],1,0))
            eikonaMeLoG = krithrio3
        else:
            krithrio4 = cv2.bitwise_and(np.where(imageLista[i]==1,1,0),np.where(imageLoGLista[i] > imageLoGLista[i-1],1,0))
            eikonaMeLoG = krithrio4
    
    #put all the krithria in one list
    ALLkrithria = []
    ALLkrithria.append(krithrio1)
    ALLkrithria.append(krithrio2)
    ALLkrithria.append(krithrio3)
    ALLkrithria.append(krithrio4)
    return eikonaMeLoG, ALLkrithria

#PAG = Poluklimakwth Anixneush Gwniwn & NOG3D = number Of Gwnies 3D
PAGurbanedges, ALLkrithriaGwniwnurbanedges = PolyklimatesGwnies(IMGurbanedges, 4, "IMGurbanedges")

#eyresh poluklimakwths anixneushs gwniwn gia thn mars image
#PAG = Poluklimakwth Anixneush Gwniwn & NOG3D = number Of Gwnies 3D
PAGmars, ALLkrithriaGwniwnmars = PolyklimatesGwnies(IMGmars, 4, "IMGmars")
#print("To plhthos twn gwniwn endiaforntos ths mars gia poluklimakwths anixneush gwniwn: " + str(NOG3Dmars))

plt.imshow(PAGmars, cmap = 'gray') 
plt.title("Poluklimakwth Anixneush Gwniwn for " + str(onomaEikonasmars))
plt.show()

#eyresh poluklimakwths anixneushs gwniwn gia thn bloodsmear image
#PAG = Poluklimakwth Anixneush Gwniwn & NOG3D = number Of Gwnies 3D
PAGbloodsmear, ALLkrithriabloodsmear = PolyklimatesGwnies(IMGbloodsmear, 4, "IMGbloodsmear")
#print("To plhthos twn gwniwn endiaforntos ths bloodsmear gia poluklimakwths anixneush gwniwn: " + str(NOG3Dbloodsmear))

plt.imshow(PAGbloodsmear, cmap = 'gray') 
plt.title("Poluklimakwth Anixneush Gwniwn for " + str(onomaEikonasbloodsmear))
plt.show()

def properArgsVisual(ALLkrithria):
    correctINPUT = []
    elements = []
    sLista = []
    NLista = [0, 1, 2, 3]
    for i in range(0,len(NLista)):
        s, r = scaleANDr(i)
        sLista.append(s)
    counterInterest = 0
    for i in range(0,4):
        correctINPUT.clear()
        pinakas =  ALLkrithria[i]
        grammes,sthles = pinakas.shape
        for row in range(0,int(grammes)):
            for column in range(0,int(sthles)):
                if(pinakas[row, column] == 1):
                    correctINPUT.append(column)
                    correctINPUT.append(row)
                    correctINPUT.append(sLista[i])
                    #correctINPUT.append(NLista[i]+1)
                    counterInterest = counterInterest +1
        correctIN = np.reshape(correctINPUT,(len(correctINPUT)//3,3)) 
        #print(correctIN)
        elements.append(correctIN)
    correct = np.concatenate((elements[0], elements[1], elements[2], elements[3]),axis=0)
    #print(correct)
    return correct, counterInterest

#VISUALIZATION OF GWNIES FOR IMAGE URBAN EDGES 
correctINurbanedges, counterGwniesurbanedges = properArgsVisual(ALLkrithriaGwniwnurbanedges)
interest_points_visualization(IMGurbanedgesRGB, correctINurbanedges)
plt.title("Visualization of gwnies for urban edges image")
plt.show()
print("NEW To plhthos twn gwniwn endiaforntos ths urban edges gia poluklimakwths anixneush gwniwn: " + str(counterGwniesurbanedges))

#VISUALIZATION OF GWNIES FOR IMAGE MARS
correctINmars, counterGwniesmars = properArgsVisual(ALLkrithriaGwniwnmars)
interest_points_visualization(IMGmarsRGB, correctINmars)
plt.title("Visualization of gwnies for mars image")
plt.show()
print("NEW To plhthos twn gwniwn endiaforntos ths mars gia poluklimakwths anixneush gwniwn: " + str(counterGwniesmars))

#VISUALIZATION OF GWNIES FOR IMAGE BLOODSMEAR
correctINbloodsmear, counterGwniesbloodsmear = properArgsVisual(ALLkrithriabloodsmear)
interest_points_visualization(IMGbloodsmearRGB, correctINbloodsmear)
plt.title("Visualization of gwnies for blood smear image")
plt.show()
print("NEW To plhthos twn gwniwn endiaforntos ths bloodsmear gia poluklimakwths anixneush gwniwn: " + str(counterGwniesbloodsmear))

"""############### ASKHSH 2.3 ################"""

#Hessian coeffs for the mars 
Rblopsmars, nunmberBlopsmars = Harris(IMGmars,2,2.5,"IMGmars",1,"Hessian")

plt.imshow(Rblopsmars, cmap = 'gray') 
plt.title("The Blops of image mars")
plt.show()

print("The number of Blops of mars is: " + str(nunmberBlopsmars)) 

#Hessian coeffs for the bloodsmear 
Rblopsbloodsmear, nunmberBlopsbloodsmear = Harris(IMGbloodsmear,2,2.5,"IMGbloodsmear",1,"Hessian")

plt.imshow(Rblopsbloodsmear, cmap = 'gray') 
plt.title("The Blops of image blood smear")
plt.show()

print("The number of Blops of bloodsmear is: " + str(nunmberBlopsbloodsmear)) 

#VISUALIZATION OF BLOBS FOR IMAGE MARS 
#N3M = Nx3 array tou mars
N3Mblobs = visualMONOs(Rblopsmars, 2)
interest_points_visualization(IMGmarsRGB, N3Mblobs)
plt.title("Visualization of blobs for mars image for s = 2")
plt.show()

#VISUALIZATION OF BLOBS FOR IMAGE BLOOD SMEAR 
#N3BS = Nx3 array tou blood smear
N3BSblobs = visualMONOs(Rblopsbloodsmear, 2)
interest_points_visualization(IMGbloodsmearRGB, N3BSblobs)
plt.title("Visualization of blobs for blood smear image for s = 2")
plt.show()

"""############### ASKHSH 2.4 ################"""

def PolyklimataBlops(eikona, N, onoma):
    imageLista = [] 
    sLista = []
    rLista = []
    imageLoGLista = []
    for i in range(0,N):
        s, r = scaleANDr(i)
        sLista.append(s)
        rLista.append(r)
        image1, numberOfShmeia = Harris(eikona, s, r, onoma, 1,"Hessian")
        imageLista.append(image1)
        image2 = eikona
        imageLoG = normalLoGs(image2, s)
        imageLoGLista.append(imageLoG)
        #print(image1.shape)
        #print(imageLoG.shape)
        #plotarisma
        #plt.imshow(imageLoGLista[i], cmap = 'gray') 
        #plt.title("LoG {} of image {}".format(onoma,(i+1)))
        #plt.show()

    for i in range(0,N):
        if(i == 0):
            krithrio1 = cv2.bitwise_and(np.where(imageLista[i]==1,1,0),np.where(imageLoGLista[i] > imageLoGLista[i+1],1,0))   
            eikonaMeLoG = krithrio1  
        elif(i == 1):
            krithrio2 = cv2.bitwise_and(np.where(imageLista[i]==1,1,0),np.where(imageLoGLista[i] > imageLoGLista[i-1],1,0),np.where(imageLoGLista[i] > imageLoGLista[i+1],1,0))
            eikonaMeLoG = krithrio2
        elif(i == 2):
            krithrio3 = cv2.bitwise_and(np.where(imageLista[i]==1,1,0),np.where(imageLoGLista[i] > imageLoGLista[i-1],1,0),np.where(imageLoGLista[i] > imageLoGLista[i+1],1,0))
            eikonaMeLoG = krithrio3
        else:
            krithrio4 = cv2.bitwise_and(np.where(imageLista[i]==1,1,0),np.where(imageLoGLista[i] > imageLoGLista[i-1],1,0))
            eikonaMeLoG = krithrio4
    
    #put all the krithria in one list
    ALLkrithria = [krithrio1, krithrio2, krithrio3, krithrio4 ]
    
    return eikonaMeLoG, ALLkrithria

#eyresh poluklimakwths anixneushs blops gia thn mars image
#PAB = Poluklimakwth Anixneush Blops & NOB3D = number Of Blops 3D
#PABmars, NOB3Dmars, ALLkrithriaBlopsmars = PolyklimataBlops(IMGmars, 4,"IMGmars")
PABmars, ALLkrithriaBlopsmars = PolyklimataBlops(IMGmars, 4,"IMGmars")
#print("To plhthos twn blops ths mars endiaforntos gia poluklimakwths anixneush blops : " + str(NOB3Dmars))

plt.imshow(PABmars, cmap = 'gray') 
plt.title("Poluklimakwth Anixneush Blops for mars")
plt.show()

#eyresh poluklimakwths anixneushs blops gia thn blood smear image
#PAB = Poluklimakwth Anixneush Blops & NOB3D = number Of Blops 3D
#PABbloodsmear, NOB3Dbloodsmear, ALLkrithriaBlopsbloodsmear = PolyklimataBlops(IMGbloodsmear, 4,"IMGbloodsmear")
PABbloodsmear, ALLkrithriaBlopsbloodsmear = PolyklimataBlops(IMGbloodsmear, 4,"IMGbloodsmear")
#print("To plhthos twn blops ths bloodsmear endiaforntos gia poluklimakwths anixneush blops : " + str(NOB3Dbloodsmear))

plt.imshow(PABbloodsmear, cmap = 'gray') 
plt.title("Poluklimakwth Anixneush Blops for bloodsmear")
plt.show()

#VISUALIZATION OF BLOPS FOR IMAGE MARS
correctINmarsBlops, counterBlopsmars = properArgsVisual(ALLkrithriaBlopsmars)
interest_points_visualization(IMGmarsRGB, correctINmarsBlops)
plt.title("Visualization of Blops for mars image")
plt.show()
print("NEW To plhthos twn blops ths mars endiaforntos gia poluklimakwths anixneush blops : " + str(counterBlopsmars))

#VISUALIZATION OF BLOPS FOR IMAGE BLOODSMEAR
correctINbloodsmearBlops, counterBlopsbloodsmear = properArgsVisual(ALLkrithriaBlopsbloodsmear)
interest_points_visualization(IMGbloodsmearRGB, correctINbloodsmearBlops)
plt.title("Visualization of Blops for blood smear image")
plt.show()
print("NEW To plhthos twn blops ths bloodsmear endiaforntos gia poluklimakwths anixneush blops : " + str(counterBlopsbloodsmear))

"""############### ASKHSH 2.5 ################"""

"""YLOPOIHSH APLWS ME TYPO XWRIS CUMSUM
def eikonaCreator(eikona):
    grammes, sthles = eikona.shape
    newEikona = np.empty((grammes, sthles))
    return newEikona

def OloklhrwtikhCreator(emptyEikona, eikona):
    S = emptyEikona
    I = eikona
    grammes, sthles = eikona.shape
    
    for i in range(0,grammes):
        S[i,0] = 0
    
    for j in range(0,sthles):
        S[0,j] = 0
        
    for i in range(1,int(grammes)):
        for j in range(1,int(sthles)):
            S[i,j] = I[i,j] - S[i-1,j-1] + S[i-1,j] + S[i,j-1]
           # S[i,j] = I[i,j] - np.cumsum(S,axis=0) + S[i-1,j] + S[i,j-1]
    return S

#euresh oloklhrwtikhs eikonas gia to image mars
emptyEikonamars = eikonaCreator(IMGmars)
Smars = OloklhrwtikhCreator(emptyEikonamars, IMGmars)

plt.imshow(Smars, cmap='gray') 
plt.title("Oloklhrwtikh eikona for mars image")
plt.show()

#euresh oloklhrwtikhs eikonas gia to image blood smear
emptyEikonabloodsmear = eikonaCreator(IMGbloodsmear)
Sbloodsmear = OloklhrwtikhCreator(emptyEikonabloodsmear, IMGbloodsmear)

plt.imshow(Sbloodsmear, cmap='gray') 
plt.title("Oloklhrwtikh eikona for blood smear image")
plt.show()
"""

""" YLOPOIHSH CUMSUM """
def multidim_cumsum(eikona, scale):
    gauss = creatorOfGaussian(scale)
    eikonaMEgauss = cv2.filter2D(eikona, -1, gauss)
    eikonaMEgauss[:,0] = 0
    eikonaMEgauss[0,:] = 0
    out = eikonaMEgauss.cumsum(-1)
    for i in range(2,eikonaMEgauss.ndim+1):
        np.cumsum(out, axis=-i, out=out)
    return out

#euresh oloklhrwtikhs eikonas gia to image mars
Smars = multidim_cumsum(IMGmars, 2)
plt.imshow(Smars, cmap = 'gray') 
plt.title("Integral image for mars")
plt.show()

#euresh oloklhrwtikhs eikonas gia to image blood smear
Sbloodsmear = multidim_cumsum(IMGbloodsmear, 2)
plt.imshow(Sbloodsmear, cmap = 'gray') 
plt.title("Integral image for bloodsmear")
plt.show()

""" BOX FILTERS VISUALIZATION"""
def boxFilterCreator(scale):
    n = int(2*np.ceil(3*scale)+1)
    megethos2 = int(2*(np.floor(n/6)) + 1)
    megethos4 = int(4*(np.floor(n/6)) + 1)
    Dxx1 = np.full((megethos4,megethos2),1)
    Dxx2 = np.full((megethos4,megethos2),(-2)) 
    DxxHelp = np.full((int((3*megethos2 - megethos4)/2), 3*megethos2),0)
    midDxx = np.concatenate((Dxx1, Dxx2, Dxx1),axis = 1)
    telikoDxx = np.concatenate((DxxHelp, midDxx, DxxHelp),axis = 0)
    return telikoDxx

Dxx = boxFilterCreator(2)
Dyy = Dxx.T

"""
plt.imshow(Dxx, cmap = 'gray') 
plt.title("Box filter of Dxx")
plt.show()
plt.imshow(Dxx.T, cmap = 'gray') 
plt.title("Box filter of Dyy")
plt.show()
"""

def eikonaCreator(eikona):
    grammes, sthles = eikona.shape
    newEikona = np.empty((grammes, sthles))
    return newEikona

def boxCalculator(arxikhEikona, scale, pinakas):  
    I = arxikhEikona
    S = multidim_cumsum(arxikhEikona, scale)
    grammes,sthles = arxikhEikona.shape
    for i in range(1,int(grammes)):
        for j in range(1,int(sthles)): 
            I[i,j] = S[pinakas[0,0],pinakas[0,1]] + S[pinakas[2,0],pinakas[2,1]] - S[pinakas[1,0],pinakas[1,1]] - S[pinakas[3,0],pinakas[3,1]]
    return 

#2.5.2
def Mon_sin (arxikhEikona, height , width):
    #oloklirotiki eikona
    S = np.cumsum(np.cumsum(arxikhEikona, axis=0), axis=1)
    #bazw pad up,down,right,left
    S = np.pad(S, ((height,0), (width,0)) )
    S = np.pad(S, ((0,height), (0,width)),  'edge')
    
    #kentro filtrou
    w = int(np.floor(width/2))
    h = int(np.floor(height/2))
   
    #gwnies tou filtrou 
    #up left
    Ul = np.roll(np.roll(S, -(w+1), axis=1), -(h+1), axis=0)
    #down right
    Dr = np.roll(np.roll(S,  w, axis=1),   h, axis=0)
    #up right
    Ur = np.roll(np.roll(S,  w, axis=1),  -(h+1), axis=0)
    #down left
    Dl = np.roll(np.roll(S,  -(w+1), axis=1), h, axis=0)

    embadon = Ul + Dr - Ur - Dl
    out = embadon[height:,width:][:-height,:-width]
    return out

def BMaker(eikona,scale):
    gauss = creatorOfGaussian(scale)
    eikonaMEgauss = cv2.filter2D(eikona, -1, gauss)
#calculate the n with the given scale
    n = int(2*np.ceil(3*scale)+1)
#for Dxx
    xx_w = int(2*(np.floor(n/6)) + 1)
    xx_h = int(4*(np.floor(n/6)) + 1)
    Lxx1 = Mon_sin(eikonaMEgauss, xx_h, xx_w)
    #pad Lxx
    Lxx = np.pad(Lxx1, ((0,0), (xx_w,xx_w)))
    Lxx = -2*Lxx + np.roll(Lxx, xx_w ,axis=1) + np.roll(Lxx, -xx_w, axis=1)
    #unpad Lxx
    Lxx3 = Lxx[:, xx_w:-xx_w]
    
#for Dyy
    yy_w = int(4*(np.floor(n/6)) + 1)
    yy_h = int(2*(np.floor(n/6)) + 1)
    Lyy1 = Mon_sin(eikonaMEgauss, yy_h, yy_w)
    #pad Lyy
    Lyy = np.pad(Lyy1, ((yy_h,yy_h),(0,0) ))
    Lyy = -2*Lyy + np.roll(Lyy, yy_h ,axis=0) + np.roll(Lyy, -yy_h, axis=0)
    #unpad Lyy
    Lyy3 = Lyy[yy_h:-yy_h, :]
    
#for Dxy
    xy_w = int(2*(np.floor(n/6)) + 1)
    xy_h = int(2*(np.floor(n/6)) + 1)
    Lxy1 = Mon_sin(eikonaMEgauss, xy_h, xy_w)
    #pad Lxy
    Lxy = np.pad(Lxy1, ((xy_h+1,xy_h+1),(xy_w+1,xy_w+1) ))
    #kentro filtrou Dxy
    w = int(np.floor(xy_w/2))
    h = int(np.floor(xy_h/2))
   

    #up_left
    Ul = np.roll(Lxy, [(h+1), (w+1)] , axis=(0,1) )
    #down right
    Dr = np.roll(Lxy, [-(h+1), -(w+1)], axis=(0,1))
    #up right
    Ur = np.roll(Lxy, [h+1, -(w+1)], axis=(0,1))
    #down left
    Dl = np.roll(Lxy, [-(h+1), w+1], axis=(0,1))
         
    
    Lxy = Ul + Dr - Ur - Dl
    #unpad Lxy
    Lxy3 = Lxy[(xy_h+1):-(xy_h+1), (xy_w+1):-(xy_w+1)]
    
    return (Lxx3 , Lyy3 , Lxy3) 
    
LDxxmars , LDyymars , LDxymars  = BMaker(IMGmars,2)
  
plt.imshow(LDxxmars, cmap = 'gray') 
plt.title("Lxx for image mars")
plt.show() 
plt.imshow(LDyymars, cmap = 'gray') 
plt.title("Lyy for image mars")
plt.show() 
plt.imshow(LDxymars, cmap = 'gray') 
plt.title("Lxy for image mars")
plt.show()

#ti tha eprepe na brainei 
LxxOG, LxyOG, LyyOG = HessianCoeffs(IMGmars, 2)
plt.imshow(LxxOG, cmap = 'gray') 
plt.title("ORIGINAL Lyy for mars")
plt.show()

plt.imshow(LxyOG, cmap = 'gray') 
plt.title("ORIGINAL Lxy for mars")
plt.show()

plt.imshow(LyyOG, cmap = 'gray') 
plt.title("ORIGINAL Lxx for mars")
plt.show()

#visualisation ONLY for multiple scales' Box Filter
def properArgsVisual2(ALLkrithria):
    correctINPUT = []
    elements = []
    sLista = []
    NLista = [0, 1, 2, 3]
    for i in range(0,len(NLista)):
        s, r = scaleANDr(i)
        sLista.append(s)
    counterInterest = 0
    for i in range(0,4):
        correctINPUT.clear()
        pinakas =  ALLkrithria[i]
        grammes,sthles = pinakas.shape
        for row in range(5,int(grammes-5)):
            for column in range(5,int(sthles-5)):
                if(pinakas[row, column] == 1):
                    correctINPUT.append(column)
                    correctINPUT.append(row)
                    correctINPUT.append(sLista[i])
                    #correctINPUT.append(NLista[i]+1)
                    counterInterest = counterInterest +1
        correctIN = np.reshape(correctINPUT,(len(correctINPUT)//3,3)) 
        #print(correctIN)
        elements.append(correctIN)
    correct = np.concatenate((elements[0], elements[1], elements[2], elements[3]),axis=0)
    #print(correct)
    return correct, counterInterest

#efarmogi krithriou R 
def KrithrionRneo(LDxx,LDyy,LDxy,s):
    RD = LDxx*LDyy -0.81*LDxy*LDxy
    #R = np.linalg.det(Hessian_matrix)
    #condition S1
    ns = np.ceil(3*s)*2+1
    B_sq = disk_strel(ns)
    RDilation = cv2.dilate(RD,B_sq)
    megisto = RD.max()
    thetacorn = 0.005
    eikonaTelikh = cv2.bitwise_and(np.where(RD==RDilation,1,0),np.where(RD>thetacorn*megisto,1,0))
    correctINPUT = []
    counter = 0 
    grammes, sthles = eikonaTelikh.shape
    for i in range(5,int(grammes-5)):
        for j in range(5,int(sthles-5)):
            if (eikonaTelikh[i, j] == 1):
                correctINPUT.append(j)
                correctINPUT.append(i)
                correctINPUT.append(s)    
                counter += 1
        correctIN = np.reshape(correctINPUT,(len(correctINPUT)//3,3)) 
    #print(correctIN)    
    return eikonaTelikh, correctIN , RD , counter

#VISUALIZATION OF BLOPS FOR IMAGE MARS for scale = 2
eikonaTelikhmarsSimple, sostoINmarsSimple ,_ ,_ = KrithrionRneo(LDxxmars,LDyymars,LDxymars,2)
interest_points_visualization(IMGmarsRGB, sostoINmarsSimple)
plt.title("Visualization of Blops for mars image with BOX FILTERS for s=2")
plt.show()

#find the blops with the box filters
def PolyklimataBoxes(LDxx,LDyy,LDxy, eikona, N):
    imageLista = [] 
    sLista = []
    rLista = []
    imageLoGLista = [] 
    for i in range(0,N):
        s, r = scaleANDr(i)
        sLista.append(s)
        rLista.append(r)
        image1, numberOfShmeia ,_ ,_ = KrithrionRneo(LDxx,LDyy,LDxy,s)
        imageLista.append(image1)
        image2 = eikona
        imageLoG = normalLoGs(image2, s)
        imageLoGLista.append(imageLoG)
       
    for i in range(0,N):
        if(i == 0):
            krithrio1 = cv2.bitwise_and(np.where(imageLista[i]==1,1,0),np.where(imageLoGLista[i] > imageLoGLista[i+1],1,0))    
        elif(i == 1):
            krithrio2 = cv2.bitwise_and(np.where(imageLista[i]==1,1,0),np.where(imageLoGLista[i] > imageLoGLista[i-1],1,0),np.where(imageLoGLista[i] > imageLoGLista[i+1],1,0))
        elif(i == 2):
            krithrio3 = cv2.bitwise_and(np.where(imageLista[i]==1,1,0),np.where(imageLoGLista[i] > imageLoGLista[i-1],1,0),np.where(imageLoGLista[i] > imageLoGLista[i+1],1,0))
        else:
            krithrio4 = cv2.bitwise_and(np.where(imageLista[i]==1,1,0),np.where(imageLoGLista[i] > imageLoGLista[i-1],1,0))
    
    #put all the krithria in one list
    ALLkrithria = [krithrio1, krithrio2, krithrio3, krithrio4]
    
    sostoIN, asxeto = properArgsVisual2(ALLkrithria)
    return sostoIN


#VISUALIZATION OF BLOPS FOR IMAGE MARS for multiple scales
correctINmarsMulti = PolyklimataBoxes(LDxxmars,LDyymars,LDxymars,IMGmars,4) 
interest_points_visualization(IMGmarsRGB, correctINmarsMulti)
plt.title("Visualization of Blops for mars image with BOX FILTERS for multiple scales")
plt.show()

LDxxbloodsmear , LDyybloodsmear , LDxybloodsmear  = BMaker(IMGbloodsmear,2)

#VISUALIZATION OF BLOPS FOR IMAGE MARS for scale = 2
eikonaTelikhbloodsmearSimple, sostoINbloodsmearSimple ,_ ,_ = KrithrionRneo(LDxxbloodsmear,LDyybloodsmear,LDxybloodsmear,2)
interest_points_visualization(IMGbloodsmearRGB, sostoINbloodsmearSimple)
plt.title("Visualization of Blops for blood smear image with BOX FILTERS for s=2")
plt.show()

#VISUALIZATION OF BLOPS FOR IMAGE MARS for multiple scales
correctINbloodsmearMulti = PolyklimataBoxes(LDxxbloodsmear,LDyybloodsmear,LDxybloodsmear,IMGbloodsmear,4) 
interest_points_visualization(IMGbloodsmearRGB, correctINbloodsmearMulti)
plt.title("Visualization of Blops for blood smear image with BOX FILTERS for multiple scales")
plt.show()

_,_, R2 , NumberBlopsBox2 = KrithrionRneo(LDxxmars,LDyymars,LDxymars,2)
_,_, R4 , NumberBlopsBox4 = KrithrionRneo(LDxxmars,LDyymars,LDxymars,4)
_,_, R6 , NumberBlopsBox6 = KrithrionRneo(LDxxmars,LDyymars,LDxymars,6)
_,_, R8 , NumberBlopsBox8 = KrithrionRneo(LDxxmars,LDyymars,LDxymars,8)

def returnR(eikona,s):
    Lxx, Lxy, Lyy = HessianCoeffs(eikona, s)
    R_real = Lxx*Lyy -Lxy*Lxy
    return R_real

R_real2 = returnR(IMGmars, 2)
R_real4 = returnR(IMGmars, 4)
R_real6 = returnR(IMGmars, 6)
R_real8 = returnR(IMGmars, 8)

_, nunmberBlopsmars2 = Harris(IMGmars,2,2.5,"IMGmars",1,"Hessian")
_, nunmberBlopsmars4 = Harris(IMGmars,4,2.5,"IMGmars",1,"Hessian")
_, nunmberBlopsmars6 = Harris(IMGmars,6,2.5,"IMGmars",1,"Hessian")
_, nunmberBlopsmars8 = Harris(IMGmars,8,2.5,"IMGmars",1,"Hessian")

#approach's quality comparison (Harris-Hessian and Box Filters)
print("Approach's quality comparison for s=2 : ")
print("The number of Blops of Rmars with Harris-Hessian for s=2 is: " + str(nunmberBlopsmars2)) 
print("The number of Blops of Rmars with Box Filters for s=2 is:    " + str(NumberBlopsBox2))  

print("Approach's quality comparison for s=4 : ")
print("The number of Blops of Rmars with Harris-Hessian for s=4 is: " + str(nunmberBlopsmars4)) 
print("The number of Blops of Rmars with Box Filters for s=4 is:    " + str(NumberBlopsBox4))

print("Approach's quality comparison for s=6 : ")
print("The number of Blops of Rmars with Harris-Hessian for s=6 is: " + str(nunmberBlopsmars6)) 
print("The number of Blops of Rmars with Box Filters for s=6 is:    " + str(NumberBlopsBox6))

print("Approach's quality comparison for s=8 : ")
print("The number of Blops of Rmars with Harris-Hessian for s=8 is: " + str(nunmberBlopsmars8)) 
print("The number of Blops of Rmars with Box Filters for s=8 is:    " + str(NumberBlopsBox8))

#visualisation of R with Harris-Hessian and Box Filters
print("Visualisation of R with Harris-Hessian and Box Filters for s=2 ,respectively:")
plt.imshow(R_real2, cmap = 'gray') 
plt.show()
plt.imshow(R2, cmap = 'gray') 
plt.show()      
print("Visualisation of R with Harris-Hessian and Box Filters for s=4 ,respectively:")      
plt.imshow(R_real4, cmap = 'gray') 
plt.show()
plt.imshow(R4, cmap = 'gray') 
plt.show()  
print("Visualisation of R with Harris-Hessian and Box Filters for s=6 ,respectively:")      
plt.imshow(R_real6, cmap = 'gray') 
plt.show()
plt.imshow(R6, cmap = 'gray') 
plt.show()  
print("Visualisation of R with Harris-Hessian and Box Filters for s=8 ,respectively:")      
plt.imshow(R_real8, cmap = 'gray') 
plt.show()
plt.imshow(R8, cmap = 'gray') 
plt.show()  


"""########################################################"""
"""##################### PART 3 REQUIRMENTS ###############"""
"""########################################################"""

def harrisDetector(eikona, s, r, k, thetacorn):
    Gs = creatorOfGaussian(s) #scale = 2 
    Gr = creatorOfGaussian(r) # r = 2.5
    IS = cv2.filter2D(eikona, -1, Gs)
    gradientXX, gradientXY, gradientYY = gradientCreator(IS)
    J1 = tanysthsCreator(Gr,gradientXX)
    J2 = tanysthsCreator(Gr,gradientXY)
    J3 = tanysthsCreator(Gr,gradientYY)
    lamdaThetiko = LAMDA(J1, J2, J3, 1)
    lamdaArnhtiko = LAMDA(J1, J2, J3, -1)
    oros1 = lamdaThetiko * lamdaArnhtiko
    athroisma = np.add(lamdaThetiko, lamdaArnhtiko)
    oros2 = athroisma * athroisma
    R = oros1 -k*oros2
    ns = np.ceil(3*s)*2+1
    B_sq = disk_strel(ns)
    RDilation = cv2.dilate(R,B_sq)
    megisto = R.max()
    eikonaTelikh = cv2.bitwise_and(np.where(R==RDilation,1,0),np.where(R>thetacorn*megisto,1,0))
    correctINPUT = []
    grammes, sthles = eikonaTelikh.shape
    for i in range(0,int(grammes)):
        for j in range(0,int(sthles)):
            if (eikonaTelikh[i, j] == 1):
                correctINPUT.append(j)
                correctINPUT.append(i)
                correctINPUT.append(s)    
        correctIN = np.reshape(correctINPUT,(len(correctINPUT)//3,3)) 
    return correctIN

def harrisHELP(eikona, s, r, k, thetacorn):
    Gs = creatorOfGaussian(s) #scale = 2 
    Gr = creatorOfGaussian(r) # r = 2.5
    IS = cv2.filter2D(eikona, -1, Gs)
    gradientXX, gradientXY, gradientYY = gradientCreator(IS)
    J1 = tanysthsCreator(Gr,gradientXX)
    J2 = tanysthsCreator(Gr,gradientXY)
    J3 = tanysthsCreator(Gr,gradientYY)
    lamdaThetiko = LAMDA(J1, J2, J3, 1)
    lamdaArnhtiko = LAMDA(J1, J2, J3, -1)
    oros1 = lamdaThetiko * lamdaArnhtiko
    athroisma = np.add(lamdaThetiko, lamdaArnhtiko)
    oros2 = athroisma * athroisma
    R = oros1 -k*oros2
    ns = np.ceil(3*s)*2+1
    B_sq = disk_strel(ns)
    RDilation = cv2.dilate(R,B_sq)
    megisto = R.max()
    eikonaTelikh = cv2.bitwise_and(np.where(R==RDilation,1,0),np.where(R>thetacorn*megisto,1,0))
    return eikonaTelikh

def properVisualHELP(ALLkrithria, N):
    correctINPUT = []
    elements = []
    sLista = []
    NLista = []
    for i in range(0,N):
        s, r = scaleANDr(i)
        sLista.append(s)
        NLista.append(i)
    counterInterest = 0
    for i in range(0,4):
        correctINPUT.clear()
        grammes,sthles = ALLkrithria[i].shape
        for row in range(0,int(grammes)):
            for column in range(0,int(sthles)):
                if(ALLkrithria[i][row, column] == 1):
                    correctINPUT.append(column)
                    correctINPUT.append(row)
                    #correctINPUT.append(NLista[i])
                    correctINPUT.append(sLista[i])
                    counterInterest = counterInterest +1
        correctIN = np.reshape(correctINPUT,(len(correctINPUT)//3,3)) 
        #print(correctIN)
        elements.append(correctIN)
    correct = np.concatenate((elements[0], elements[1], elements[2], elements[3]),axis=0)
    #print(correct)
    return correct

def harrisLaplaceDetector(eikona, scale, r, k, thetacorn, sigma, N):
    imageLista = [] 
    sLista = []
    rLista = []
    imageLoGLista = []
    for i in range(0,N):
        s, r = scaleANDr(i)
        sLista.append(s)
        rLista.append(r)
        image1 = harrisHELP(eikona, s, r, k, thetacorn)
        imageLista.append(image1)
        image2 = eikona
        imageLoG = normalLoGs(image2, s)
        imageLoGLista.append(imageLoG)

    for i in range(0,N):
        if(i == 0):
            krithrio1 = cv2.bitwise_and(np.where(imageLista[i]==1,1,0),np.where(imageLoGLista[i] > imageLoGLista[i+1],1,0))    
        elif(i == 1):
            krithrio2 = cv2.bitwise_and(np.where(imageLista[i]==1,1,0),np.where(imageLoGLista[i] > imageLoGLista[i-1],1,0),np.where(imageLoGLista[i] > imageLoGLista[i+1],1,0))
        elif(i == 2):
            krithrio3 = cv2.bitwise_and(np.where(imageLista[i]==1,1,0),np.where(imageLoGLista[i] > imageLoGLista[i-1],1,0),np.where(imageLoGLista[i] > imageLoGLista[i+1],1,0))
        else:
            krithrio4 = cv2.bitwise_and(np.where(imageLista[i]==1,1,0),np.where(imageLoGLista[i] > imageLoGLista[i-1],1,0))
    
    #put all the krithria in one list
    ALLkrithria = [krithrio1, krithrio2, krithrio3, krithrio4]
        
    correctIN = properVisualHELP(ALLkrithria, N)
    
    return correctIN

def hessianDetector(eikona, scale, sigma, thetacorn):
    Lxx, Lxy, Lyy = HessianCoeffs(eikona, scale)
    R = Lxx*Lyy -Lxy*Lxy    
    ns = np.ceil(3*scale)*2+1
    B_sq = disk_strel(ns)
    RDilation = cv2.dilate(R,B_sq)
    megisto = R.max()
    eikonaTelikh = cv2.bitwise_and(np.where(R==RDilation,1,0),np.where(R>thetacorn*megisto,1,0))
    correctIN = visualMONOs(eikonaTelikh, scale)
    return correctIN

def hessianHELP(eikona, scale, sigma, thetacorn):
    Lxx, Lxy, Lyy = HessianCoeffs(eikona, scale)
    R = Lxx*Lyy -Lxy*Lxy    
    #ns = np.ceil(3*sigma)*2+1
    ns = np.ceil(3*scale)*2+1
    B_sq = disk_strel(ns)
    RDilation = cv2.dilate(R,B_sq)
    megisto = R.max()
    eikonaTelikh = cv2.bitwise_and(np.where(R==RDilation,1,0),np.where(R>thetacorn*megisto,1,0))
    return eikonaTelikh

def hessianLaplaceDetector(eikona, scale, sigma, thetacorn, N):
    imageLista = [] 
    sLista = []
    rLista = []
    imageLoGLista = []
    for i in range(0,N):
        s, r = scaleANDr(i)
        sLista.append(s)
        rLista.append(r)
        image1 = hessianHELP(eikona, scale, sigma, thetacorn)
        imageLista.append(image1)
        image2 = eikona
        imageLoG = normalLoGs(image2, s)
        imageLoGLista.append(imageLoG)

    for i in range(0,N):
        if(i == 0):
            krithrio1 = cv2.bitwise_and(np.where(imageLista[i]==1,1,0),np.where(imageLoGLista[i] > imageLoGLista[i+1],1,0))   
        elif(i == 1):
            krithrio2 = cv2.bitwise_and(np.where(imageLista[i]==1,1,0),np.where(imageLoGLista[i] > imageLoGLista[i-1],1,0),np.where(imageLoGLista[i] > imageLoGLista[i+1],1,0))
        elif(i == 2):
            krithrio3 = cv2.bitwise_and(np.where(imageLista[i]==1,1,0),np.where(imageLoGLista[i] > imageLoGLista[i-1],1,0),np.where(imageLoGLista[i] > imageLoGLista[i+1],1,0))
        else:
            krithrio4 = cv2.bitwise_and(np.where(imageLista[i]==1,1,0),np.where(imageLoGLista[i] > imageLoGLista[i-1],1,0))
       
    #put all the krithria in one list
    ALLkrithria = [krithrio1, krithrio2, krithrio3, krithrio4]
    correctIN = properVisualHELP(ALLkrithria, N)
    
    return correctIN

def properVisualHELP2(ALLkrithria, N):
    correctINPUT = []
    elements = []
    sLista = []
    NLista = [0, 1, 2, 3]
    for i in range(0,len(NLista)):
        s, r = scaleANDr(i)
        sLista.append(s)
    for i in range(0,4):
        correctINPUT.clear()
        pinakas =  ALLkrithria[i]
        grammes,sthles = pinakas.shape
        for row in range(5,int(grammes-5)):
            for column in range(5,int(sthles-5)):
                if(pinakas[row, column] == 1):
                    correctINPUT.append(column)
                    correctINPUT.append(row)
                    correctINPUT.append(sLista[i])
        correctIN = np.reshape(correctINPUT,(len(correctINPUT)//3,3)) 
        #print(correctIN)
        elements.append(correctIN)
    correct = np.concatenate((elements[0], elements[1], elements[2], elements[3]),axis=0)
    #print(correct)
    return correct


def boxingDetecting(eikona, scale, N):
    gauss = creatorOfGaussian(scale)
    eikonaMEgauss = cv2.filter2D(eikona, -1, gauss)
#calculate the n with the given scale
    n = int(2*np.ceil(3*scale)+1)
#for Dxx
    xx_w = int(2*(np.floor(n/6)) + 1)
    xx_h = int(4*(np.floor(n/6)) + 1)
    Lxx1 = Mon_sin(eikonaMEgauss, xx_h, xx_w)
    #pad Lxx
    Lxx = np.pad(Lxx1, ((0,0), (xx_w,xx_w)))
    Lxx = -2*Lxx + np.roll(Lxx, xx_w ,axis=1) + np.roll(Lxx, -xx_w, axis=1)
    #unpad Lxx
    Lxx3 = Lxx[:, xx_w:-xx_w]
    
#for Dyy
    yy_w = int(4*(np.floor(n/6)) + 1)
    yy_h = int(2*(np.floor(n/6)) + 1)
    Lyy1 = Mon_sin(eikonaMEgauss, yy_h, yy_w)
    #pad Lyy
    Lyy = np.pad(Lyy1, ((yy_h,yy_h),(0,0) ))
    Lyy = -2*Lyy + np.roll(Lyy, yy_h ,axis=0) + np.roll(Lyy, -yy_h, axis=0)
    #unpad Lyy
    Lyy3 = Lyy[yy_h:-yy_h, :]
    
#for Dxy
    xy_w = int(2*(np.floor(n/6)) + 1)
    xy_h = int(2*(np.floor(n/6)) + 1)
    Lxy1 = Mon_sin(eikonaMEgauss, xy_h, xy_w)
    #pad Lxy
    Lxy = np.pad(Lxy1, ((xy_h+1,xy_h+1),(xy_w+1,xy_w+1) ))
    #kentro filtrou Dxy
    w = int(np.floor(xy_w/2))
    h = int(np.floor(xy_h/2))
   

    #up_left
    Ul = np.roll(Lxy, [(h+1), (w+1)] , axis=(0,1) )
    #down right
    Dr = np.roll(Lxy, [-(h+1), -(w+1)], axis=(0,1))
    #up right
    Ur = np.roll(Lxy, [h+1, -(w+1)], axis=(0,1))
    #down left
    Dl = np.roll(Lxy, [-(h+1), w+1], axis=(0,1))
         
    
    Lxy = Ul + Dr - Ur - Dl
    #unpad Lxy
    Lxy3 = Lxy[(xy_h+1):-(xy_h+1), (xy_w+1):-(xy_w+1)]
    
    imageLista = [] 
    sLista = []
    rLista = []
    imageLoGLista = [] 
    for i in range(0,N):
        s, r = scaleANDr(i)
        sLista.append(s)
        rLista.append(r)
        image1, numberOfShmeia,_,_ = KrithrionRneo(Lxx3,Lyy3,Lxy3,s)
        imageLista.append(image1)
        image2 = eikona
        imageLoG = normalLoGs(image2, s)
        imageLoGLista.append(imageLoG)
       
    for i in range(0,N):
        if(i == 0):
            krithrio1 = cv2.bitwise_and(np.where(imageLista[i]==1,1,0),np.where(imageLoGLista[i] > imageLoGLista[i+1],1,0))    
        elif(i == 1):
            krithrio2 = cv2.bitwise_and(np.where(imageLista[i]==1,1,0),np.where(imageLoGLista[i] > imageLoGLista[i-1],1,0),np.where(imageLoGLista[i] > imageLoGLista[i+1],1,0))
        elif(i == 2):
            krithrio3 = cv2.bitwise_and(np.where(imageLista[i]==1,1,0),np.where(imageLoGLista[i] > imageLoGLista[i-1],1,0),np.where(imageLoGLista[i] > imageLoGLista[i+1],1,0))
        else:
            krithrio4 = cv2.bitwise_and(np.where(imageLista[i]==1,1,0),np.where(imageLoGLista[i] > imageLoGLista[i-1],1,0))
    
    #put all the krithria in one list
    ALLkrithria = [krithrio1, krithrio2, krithrio3, krithrio4]
    
    sostoIN = properVisualHELP2(ALLkrithria, N)
    return sostoIN


