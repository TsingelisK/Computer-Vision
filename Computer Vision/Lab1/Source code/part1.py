# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 12:55:50 2021

@authors: Panagiotis Gardelis & Konstantinos Spathis
"""

import cv2 
import matplotlib.pyplot as plt
import numpy as np

"""###########ASKHSH 1.1.1 ###############"""

#diabazoyme thn griza eikona 
IMGedgetest = cv2.imread("edgetest_10.png", 0) #to 0 einai to antistoixo tou cv2.IMAGE_GRAYSCALE

print("The pixels of image are the following: ")
print(IMGedgetest)
print("Resolution : " , IMGedgetest.shape, "pixels")
print("Range : {} - {} ".format(IMGedgetest.min(),IMGedgetest.max()))

"""
#rows,columns,channels = IMGedgetest.shape
rows,columns = IMGedgetest.shape
print("rows : " + str(rows))
print("columns : " + str(columns))
#print("channels: " + str(channels))
"""
#normalize from 0-255 to [0,1]
#metatropi eikonas se float
IMGedgetest = IMGedgetest.astype(np.float) / 255
print("NEW Range : {} - {} ".format(IMGedgetest.min(),IMGedgetest.max()))

#if no cmap argument given, the default is color image 
#plt.imshow(IMGedgetest)
plt.imshow(IMGedgetest, cmap = "gray") 
plt.title("Original Image of egdetest_10")
plt.show()

#clear the image by changing it in a binary image with thresholding
_, IMGedgetestBinary = cv2.threshold(IMGedgetest, 0.5, 1, cv2.THRESH_BINARY)

plt.imshow(IMGedgetestBinary, cmap='gray') # gray_r colormap shows 1 as black
plt.title("Clear Binary Image of egdetest_10")
plt.show()


"""########### ASKHSH 4.1 (reading urban)###############ArxiParembolhs"""
#reading the urban_edges 
IMGurbanedges  = cv2.imread("urban_edges.jpg", cv2.IMREAD_COLOR) 
IMGurbanedges = cv2.cvtColor(IMGurbanedges, cv2.COLOR_BGR2RGB)

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
"""########### ASKHSH 4.1 (reading urban)###############TelosParembolhs"""

"""###########ASKHSH 1.1.2 ###############"""

##function to calculate the scale (sn) given the PSNR 
##and the eikona I0 not the eikona with the image
def getTheScale(eikona, PSNR):
    Imax = np.max(eikona)  
    Imin = np.min(eikona)
    ekthetiko = 10**(PSNR/20) 
    scale = (Imax - Imin)/(ekthetiko)
    return scale

#noisy image with PSNR = 20dB
IMGedgetestWithNoise20 = IMGedgetest + np.random.normal(0,getTheScale(IMGedgetest, 20),IMGedgetest.shape)

#noisy image with PSNR = 10dB
IMGedgetestWithNoise10 = IMGedgetest + np.random.normal(0,getTheScale(IMGedgetest, 10),IMGedgetest.shape)

######## 1h ylopoish me xrhsh toy typou PSNR ###########"""
"""######## 1h ylopoish me xrhsh toy typou PSNR ###########
#noisy image with PSNR = 20dB so with scale = 1.5
IMGedgetestWithNoise20 = IMGedgetest + np.random.normal(0,0.15,IMGedgetest.shape)

#noisy image with PSNR = 10dB so with scale = 3
IMGedgetestWithNoise10 = IMGedgetest + np.random.normal(0,0.3,IMGedgetest.shape)
"""
plt.imshow(IMGedgetestWithNoise20, cmap = "gray") 
plt.title("Noisy Image of egdetest_10 with PSNR=20dB")
plt.show()

plt.imshow(IMGedgetestWithNoise10, cmap = "gray") 
plt.title("Noisy Image of egdetest_10 with PSNR=10dB")
plt.show()

"""
DOKIMI ME BINARY
#noisy image with PSNR = 20dB
IMGedgetestWithNoise20 = IMGedgetestBinary + np.random.normal(0,getTheScale(IMGedgetestBinary, 20),IMGedgetestBinary.shape)

#noisy image with PSNR = 10dB
IMGedgetestWithNoise10 = IMGedgetestBinary + np.random.normal(0,getTheScale(IMGedgetestBinary, 10),IMGedgetestBinary.shape)

plt.imshow(IMGedgetestWithNoise20, cmap = "gray") 
plt.title("Noisy Image of egdetest_10 with PSNR=20dB")
plt.show()

plt.imshow(IMGedgetestWithNoise10, cmap = "gray") 
plt.title("Noisy Image of egdetest_10 with PSNR=10dB")
plt.show()
"""

"""###########ASKHSH 1.2 ###############"""

def filteringImage(eikona, scale, krithsFilter):
    n = int(2*np.ceil(3*scale)+1)
    gauss1D = cv2.getGaussianKernel(n, scale)
    gauss2D = gauss1D @ gauss1D.T 
    eikonaMeGaussian = cv2.filter2D(eikona, -1, gauss2D)
    eikonaLoG = cv2.Laplacian(eikonaMeGaussian,cv2.CV_64FC1,3)
    if(krithsFilter == 0) : #bazw 0 gia gaussian filter
        eikonaSmooth = eikonaMeGaussian
    elif(krithsFilter == 1): #bazw 1 gia to LoG filter
        eikonaSmooth = eikonaLoG
    return eikonaSmooth

def EdgeDetect(eikona, scale, judgeLinear, theta): 
    #creation of the kernel B
    B = np.array([  
           [0,-1,0],
           [-1,4,-1],
           [0,-1,0] 
    ], dtype=np.uint8)
    
    #creation of gaussian image
    eikonaMeGaussian = filteringImage(eikona,scale,0)
    
    ### judge = 1 for LINEAR ###
    if(judgeLinear == 1): 
       #creation of Linear image
       eikonaLinear = filteringImage(eikona,scale,1)
       #declare Laplacian ish me me thn Linear
       eikonaLaplacian = eikonaLinear
    ### judge = 0 for NON-LINEAR ###
    elif(judgeLinear == 0):
        #creation of gaussian image
        #eikonaMeGaussian = filteringImage(eikona,scale,0)
        #creation of dilation of image
        eikonaMeDilation = cv2.dilate(eikonaMeGaussian, B)
        #creation of erosion of image
        eikonaMeErosion = cv2.erode(eikonaMeGaussian,B)
        #creation of NON-Linear image
        eikonaNonLinear = eikonaMeDilation + eikonaMeErosion - 2*eikonaMeGaussian
        #declare Laplacian ish me me thn Linear
        eikonaLaplacian = eikonaNonLinear

    _, eikonaX = cv2.threshold(eikonaLaplacian, 0, 1, cv2.THRESH_BINARY) 
    #eikonaX = eikonaLaplacian
    #creation of dilation of image eikonaX
    eikonaXmeDilation = cv2.dilate(eikonaX, B)
    #creation of erosion of image eikonaX
    eikonaXmeErosion = cv2.erode(eikonaX,B)
    #creation of zerocrossing image eikonaY
    eikonaY1 = eikonaXmeDilation - eikonaXmeErosion 
    
    _, eikonaY = cv2.threshold(eikonaY1, 0, 1, cv2.THRESH_BINARY) 
    
    eikonaMeGaussianGradient1 = np.gradient(eikonaMeGaussian,axis=1)
    eikonaMeGaussianGradient2 = np.gradient(eikonaMeGaussian,axis=0)
    eikonaMAX1=np.max(eikonaMeGaussianGradient1)
    eikonaMAX2=np.max(eikonaMeGaussianGradient2)
    
    #eikonaD = cv2.bitwise_and(np.where(eikonaY==1,1,0),np.where(cv2.bitwise_or(np.where(abs(eikonaMeGaussianGradient1)>theta*abs(eikonaMAX1)),np.where(abs(eikonaMeGaussianGradient2)>theta*abs(eikonaMAX2))),1,0))
   
    grammes,sthles = eikonaY.shape
    for i in range(0,int(grammes)):
        for j in range(0,int(sthles)):
            if (eikonaY[i,j]==1 and (abs(eikonaMeGaussianGradient1[i,j])>theta*abs(eikonaMAX1)  or abs(eikonaMeGaussianGradient2[i,j])>theta*abs(eikonaMAX2))) :
                eikonaY[i,j]=1
            else:
                eikonaY[i,j]=0
    eikonaD = eikonaY
    
    return  eikonaD

def plotarismaEdgeDetect(NoisyEikona, scale, judgeLinear, theta, onomaPSNR, onomaEikonas):        
    if(judgeLinear == 1):
        onomaL = "'Linear'"
    if(judgeLinear == 0):
        onomaL = "'NON-Linear'"
    
    eikonaPlot = EdgeDetect(NoisyEikona, scale, judgeLinear, theta)  
    
    plt.imshow(eikonaPlot, cmap = "gray_r") 
    plt.title("Akmes of noisy image "+onomaEikonas+" with PSNR ="+str(onomaPSNR)+" and approximation="+onomaL+" scale ="+str(scale)+" & theta="+str(theta))
    plt.show()
    return eikonaPlot

#ET --> Edge_test LINEAR 
eikonaET20L = plotarismaEdgeDetect(IMGedgetestWithNoise20, 1.5, 1, 0.2, "20", "Edge_test")
eikonaET10L =plotarismaEdgeDetect(IMGedgetestWithNoise10, 3, 1, 0.2, "10", "Edge_test")

#ET --> Edge_test NON-LINEAR 
eikonaET20NL = plotarismaEdgeDetect(IMGedgetestWithNoise20, 1.5, 0, 0.2, "20", "Edge_test")
eikonaET10NL =plotarismaEdgeDetect(IMGedgetestWithNoise10, 3, 0, 0.2, "10", "Edge_test")

#UE --> Urban_edges LINEAR 
eikonaUEL = plotarismaEdgeDetect(IMGurbanedges, 0.5, 1, 0.2, "0", "Urban_edges")

#UE --> Urban_edges NON-LINEAR 
eikonaUENL = plotarismaEdgeDetect(IMGurbanedges, 0.5, 0, 0.2, "0", "Urban_edges")

"""###########ASKHSH 1.3 ###############"""

"""###########ASKHSH 1.3.1 ###############"""

def EdgeOperator(eikona, thetareal):
    #creation of the kernel B
    B = np.array([  
           [0,1,0],
           [1,1,1],
           [0,1,0] 
    ], dtype=np.uint8)
    #creation of dilation of image
    eikonaMeDilation = cv2.dilate(eikona, B)
    #creation of erosion of image
    eikonaMeErosion = cv2.erode(eikona,B)
    #creation of M
    M = eikonaMeDilation - eikonaMeErosion
    _, T= cv2.threshold(M, thetareal, 1, cv2.THRESH_BINARY)
    return T

def plotarismaEdgeOperator(eikona, thetareal1, onomaEikonas):
    #calculate the akmes
    eikonaToPlot = EdgeOperator(eikona, thetareal1)
    
    #plotarisma eikonas eisodou
    plt.imshow(eikonaToPlot, cmap = "gray_r") 
    plt.title("REAL EDGES of "+onomaEikonas+" with thetareal=" + str(thetareal1))
    plt.show()
    return eikonaToPlot

#ET --> Edge_test
eikonaET = plotarismaEdgeOperator(IMGedgetest, 0.2,"edgetest_10")

#UE --> UrbanEdges
eikonaUE = plotarismaEdgeOperator(IMGurbanedges, 0.2, "urban_edges")

"""###########ASKHSH 1.3.2 ###############"""

#creation of the kernel B
B = np.array([  
           [0,1,0],
           [1,1,1],
           [0,1,0] 
    ], dtype=np.uint8)

def posostoImages(eikona1, eikona2, krithsPosostou):
    #creation of the instersection of two images
    eikonaTomi = cv2.bitwise_and(eikona1,eikona2)
    #calculate the card for the intersection 
    eikonaTomiCard = eikonaTomi.sum()
    #calculate the card for the eikonaT
    eikona2Card = eikona2.sum()
    #calculate the card for the eikonaD
    eikona1Card = eikona1.sum()
    if(krithsPosostou == 0):
        pososto = eikonaTomiCard/eikona2Card
    if(krithsPosostou == 1):
        pososto = eikonaTomiCard/eikona1Card
    return pososto

""" POSOSTO FOR THE IMAGE Edge_test """

def posotoPrinter(eikonaAprrox, eikonaReal):
    #print("POSOSTO FOR THE IMAGE" + onomaEikonas)
    eikona0 = posostoImages(eikonaAprrox,eikonaReal, 0)
    eikona1 = posostoImages(eikonaAprrox,eikonaReal, 1)
    pososto = (eikona0+eikona1)/2
    return pososto 

print("POSOSTA FOR THE IMAGE Edge_test ")

"""LINEAR EDGE TEST"""
#pososta gia thn eikonaD1 me ta 20dB 
ET20Cgia20L = posotoPrinter(eikonaET20L, eikonaET)
print("C for the Linear Edge_test with PSNR=20dB: " +str(ET20Cgia20L))
#pososta gia thn eikonaD1 me ta 10dB 
ET10Cgia10L = posotoPrinter(eikonaET10L, eikonaET)
print("C for the Linear Edge_test with PSNR=10dB: " +str(ET10Cgia10L))

"""NON LINEAR EDGE TEST"""
#pososta gia thn eikonaD1 me ta 20dB 
ET20Cgia20NL = posotoPrinter(eikonaET20NL, eikonaET)
print("C for the NON Linear Edge_test with PSNR=20dB: " +str(ET20Cgia20NL))
#pososta gia thn eikonaD1 me ta 10dB 
ET10Cgia10NL = posotoPrinter(eikonaET10NL, eikonaET)
print("C for the NON Linear Edge_test with PSNR=10dB: " +str(ET10Cgia10NL))

print("POSOSTA FOR THE IMAGE Urban_edges ")

"""LINEAR URBAN EDGES"""
#pososta gia thn eikonaD1 me ta 20dB 
UECgiaL = posotoPrinter(eikonaUEL, eikonaUE)
print("C for the Linear Urban_edges: " +str(UECgiaL))


"""NON LINEAR URBAN EDGES"""
#pososta gia thn eikonaD1 me ta 20dB 
UECgiaNL = posotoPrinter(eikonaUENL, eikonaUE)
print("C for the NON Linear Urban_edges: " +str(UECgiaNL))
