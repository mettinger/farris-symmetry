
# coding: utf-8

# In[2]:

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from IPython.display import display
import pdb
from multiprocessing import Pool
import os
import pandas as pd
import cv2


# ## Load the list of images for possible colorwheels

# In[3]:

imageList = os.listdir('../images/')


# ## Define the graphics and plotting functions

# In[4]:

# GRAPHICS FUNCTIONS

def getRGB(z,minx,maxx,miny,maxy, imArray):
    dimx,dimy,_ = imArray.shape
    x = z.real
    y = z.imag
    xCoor = int(np.round(((x - minx)/float(maxx - minx)) * (dimx-1)))
    # if maxy = miny and you get an error, this is a degeneracy. use complex coefficients in nmDict !!!!
    yCoor = int(np.round(((y - miny)/float(maxy - miny)) * (dimy-1)))
        
    value = imArray[xCoor, yCoor, :]
    return value

def colorwheelApply(gridApply, imArray):
    allx = [z.real for z in gridApply.values()]
    ally = [z.imag for z in gridApply.values()]
    minx = min(allx)
    maxx = max(allx)
    miny = min(ally)
    maxy = max(ally)
    
    fundamentalDict = {}
    for key,value in gridApply.items():
        fundamentalDict[key] = getRGB(value,minx,maxx,miny,maxy, imArray)
    return fundamentalDict


def checkSymmetrySimple(gridApply):
    
    fundamentalDict = {}
    for key,value in gridApply.items():
         fundamentalDict[key] = np.array(((value.real * 1024) % 256 ,0, (value.imag * 1024) % 256), dtype='uint8')
        
    return fundamentalDict


def checkSymmetrySimpleReal(gridApply):
    
    fundamentalDict = {}
    allValues = [i.real for i in gridApply.values()]
    minx = min(allValues)
    maxx = max(allValues)
    span = maxx - minx
    for key,value in gridApply.items():
         fundamentalDict[key] = np.array((int(((value.real - minx)/span) * 255),0,0), dtype = 'uint8')
        
    return fundamentalDict

def checkSymmetrySimpleImag(gridApply):
    
    fundamentalDict = {}
    allValues = [i.imag for i in gridApply.values()]
    miny = min(allValues)
    maxy = max(allValues)
    span = maxy - miny
    for key,value in gridApply.items():
         fundamentalDict[key] = np.array((0,0,int(((value.imag - miny)/span) * 255)), dtype = 'uint8')
        
    return fundamentalDict

def fundamentalColorGet(thisTuple):
    
    x,y = thisTuple[0]
    fundamentalDict = thisTuple[1]
    numStep = thisTuple[2]
    i,j = thisTuple[3]
    
    # convert x,y to lattice basis
    
    xPrime, yPrime = np.dot(d,[x,y])
    xIndex = int((xPrime % 1) * (numStep-1))
    yIndex = int((yPrime % 1) * (numStep-1))
    
    value = fundamentalDict[(xIndex,yIndex)]
    
    return ((i,j), value)
    

def tileFundamental(fundamentalDict, numStep, xmin, xmax, ymin, ymax, resx, resy):

    if resy == None:
        resy = int((resx * (ymax-ymin))/float(xmax - xmin))
        
    xv = np.linspace(xmin, xmax, resx)
    yv = np.linspace(ymin, ymax, resy)
    
    grid = [((x,y), fundamentalDict, numStep, (i,j)) for i,x in enumerate(xv) for j,y in enumerate(yv)]
    coordinateDict = dict([fundamentalColorGet(i) for i in grid])
    
    imOut = np.zeros((resy, resx, 3), dtype='uint8')
    for key,value in coordinateDict.items():
        i,j = key
        imOut[resy - j - 1, i, :] = value
        
    return imOut


def latticeEval(thisTuple):
    i,x,j,y,myF,nmDict,lattice_vector_1,lattice_vector_2 = thisTuple
    
    z = (x * lattice_vector_1) + (y * lattice_vector_2)
    
    return ((i,j), myF(nmDict)((z.real, z.imag)))


def fundamentalCellDisplay(fundamentalDict, resx, resy):

    extremePoints = [0., lattice_vector_1, lattice_vector_2, lattice_vector_1 + lattice_vector_2]
    minx = min([i.real for i in extremePoints])
    maxx = max([i.real for i in extremePoints])
    miny = min([i.imag for i in extremePoints])
    maxy = max([i.imag for i in extremePoints])
    
    gap = max([maxy - miny, maxx - minx])
    maxx = minx + gap
    maxy = miny + gap
    
    imOut = np.zeros((resy, resx, 3), dtype='uint8')
    for key,value in fundamentalDict.items():
        z = ((lattice_vector_1 * key[0]) + (lattice_vector_2 * key[1])) / numStep
        x = z.real
        y = z.imag
        
        i = int(((x - minx)/(maxx-minx)) * (resx-1))
        j = int(((y - miny)/(maxy-miny)) * (resy-1))
        imOut[resy - j - 1,i,:] = value
     
    display(Image.fromarray(imOut))

    
def colorWheelGet(imageList, index='random'):
    
    if index=='random':
        imageName = np.random.choice(imageList)
    elif type(index) == str:
        imageName = index
    else:
        imageName = imageList[index]
        
    im = Image.open("../images/" + imageName)
    colorWheel = np.array(im)
    return colorWheel, imageName


# ## Here we define the essential symmetry functions, following the appendix in the book.

# In[5]:

# SYMMETRY FUNCTIONS

# GENERAL LATTICE
def E_general(nmPair,x,y):
    n,m = nmPair
    a = lattice_vector_2.real
    b = lattice_vector_2.imag
    X = x - ((a * y)/b)
    Y = y/b
    value = np.exp(2 * np.pi * 1j * ((n * X) + (m * Y)))
    return value

def f_general(nmDict):
    g = lambda xyPair : np.sum([value * E_general(key,xyPair[0],xyPair[1]) for key,value in nmDict.items()])
    return g


# RHOMBIC LATTICE
def E_rhombic(nmPair,x,y):
    n,m = nmPair
    b = lattice_vector_2.imag
    X = x + (y/(2.*b))
    Y = x - (y/(2.*b))
    value = np.exp(2 * np.pi * 1j * ((n * X) + (m * Y)))
    return value

def f_rhombic(nmDict):
    g = lambda xyPair : np.sum([value * E_rhombic(key,xyPair[0],xyPair[1]) for key,value in nmDict.items()])
    return g


# RECTANGULAR LATTICE
def E_rectangular(nmPair,x,y):
    n,m = nmPair
    L = lattice_vector_2.imag
    X = x 
    Y = y/L
    value = np.exp(2 * np.pi * 1j * ((n * X) + (m * Y)))
    return value

def f_rectagular(nmDict):
    g = lambda xyPair : np.sum([value * E_rectangular(key,xyPair[0],xyPair[1]) for key,value in nmDict.items()])
    return g


# SQUARE LATTICE
def E_square(nmPair,x,y):
    n,m = nmPair
    X = x 
    Y = y
    value = np.exp(2 * np.pi * 1j * ((n * X) + (m * Y)))
    return value

def S(nmPair0,x,y):
    n,m = nmPair0
    nmPair1 = (m,-n)
    nmPair2 = (-n,-m)
    nmPair3 = (-m,n)
    value = (E_square(nmPair0,x,y) + E_square(nmPair1,x,y) + E_square(nmPair2,x,y) + E_square(nmPair3,x,y))/4.
    return value

def f_square(nmDict):
    g = lambda xyPair : np.sum([value * S(key,xyPair[0],xyPair[1]) for key,value in nmDict.items()])
    return g

# HEXAGONAL LATTICE
def E_hex(nmPair,x,y):
    n,m = nmPair
    X = x + (y/np.sqrt(3))
    Y = (2 * y)/np.sqrt(3)
    value = np.exp(2 * np.pi * 1j * ((n * X) + (m * Y)))
    return value

def W(nmPair0,x,y):
    n,m = nmPair0
    nmPair1 = (m, -n - m)
    nmPair2 = (-n - m, n)
    value = (E_hex(nmPair0,x,y) + E_hex(nmPair1,x,y) + E_hex(nmPair2,x,y))/3.
    return value

def f_hex(nmDict):
    g = lambda xyPair : np.sum([value * W(key,xyPair[0],xyPair[1]) for key,value in nmDict.items()])
    return g


# ## Set the lattice type and the symmetry group.

# In[6]:

def functionAndLatticeGet(flag, latticeData):
    
    if flag == 'general':
        myF = f_general

        # lattice_vector_2 is arbitrary
        lattice_vector_1 = 1.
        lattice_vector_2 = latticeData

    elif flag == 'rhombic':
        myF = f_rhombic

        # lattice vectors are defined by b
        b = latticeData
        lattice_vector_1 = .5 + (b*1j)
        lattice_vector_2 = .5 - (b*1j)

    elif flag == 'rectangular':
        myF = f_rectagular

        # lattice vectors determined by L
        L = latticeData
        lattice_vector_1 = 1.
        lattice_vector_2 = L * 1j

    elif flag == 'square':
        myF = f_square

        # lattice vectors are fixed
        lattice_vector_1 = 1.
        lattice_vector_2 = 1j

    elif flag == 'hexagonal':
        myF = f_hex

        # lattice vectors are fixed
        lattice_vector_1 = 1.
        lattice_vector_2 = (-1 + (1j * np.sqrt(3)))/2.


    return (myF, lattice_vector_1, lattice_vector_2)

def latticeTypeFromGroup(group):
    
    if group in ['p1','p2']:
        latticeType = 'general'
    elif group in ['cm','cmm']:
        latticeType = 'rhombic'
    elif group in ['p4','p4m','p4g']:
        latticeType = 'square'
    elif group in ['pm','pg','pmm','pmg','pgg']:
        latticeType = 'rectangular'
    elif group in ['p3','p31m','p3m1','p6','p6m']:
        latticeType = 'hexagonal'
    else:
        print "Invalid group..."
        return Non
    
    return latticeType

def randomGroupGet():
    allGroups = ['p1','p2','cm','cmm','p4','p4m','p4g','pm','pg','pmm','pmg','pgg','p3','p31m','p3m1','p6','p6m']
    return np.random.choice(allGroups)


# ## Functions for random wallpaper functions

# In[7]:

# RETURN A RANDOM INTEGER PAIR WITH GIVEN l_infinity NORM
def uniform_nm_fixed_magnitude(magnitude):
    a = np.random.randint(-magnitude, magnitude + 1)
    flip = np.random.randint(0,2)
    if flip:
        magnitude = -magnitude
        
    flip = np.random.randint(0,2)
    if flip:
        return (magnitude, a)
    else:
        return (a, magnitude)
    
# GENERATE A RANDOM n,m PAIR
def nmPairRandom(nmMagnitudeFunction):
    magnitude = nmMagnitudeFunction()
    return uniform_nm_fixed_magnitude(magnitude)


# GENERATE A RANDOM COMPLEX NUMBER WITH UNIFORMLY RANDOM PHASE
def randomComplex(magnitudeFunction):
    
    #uniformly random phase
    randomPhase = np.random.uniform(0,2 * np.pi)
    
    #random magnitude
    randomAbsoluteValue = magnitudeFunction()
    
    return randomAbsoluteValue * np.exp(1j * randomPhase)

# GENERATE A RANDOM nmDict
def nmDictRandom(groupType, numSampleCoeff, coefficientFunction, nmMagnitudeFunction):
    
    nmDict = {}
    if groupType in ['p1','p4','p3']:
        for i in range(numSampleCoeff):
            nmPair = nmPairRandom(nmMagnitudeFunction)
            a = randomComplex(coefficientFunction)
            nmDict[nmPair] = a
            
    elif groupType in ['p2','p6']:
        for i in range(numSampleCoeff):
            nmPair = nmPairRandom(nmMagnitudeFunction)
            a = randomComplex(coefficientFunction)
            nmDict[nmPair] = a
            n,m = nmPair
            nmDict[(-n,-m)] = a
            
    elif groupType in ['cm','p4m','p31m']:
        for i in range(numSampleCoeff):
            nmPair = nmPairRandom(nmMagnitudeFunction)
            a = randomComplex(coefficientFunction)
            nmDict[nmPair] = a
            n,m = nmPair
            nmDict[(-n,-m)] = a
            
    elif groupType in ['cmm','p6m']:
        for i in range(numSampleCoeff):
            nmPair = nmPairRandom(nmMagnitudeFunction)
            a = randomComplex(coefficientFunction)
            nmDict[nmPair] = a
            n,m = nmPair
            nmDict[(-n,-m)] = a
            nmDict[(m,n)] = a
            nmDict[(-m,-n)] = a
            
    elif groupType in ['p4g']:
        for i in range(numSampleCoeff):
            nmPair = nmPairRandom(nmMagnitudeFunction)
            a = randomComplex(coefficientFunction)
            nmDict[nmPair] = a
            n,m = nmPair
            nmDict[(m,n)] = ((-1)**(n+m)) * a
            
    elif groupType in ['pm']:
        for i in range(numSampleCoeff):
            nmPair = nmPairRandom(nmMagnitudeFunction)
            a = randomComplex(coefficientFunction)
            nmDict[nmPair] = a
            n,m = nmPair
            nmDict[(n,-m)] = a
            
    elif groupType in ['pg']:
        for i in range(numSampleCoeff):
            nmPair = nmPairRandom(nmMagnitudeFunction)
            a = randomComplex(coefficientFunction)
            nmDict[nmPair] = a
            n,m = nmPair
            nmDict[(n,-m)] = ((-1)**n) * a
            
    elif groupType in ['pmm']:
        for i in range(numSampleCoeff):
            nmPair = nmPairRandom(nmMagnitudeFunction)
            a = randomComplex(coefficientFunction)
            nmDict[nmPair] = a
            n,m = nmPair
            nmDict[(-n,-m)] = a
            nmDict[(-n,m)] = a
            nmDict[(n,-m)] = a        
    
    elif groupType in ['pmg']:
        for i in range(numSampleCoeff):
            nmPair = nmPairRandom(nmMagnitudeFunction)
            a = randomComplex(coefficientFunction)
            nmDict[nmPair] = a
            n,m = nmPair
            nmDict[(-n,-m)] = a
            nmDict[(n,-m)] = ((-1)**n) * a
            nmDict[(-n,m)] = ((-1)**n) * a
            
    elif groupType in ['pgg']:
        for i in range(numSampleCoeff):
            nmPair = nmPairRandom(nmMagnitudeFunction)
            a = randomComplex(coefficientFunction)
            nmDict[nmPair] = a
            n,m = nmPair
            nmDict[(-n,-m)] = a
            nmDict[(n,-m)] = ((-1)**(n+m)) * a
            nmDict[(-n,m)] = ((-1)**(n+m)) * a
            
    elif groupType in ['p3m1']:
        for i in range(numSampleCoeff):
            nmPair = nmPairRandom(nmMagnitudeFunction)
            a = randomComplex(coefficientFunction)
            nmDict[nmPair] = a
            n,m = nmPair
            nmDict[(-m,-n)] = a
            
    return nmDict



# ## Functions defining random distributions for n,m pairs and complex coefficients

# In[5]:

# n,m MAGNITUDE FUNCTIONS FOR nmPairRandom
def geometricDist(p):
    return lambda : np.random.geometric(p)


# MAGNITUDE FUNCTIONS FOR randomComplex
def exponentialDist(scale):
    return lambda : np.random.exponential(scale)

def constantDist(scale):
    return lambda : scale

def uniformDist(low,high):
    return lambda : np.random.randint(low,high)


# In[ ]:



