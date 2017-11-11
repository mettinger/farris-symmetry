
# coding: utf-8

# In[ ]:

from farris_symmetry import *


# In[ ]:

while True:
    imageIndex = 'random'
    colorWheel, imageName = colorWheelGet(imageList,index = imageIndex)
    print "color wheel: " + imageName
    
    ## Choose your symmetry!

    numStep = 1000 # number of steps along the two lattice vectors
    numCPU = 4 # number of CPUs to use with multiprocessing

    # SAMPLE THE GROUP
    groupType = randomGroupGet()
    print "group: " + groupType
    
    if groupType in ['p1','p4','p3']:
        numSampleCoeff = np.random.randint(1,9)
    elif groupType in ['p2', 'cm','pm','pg','p4m','p4g','p31m','p3m1','p6']:
        numSampleCoeff = np.random.randint(1,5)
    elif groupType in ['cmm','pmm','pmg','pgg','p6m']:
        numSampleCoeff = np.random.randint(1,4)
    else:
        print 'numSampleCoeff error...'
        
    print "number of free coefficients before contraints: " + str(numSampleCoeff)
    
    # HOW TO SAMPLE THE NM DICT?
    #coefficientFunction = exponentialDist(1.0)
    #nmMagnitudeFunction = geometricDist(.6)
    
    coefficientFunction = exponentialDist(1.0)
    nmMagnitudeFunction = uniformDist(1,3)

    # DETERMINE THE LATTICE TYPE
    latticeType = latticeTypeFromGroup(groupType)
    print "lattice type: " + latticeType
    
    if latticeType == 'general':
        latticeData = .5 + 1j 
    elif latticeType == 'rhombic':
        latticeData = .25
    elif latticeType == 'rectangular':
        latticeData = 2
    else:
        latticeData = None

    myF, lattice_vector_1, lattice_vector_2 = functionAndLatticeGet(latticeType, latticeData)
    print "lattice vector 1: " + str(lattice_vector_1)
    print "lattice vector 2: " + str(lattice_vector_2)
    
    nmDict = nmDictRandom(groupType, numSampleCoeff, coefficientFunction, nmMagnitudeFunction)
    print "nmDict: "
    print nmDict
    
    # compute matrix "d" used for tiling a region by the fundamental cell
    #c = np.array([[lattice_vector_1.real, lattice_vector_2.real],[lattice_vector_1.imag,lattice_vector_2.imag]])
    #d = np.linalg.inv(c)

    v = np.linspace(0,1,numStep)

    if numStep <=200 or numCPU==1:
        gridApply = dict([latticeEval((i,x,j,y,myF,nmDict,lattice_vector_1,lattice_vector_2)) for i,x in enumerate(v) for j,y in enumerate(v)])
    else:
        pool = Pool(numCPU)
        grid = [(i,x,j,y,myF,nmDict,lattice_vector_1,lattice_vector_2) for i,x in enumerate(v) for j,y in enumerate(v)]
        gridApply = dict(pool.map(latticeEval,grid))
        pool.close()

    fundamentalColorDict = colorwheelApply(gridApply, colorWheel)

    resx = 1500
    resy = 1000
    ratio = float(resx)/resy
    
    ymin = -1
    ymax = 1
    xspan = (ymax - ymin) * ratio
    xmin = -1 * int(xspan/2.)
    xmax = int(xspan/2.)
    
    print "x range, y range: " + str((xmin,xmax,ymin,ymax))
    imOut = tileFundamental(fundamentalColorDict, 
                            numStep, 
                            lattice_vector_1,
                            lattice_vector_2,
                            xmin=xmin, 
                            xmax=xmax, 
                            ymin=ymin, 
                            ymax=ymax, 
                            resx=resx, 
                            resy=resy)
    
    #imOut = cv2.resize(imOut, (1500,1000), interpolation=cv2.INTER_CUBIC)
    print "finished..."
    print 
    print
    cv2.namedWindow("test")
    cv2.destroyAllWindows()
    cv2.imshow("test", imOut)
    cv2.waitKey(1000)
    


# In[ ]:



