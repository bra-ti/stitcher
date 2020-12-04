import cv2

def getArea(x,y,gridx,gridy,directory,i):

    #loading actual stitched region image, from here relocate classified patches with coordinates x,y
    #and patch-sizes gridx,gridy
    img=cv2.imread(directory+'/stitch_'+str(i)+'.jpg')

    #Calculation of value representing area of pitting with value between 0 and 1 as float
    
    #placeholder
    analysed=1


    return float(analysed)
