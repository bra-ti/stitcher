import numpy as np
import cv2
import time
from PIL import Image
import os
import json
import sys

def ROI(img_old,ROI_list):
    return img_old[int(ROI_list[1]):int(ROI_list[3]), int(ROI_list[0]):int(ROI_list[2])]

def bbox(img):
    img = (img > 0)
    cols = np.all(img, axis=0)
    cmin, cmax = np.argmax(cols), img.shape[1] - 1 - np.argmax(np.flipud(cols))
    return cmin, cmax

def col_crop_black(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_crop=img
    ret, thresh_img = cv2.threshold(img_crop,1,255,cv2.THRESH_BINARY)
    cmin, cmax = bbox(thresh_img)
    return cmin, cmax

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def detector_descriptor(img1, img2):             
    img1 = img1
    img2 = img2
    #Umwandlung Farbkanal
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # find Harris corners
    dst1 = cv2.cornerHarris(img1,2,3,0.04)
    dst1 = cv2.dilate(dst1,None)
    ret, dst1 = cv2.threshold(dst1,0.01*dst1.max(),255,0)
    dst1 = np.uint8(dst1)
    # find centroids
    ret, labels, stats, centroids1 = cv2.connectedComponentsWithStats(dst1)
    # convert coordinates to Keypoint type
    centroids1 = [cv2.KeyPoint(crd[0], crd[1], 13) for crd in centroids1]

    dst2 = cv2.cornerHarris(img2,2,3,0.04)
    dst2 = cv2.dilate(dst2,None)
    ret, dst2 = cv2.threshold(dst2,0.01*dst2.max(),255,0)
    dst2 = np.uint8(dst2)
    # find centroids
    ret, labels, stats, centroids2 = cv2.connectedComponentsWithStats(dst2)
    # convert coordinates to Keypoint type
    centroids2 = [cv2.KeyPoint(crd[0], crd[1], 13) for crd in centroids2]

    # extract descriptor  (BRIEF)
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp1, des1 = brief.compute(img1, centroids1)
    kp2, des2 = brief.compute(img2, centroids2)
     
    return kp1, des1, kp2, des2

def matching(des1, des2, img1, kp1, img2, kp2):   
    # Create a Brute Force Matcher object.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
        
    return matches

def findHomography(points1, points2): 
    # Find homography with RANSAC-based robust method
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    return h, mask

def stitching(input_path, output_path, firstROI, secondROI, rotation, blend_area):
    
    #Settings
    numRot = 2 #number of times cropped image is to be rotated    
    frame_list = []   
    i=2
    number=0
    j=1
    x=0
    region=1
    ROI_cropping1=firstROI #ROI for cropping before rotation
    ROI_cropping2f=secondROI #fine cropping after rotation
    video_name = input_path                                        
    num_stitched_frames=1                                                                                                                                         
    save_name=output_path    #
    start_frame=0 
    col_min_sum=0
    #num_iteration=0
    
    #filling frame_list before the sititching starts so its not empty 
    while(number!=2):
        #  Open and play video, define windows
        #cap = cv2.VideoCapture(video_name+'.mp4') #.h264
        cap = cv2.VideoCapture(video_name)
        ret, frame = cap.read()

        if number%1==0:
            
            imgTemp=ROI(frame,ROI_cropping1)
            croppedTemp=np.rot90(imgTemp, numRot)
            croppedTemp=rotate_image(croppedTemp, rotation)
            cropped=ROI(croppedTemp.copy(),ROI_cropping2f) 

            frame_list.append(cropped)
            frame_list.append(cropped)
            frame_list.append(cropped)
                    
        number+=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    cap = cv2.VideoCapture(video_name)

    #safty Homography for beginning, if unfortunatlly alrady the first frame has problems in quality (overwritten in each step --> more accurate)
    frame_list[2]=np.matrix([[9.88371611e-01,  1.51930172e-02,  2.10478783e+01], [-4.52232771e-02,  1.01078022e+00, -1.84143853e+00], [-9.79741162e-05,  1.91518538e-05,  1.00000000e+00]])  #aus ROI1 mit einser steps
    #print('Starting stitching...')
    while(j!=30):
    
        #print('Stitching frame',j,'...')
                
        ret, frame = cap.read()

        # if Video stream ends
        if frame is None:
            
            #cv2.imwrite(save_name+'/stitch_'+str(region)+'.jpg',result[:,img2.shape[1]:result.shape[1]-img2.shape[1]])                        
            #frame = frame_list[3]
            
            #print('Stitching completed...')            
            
            #return
            break

        #Preprocessing input image until cropped and rotated in right position (paramters from calibration)
        imgTemp=ROI(frame,ROI_cropping1)
        croppedTemp=np.rot90(imgTemp, numRot)
        croppedTemp=rotate_image(croppedTemp, rotation) #-3
        cropped=ROI(croppedTemp,ROI_cropping2f)
                       
        #parameters start frame (with which frame to start with stitching) and num_stitched_frames(1=every frame ,2=every second frame will bestitchted, ...) in normal case not important
        #possibility to speed process up
        if j>=start_frame and j%num_stitched_frames==0:

            #img1 last part of stitching which will be stitchted with new frame input of img2
            #h_old is safty Homography from last step when problem with finding enough matches                                   
            img1=frame_list[1]
            img2=cropped
            result_old=frame_list[0]
            h_old=frame_list[2]
            
            #Deetector
            kp1, des1, kp2, des2= detector_descriptor(img1, img2)
            
            #Matcher
            matches= matching(des1, des2, img1, kp1, img2, kp2)
            
            # Extract location of good matches
            points1 = np.zeros((len(matches), 2), dtype=np.float32)
            points2 = np.zeros((len(matches), 2), dtype=np.float32)            
            
            for i, match in enumerate(matches):
                points1[i, :] = kp1[match.queryIdx].pt
                points2[i, :] = kp2[match.trainIdx].pt
        
            # Find homography
            h, mask = findHomography(points2, points1)
            
            #safty to check if good warping possible
            if len(matches)<= 10:
                h=h_old
                x+=1
                            
            width = img1.shape[1] + img2.shape[1] 
            height = img1.shape[0]

            # Warping with new appraoch to alligne in a stable and plane way
            out = cv2.warpPerspective(img2, h, (width,height))
            #out_trim=out[out.shape[0]+150:out.shape[0]-150,0:out.shape[1]]
            out_trim=out[200:350,0:out.shape[1]]

            #col_min = movement step between img1 and img2
            col_min,col_max = col_crop_black(out_trim)  

            col_min_sum+=col_min                 
           
            #complex alpha blend with reduced area           
            blend_img1=img1[:,col_min:col_min+blend_area]
            blend_img2=img2[:,0:blend_area]

            if blend_img1.shape[0] > 0 and blend_img1.shape[1] > 0 and blend_img1.shape[1] == blend_img2.shape[1]:

                #gradiation_h_short.jpg image need to be in working directory where stitcher functions is saved
                #with this file possibility to change blending disposition if needed 
                mask1 = np.array(Image.open('gradation_h_short.jpg').resize(blend_img1.shape[1::-1], Image.BILINEAR))
                mask1 = mask1 / 255

                #blended result of img1 and img2
                blend = blend_img1*mask1+blend_img2*(1-mask1)                

                #make result  image bigger so new stichted part fits in it then copy pace it at the end
                pre_result = cv2.copyMakeBorder(img1.copy(),0,0,0,int(col_min), cv2.BORDER_CONSTANT)               
                pre_result[0:img1.shape[0],col_min:col_min+img2.shape[1]]=img2.copy()                
                pre_result[0:img1.shape[0],col_min:col_min+blend_area]=blend                

                result = cv2.copyMakeBorder(result_old.copy(),0,0,0,pre_result.shape[1], cv2.BORDER_CONSTANT)
                result[0:img1.shape[0],result_old.shape[1]:result_old.shape[1]+pre_result.shape[1]]=pre_result.copy()                

                #save needed parts for next iteration in frame_list
                frame_list[0]=result[0:img1.shape[0],0:result.shape[1]-img2.shape[1]]            
                frame_list[1]=result[0:img2.shape[0],result.shape[1]-img2.shape[1]:result.shape[1]] #anstatt img2                
                frame_list[2]=h
                #frame_list[3]=frame                      
            

        j+=1

    #print(col_min_sum)
    blend_area=((col_min_sum/(j-1))*25)/13.75
    print('Your Parameter for blend_area is', ((col_min_sum/(j-1))*25)/13.75)
    return blend_area


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def ROI(img_old,ROI_list):
    return img_old[int(ROI_list[1]):int(ROI_list[3]), int(ROI_list[0]):int(ROI_list[2])]

def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        #cv2.circle(img,(x,y),100,(255,0,0),-1)
        mouseX,mouseY = x,y

def filterImgs(raw_image, mode):
    #  Filter images
    #  mode 0 = Sobel, mode 1 = Laplace, mode 2 = Canny
    #  Change color
    gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    #  Smooth image
    smooth_img = cv2.bilateralFilter(gray, 9, 100.0, 100.0)
    ## Sobel
    if mode == 0:
        filtered_image_x = cv2.Sobel(smooth_img, cv2.CV_16S, 1, 0, 9) # Pixel depth value -> muss gewisse größe haben und mehr zu entdecken
        filtered_image_y = cv2.Sobel(smooth_img, cv2.CV_16S, 0, 1, 9)
        filtered_image = 0.25 * filtered_image_x + 1 * filtered_image_y
    #  Laplacian
    elif mode == 1:
        filtered_image = cv2.Laplacian(smooth_img, cv2.CV_16S, 5) 
    #  Canny
    elif mode == 2:
        filtered_image_canny = cv2.Canny(smooth_img, 0, 10, 3) #min, max thershold
        kernel = np.ones((7, 7), np.uint8)
        filtered_image = cv2.dilate(filtered_image_canny, kernel)
    else:
        pass
    return filtered_image

def match(offset_y, cropped_width, cropped_height, steps_h, imgEdges_list):
    # Match image areas for stitching
    ROI_stitching_old_image = [0, offset_y, cropped_width,cropped_height]  # x1,y1,x2,y2
    ROI_stitching_new_image = [0, 0, cropped_width, cropped_height-offset_y]  # x1,y1,x2,y2
    # For Loop for vertical matching
    for v in range(steps_v):
        ROI_stitching_old_image[0] = 0
        ROI_stitching_old_image[2] = cropped_width - steps_h/2
        ROI_stitching_new_image[0] = steps_h/2
        ROI_stitching_new_image[2] = cropped_width
        # For Loop for horizontal matching
        for h in range(steps_h):
            difference_abs = cv2.absdiff(ROI(imgEdges_list[0], ROI_stitching_old_image),ROI(imgEdges_list[1], ROI_stitching_new_image))

            diff_matrix[v, h] = np.sum(difference_abs) / ((ROI_stitching_old_image[2] - ROI_stitching_old_image[0]) * (ROI_stitching_old_image[3] - ROI_stitching_old_image[1]))  # Sum of the differece matrix divided by the number of compared pixels
            # Shift the images over the ROI
            if ROI_stitching_new_image[0] > 0:
                ROI_stitching_old_image[2] += 1
                ROI_stitching_new_image[0] -= 1
            else:
                ROI_stitching_old_image[0] += 1
                ROI_stitching_new_image[2] -= 1
        ROI_stitching_old_image[1] += 1
        ROI_stitching_new_image[3] -= 1
    return diff_matrix


def stitch(diff_matrix, copy_pos_x, copy_pos_y, steps_h, offset_y, imgStitched_list, cropped_width, cropped_height):
    min_diff_matrix, max_diff_matrix, min_diff_matrix_pos, max_diff_matrix_pos = cv2.minMaxLoc(diff_matrix)
    copy_pos_x += min_diff_matrix_pos[0] - steps_h/2
    #copy_pos_y += offset_y + min_diff_matrix_pos[1]
    copy_pos_y += offset_y + min_diff_matrix_pos[1]
    border_right = 0
    border_left = 0
    border_bottom = 0
    if copy_pos_x < 0:
        border_left = int(0.1 * np.shape(imgStitched_list[0])[1]) #?? 0.1 eeinfach mal so festgelegt?? Also Abschätzung?? -> wieso nicht direkt border_left = int(abs(copy_pos_x))
        if border_left < abs(copy_pos_x):
            border_left = int(abs(copy_pos_x))         
        copy_pos_x += border_left
    elif copy_pos_x + cropped_width > np.shape(imgStitched_list[0])[1]:
        border_right = int(0.1 * np.shape(imgStitched_list[0])[1])
        if border_right < copy_pos_x:
            border_right = int(copy_pos_x)
    if copy_pos_y + cropped_height > np.shape(imgStitched_list[0])[0]:
        border_bottom = int(0.1 * np.shape(imgStitched_list[0])[0])
        if border_bottom < copy_pos_y + cropped_height - np.shape(imgStitched_list[0])[0]:
            border_bottom = int(copy_pos_y + cropped_height - np.shape(imgStitched_list[0])[0])
            
    if border_bottom > 0 or border_left > 0 or border_right > 0:
        imgStitched_list.append(cropped)
        #cv2.imshow('cropped', cropped)
        imgStitched_list[1] = cv2.copyMakeBorder(imgStitched_list[0], int(0), border_bottom, border_left, border_right, cv2.BORDER_CONSTANT)#, imgStitched_list[1], [0,0,0])
        del(imgStitched_list[0])

    return imgStitched_list, copy_pos_x, copy_pos_y, cropped_height, cropped_width, min_diff_matrix_pos, min_diff_matrix, max_diff_matrix


def copyImages(imgStitched_list, copy_pos_y, copy_pos_x, cropped_height, cropped_width, min_diff_matrix_pos):
# Copy images together
    c=cropped[int(cropped_height - min_diff_matrix_pos[1]):int(cropped_height), int(0):int(cropped_width)]
    #print(c)
    #cv2.imshow('cropped',c)
    imgStitched_list[0][int(copy_pos_y + cropped_height - min_diff_matrix_pos[1]):int(copy_pos_y + cropped_height),
                          int(copy_pos_x):int(copy_pos_x + cropped_width)] = cropped[int(cropped_height - min_diff_matrix_pos[1]):int(cropped_height), int(0):int(cropped_width)]
    return imgStitched_list

def blend(imgStitched_list, copy_pos_y, copy_pos_x, cropped_height, cropped_width, min_diff_matrix_pos):
    imgStitched_list[0][int(copy_pos_y):int(copy_pos_y + cropped_height - min_diff_matrix_pos[1]),
                          int(copy_pos_x):int(copy_pos_x + cropped_width)] = 0.5 * cropped[int(0):int(cropped_height - min_diff_matrix_pos[1]), int(0):int(cropped_width)]\
                               + 0.5 * imgStitched_list[0][int(copy_pos_y):int(copy_pos_y + cropped_height - min_diff_matrix_pos[1]), int(copy_pos_x):int(copy_pos_x + cropped_width)]
    return imgStitched_list

def display(frame, imgStitched_list, diff_matrix, min_diff_matrix, max_diff_matrix, imgEdges_list):
    #cv2.imshow('frame',frame)

    filtered_image_rescaled = cv2.convertScaleAbs(imgEdges_list[1])  # Unexpected Behavior: Must be set as placeholder otherwise the next row throws an error
    cv2.normalize(cv2.convertScaleAbs(imgEdges_list[1]), filtered_image_rescaled, 0, 255, cv2.NORM_MINMAX)  # Rescale the image to 0...255
    #cv2.imshow('filtered', filtered_image_rescaled)

def getParameters(input_path):
    # Define Variables
    #video_name = 'D:/MA/v1_sync'
    frame_list = []  # Store Frames
    imgStitched_list = []  # Stores stitched images
    imgEdges_list = []  # Store edges
    steps_h = 40  # Stepsize horizontally - must be an even value
    global steps_v 
    steps_v = 10  # Stepsize vertically
    offset_y = 0  # Offset in y direction
    mode = 0  # Mode for edge detection, see filterImgs function
    global diff_matrix
    diff_matrix = np.zeros((steps_v,steps_h))
    copy_pos_x = 0  # Positions to copy image
    copy_pos_y = 0


    numRot = 0 #number of times cropped image is to be rotated
    # Windowsize ratio



    video_name=input_path
    #video_name='D:/MA/v1_sync.mp4'

    ratio=2
    res_width=1920/ratio
    res_height=1080/ratio
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', int(res_width), int(res_height))


    i=0
    c=0
    mouseX_list=[]
    mouseY_list=[]


    global mouseX
    global mouseY

    mouseX_list2=[]
    mouseY_list2=[]

    #video_name = 'D:/MA/v1_sync'
    cap = cv2.VideoCapture(video_name)
    ret, frame = cap.read()
    #frame=np.rot90(frame, 2)
    img=frame
    img_rec=frame

    cv2.setMouseCallback('img',draw_circle)    

    print('Select ROI1 with left double cklick on mouse and afterwards pressing -a- for upper left corner and then again lower right corner')
    while(i!=2):
        cv2.imshow('img',img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
        elif k == ord('a'):
            #print(mouseX,mouseY)
            mouseX_list.append(mouseX)
            mouseY_list.append(mouseY)
            i+=1

    cv2.destroyAllWindows()

    #print(mouseX_list)
    #print('mouseylist',mouseY_list)
    cv2.namedWindow('img_rec', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img_rec', int(res_width), int(res_height))

    img_rec=np.array(img_rec)
    img_rec=cv2.rectangle(img_rec, (int(mouseX_list[0]),int(mouseY_list[0])),(int(mouseX_list[1]),int(mouseY_list[1])), (0,255,0), 2)
    cv2.imshow('img_rec',img_rec)

    print('Is selected Region fine to continue? [y/n]')
    while(1):
        k = cv2.waitKey(20) & 0xFF
        if k == ord('y'):
            break
        elif k == ord('n'):
            print('Abort')
            sys.exit()
    print('Continue')

    cap = cv2.VideoCapture(video_name)
    ret, frame = cap.read()
    #frame=np.rot90(frame, numRot)
    img=frame

    x1=mouseX_list[0]
    y1=mouseY_list[0]
    x2=mouseX_list[1]
    y2=mouseY_list[1]

    rot_img=ROI(img,[x1,y1,x2,y2])

    # Direct stitchen
    ROI_cropping = [x1, y1, x2, y2]

    cap = cv2.VideoCapture(video_name)
    ret, frame = cap.read()

    #Initialization
    # Crop and Rotate Frame
    imgTemp = ROI(frame, ROI_cropping) # Size of cropped image
    global cropped
    cropped=np.rot90(imgTemp, numRot)

    # Append filtered images - initialize
    imgEdges_list.append(filterImgs(cropped, mode))
    imgEdges_list.append(filterImgs(cropped, mode))
    imgStitched_list.append(cropped)  # Initialize the list with the first frame
    count=0

    while(count!=90):
        print(count)
        ret, frame =cap.read()
        if not ret:
            break
        #c=frame
        # Crop and Rotate Frame
        imgTemp = ROI(frame, ROI_cropping) # Size of cropped image
        cropped=np.rot90(imgTemp , numRot)

        cropped_width = np.shape(cropped)[1]  # width of cropped image
        cropped_height = np.shape(cropped)[0]  # height of cropped image

        # Filter images an append to list
        imgEdges_list.append(filterImgs(cropped, mode))
        del imgEdges_list[0]

        match(offset_y, cropped_width, cropped_height, steps_h, imgEdges_list)

        imgStitched_list, copy_pos_x, copy_pos_y, cropped_height, cropped_width, min_diff_matrix_pos, min_diff_matrix, max_diff_matrix\
            = stitch(diff_matrix, copy_pos_x, copy_pos_y, steps_h, offset_y, imgStitched_list, cropped_width, cropped_height)
        
        copyImages(imgStitched_list, copy_pos_y, copy_pos_x, cropped_height, cropped_width, min_diff_matrix_pos)

        blend(imgStitched_list, copy_pos_y, copy_pos_x, cropped_height, cropped_width, min_diff_matrix_pos)

        display(frame, imgStitched_list, diff_matrix, min_diff_matrix, max_diff_matrix, imgEdges_list)

        count+=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('malte_result_test.jpg',imgStitched_list[0])
            break
    cv2.imwrite('zwischenstand_calibration.jpg',imgStitched_list[0])
    croppedTemp=rotate_image(imgStitched_list[0], 0)


    rot_img2=croppedTemp[:,int(croppedTemp.shape[1]/10):int(croppedTemp.shape[1]/2)]



    slope_img=cv2.cvtColor(rot_img2, cv2.COLOR_BGR2RGB)
    slope_img=cv2.cvtColor(rot_img2, cv2.COLOR_RGB2GRAY)



    ret, thresh_img = cv2.threshold(slope_img,5,255,cv2.THRESH_BINARY)


    start=thresh_img[:,0]
    middle=thresh_img[:,int(thresh_img.shape[1]/2)]
    end=thresh_img[:,thresh_img.shape[1]-1]

    pos_start=np.where(start==255)

    ypos1=pos_start[0][0]
    ypos4=pos_start[0][-1]

    pos_middle=np.where(middle==255)
    ypos2=pos_middle[0][0]
    ypos5=pos_middle[0][-1]

    pos_end=np.where(end==255)
    ypos3=pos_end[0][0]
    ypos6=pos_end[0][-1]

    xpos_start=0
    xpos_middle=int(thresh_img.shape[1]+1)/2
    xpos_end=thresh_img.shape[1]+1

    delta1=(ypos2-ypos1)/(xpos_middle-xpos_start)
    delta2=(ypos3-ypos2)/(xpos_end-xpos_middle)
    delta3=(ypos5-ypos4)/(xpos_middle-xpos_start)
    delta4=(ypos6-ypos5)/(xpos_end-xpos_middle)



    delta_average=(delta1+delta2+delta3+delta4)/(4)

    rotation=delta_average*57.496
    print(rotation)
    # angle=-4
    # thresh_img=rotate_image(thresh_img, angle)

    rot_img=rotate_image(rot_img, rotation)

    print('Select ROI2 with left double cklick on mouse and afterwards pressing -a- for upper left corner and then again lower right corner')

    del mouseX, mouseY

    cv2.namedWindow('rot', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('rot',draw_circle)    


    while(c!=2):
        cv2.imshow('rot',rot_img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
        elif k == ord('a'):
            #print(mouseX,mouseY)
            mouseX_list2.append(mouseX)
            mouseY_list2.append(mouseY)
            c+=1

    cv2.destroyAllWindows()

    x1_2=mouseX_list2[0]
    y1_2=mouseY_list2[0]
    x2_2=mouseX_list2[1]
    y2_2=mouseY_list2[1]


    img_final=ROI(rot_img,[x1_2,y1_2,x2_2,y2_2])
    #print(img_final.shape)
    cv2.imshow('img_final',img_final)


    print('Your parameters for firstROI are (x1,y1,x2,y2):', '['+str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+']' )
    print('Your parameters for secondROI are (x1,y1,x2,y2):', '['+str(x1_2)+','+str(y1_2)+','+str(x2_2)+','+str(y2_2)+']' )
    print('Your parameter for rotation is:', rotation)

    output_path='placeholder'
    firstROI=[x1,y1,x2,y2]
    secondROI=[x1_2,y1_2,x2_2,y2_2]
    blend_area=25
    #input_path='D:/MA/v1_sync.mp4'

    blend_area=stitching(input_path, output_path, firstROI, secondROI, rotation, blend_area)



def firstCalibration(video_path):

    getParameters(video_path)
    print('When choosing your parameters for ROI2, the difference between the fourth and second value should be a multiple of the patch size gridy')
    print('--> Example: [50,100,250,550]')
    print('end')