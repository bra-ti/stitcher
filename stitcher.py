import cv2
import numpy as np
import time
from PIL import Image
import os
import json
from area import *

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

def stitch(input_path, output_path, firstROI, secondROI, rotation, blend_area):
    
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
    print('Starting stitching...')
    while(True):
    
        print('Stitching frame',j,'...')
                
        ret, frame = cap.read()

        # if Video stream ends
        if frame is None:
            
            cv2.imwrite(save_name+'/stitch_'+str(region)+'.jpg',result[:,img2.shape[1]:result.shape[1]-img2.shape[1]])                        
            #frame = frame_list[3]
            
            print('Stitching completed...')            
            
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
            
            #Detector
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
           
            #complex alpha blend with reduced area           
            blend_img1=img1[:,col_min:col_min+int(blend_area)]
            blend_img2=img2[:,0:int(blend_area)]

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
                pre_result[0:img1.shape[0],col_min:col_min+int(blend_area)]=blend                

                result = cv2.copyMakeBorder(result_old.copy(),0,0,0,pre_result.shape[1], cv2.BORDER_CONSTANT)
                result[0:img1.shape[0],result_old.shape[1]:result_old.shape[1]+pre_result.shape[1]]=pre_result.copy()                

                #save needed parts for next iteration in frame_list
                frame_list[0]=result[0:img1.shape[0],0:result.shape[1]-img2.shape[1]]            
                frame_list[1]=result[0:img2.shape[0],result.shape[1]-img2.shape[1]:result.shape[1]] #anstatt img2                
                frame_list[2]=h
                #frame_list[3]=frame                      
            

            #after 1000 frames save as region pacakge
            if j == (region*1000):
                if region == 1:
                    cv2.imwrite(save_name+'/stitch_'+str(region)+'.jpg',result[0:img1.shape[0],img2.shape[1]:result.shape[1]-img2.shape[1]-img2.shape[1]])
                    frame_list[0]=result[0:img1.shape[0],result.shape[1]-img2.shape[1]:result.shape[1]-img2.shape[1]]
                    region+=1

                
                else: 
                    cv2.imwrite(save_name+'/stitch_'+str(region)+'.jpg',result[0:img1.shape[0],0:result.shape[1]-img2.shape[1]-img2.shape[1]])
                    frame_list[0]=result[0:img1.shape[0],result.shape[1]-img2.shape[1]:result.shape[1]-img2.shape[1]]
                    region+=1
            else:
                pass
                

        j+=1

def classificatorToJson(classificator_model, video_name, directory, folder_classification, json_file_path, gridx, gridy):
    from keras.models import load_model
    
   
    print('Starting classification...')

    model = load_model(classificator_model)
        
    i=1
    stitch_areas=[0]*10
    pitting_loc=[]    
    #img=cv2.imread(r'C:\Users\Tim\Documents\Studium\KIT\Master\Masterarbeit\python\Test Daten\stitch blend sammlung\stitch_v7_sync_32.jpg')

    #sort for read out folder in numerical order
    filelist = os.listdir(directory)
    filelist.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    for filename in filelist:
        print('Classification of',filename)
        img=cv2.imread(directory+'/stitch_'+str(i)+'.jpg')
        
        img2=img.copy()
        
        imageHeight, imageWidth, channel = img.shape
        # gridx=150
        # gridy=150
        rangex=int(imageWidth/gridx)
        rangey=int(imageHeight/gridy)
        #count = 1

        #sliding window with classification
        for x in range(rangex):
            for y in range(rangey):
                print('Classification of Column ',x,'...')
                                
                slice_bit=img[y*gridy:y*gridy+gridy, x*gridx:x*gridx+gridx]
               
                slice_bit=slice_bit.astype('float32')
                slice_bit /=255
                
                slice_bit = np.reshape(slice_bit,[1,gridy,gridx,3])

                classes = model.predict_classes(slice_bit)
               
                if classes == [1]:
                    #print('P')
                    img2=cv2.rectangle(img2, (x*gridy, y*gridy),(x*gridy+gridy, y*gridy+gridy),(0,255,0),2)

                    #saving coordinates of rectangles upper left corner 
                    #list collects every pitting coordinate
                    pitting_loc.append([x*gridy, y*gridy])

        cv2.imwrite(str(folder_classification)+'/classificated_'+str(i)+'.jpg',img2)
      
        #creating JSON

        #info extraction for saving in json
        total=rangex*rangey
        percent_10_area=imageWidth/10

        for list_pos in range(len(pitting_loc)):
            #size of 10% areas on region stitch
            area=int(pitting_loc[list_pos][0]/percent_10_area)
            
            #extracting x-,y-position upper left point of rectangle out of list pitting_loc for further processing and refinding in dummy function getArea()
            x=pitting_loc[list_pos][0]
            y=pitting_loc[list_pos][1]
           
            #dummy function for further more precise pitting area calculation, import from file area.py
            analysed=getArea(x,y,gridx,gridy,directory,i)

            stitch_areas[area]+=float(analysed)
      
        #structure of json per region
        y= {            
            str(video_name)+" region "+str(i): [
                {"p_total": len(pitting_loc)},
                {"share": len(pitting_loc)/total},
                {"areas": [
                    {"area": x+1, "p_total": float(stitch_areas[x]), "share": float(stitch_areas[x]/((percent_10_area/gridx)*rangey))} for x in range(10)               
                ]}                             
            ]           
        }

        #check if json already exists in folder otherwise build one 
        try:
            with open(json_file_path,'r+') as json_file:
                data = json.load(json_file)
                data.update(y)
                json_file.seek(0)
                json_file.truncate(0)            
                json.dump(data, json_file, indent=4)
                json_file.close()
                
        except IOError:

            with open(json_file_path,'w') as outfile:
                json.dump(y, outfile, indent=4)
                outfile.close()       

        #clearing for next iteration    
        del y 
        pitting_loc=[]
        
        stitch_areas=[0]*10
        
        i+=1    
       
def stitchAndClassify(input_path, firstROI, secondROI, rotation, blend_area, classificator_model, json, gridx, gridy):
    
    #extract file name from input path
    base=os.path.basename(input_path)
    base=os.path.splitext(base)[0]

    #create destination folders
    current_directory=os.getcwd()
    final_directory = os.path.join(current_directory, base)
    sub_stitch=os.path.join(current_directory, str(base)+'\stitched')
    sub_classify=os.path.join(current_directory, str(base)+'\classified')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
        os.makedirs(sub_stitch)
        os.makedirs(sub_classify)

    #parameter
    output_path=sub_stitch
    video_name=base
    directory=sub_stitch
    folder_classification=sub_classify
    json_file_path=str(final_directory)+'/'+str(json)
    # print(json_file_path)
    # print(type(json_file_path))
    # print(folder_classification)
    # print(sub_stitch)
    
    stitch(input_path, output_path, firstROI, secondROI, rotation, blend_area)
    classificatorToJson(classificator_model, video_name, directory, folder_classification, json_file_path, gridx, gridy)

    print('stitchAndClassify done.')


