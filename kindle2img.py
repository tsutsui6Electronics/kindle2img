import cv2
import numpy as np
import os
import os.path as path
import sys

args = sys.argv

MOVIE_PATH=args[1]

#全画像データを格納するための配列
#指定動画ファイル

caption_data= cv2.VideoCapture(MOVIE_PATH)

img_arr=[]

n=120

c=0
F=0
org_img=None

while True:
    ret, frame = caption_data.read()
    
    if ret:
        frame=frame[90:956,510:1460]
        if F==0:
            F=1
            trans_img=frame.copy()
            trans_img=cv2.cvtColor(trans_img,cv2.COLOR_BGR2GRAY)
            trans_img[trans_img<n]=0
            trans_img[trans_img>n]=255
            org_img=trans_img
            img_arr.append(frame)
            
        else:
            trans_img=frame.copy()
            trans_img=cv2.cvtColor(trans_img,cv2.COLOR_BGR2GRAY)
            trans_img[trans_img<n]=0
            trans_img[trans_img>n]=255
            
            a=np.abs(np.where(trans_img==255)[0].shape[0]-np.where(org_img==255)[0].shape[0])
            
            #同じでない
            if a>100:                
                img_arr.append(frame)
                print("  {} |   {}   :   {}".format(str(c),np.where(trans_img==255)[0].shape[0],np.where(org_img==255)[0].shape[0]))
                
            org_img=trans_img
        c+=1

    else:
        break

f=0

index_arr=[]
for i,src in enumerate(img_arr):
    if f==0:
        obj=src
        f=1
    else:
        orb=cv2.ORB_create()
        key_src,desc_src=orb.detectAndCompute(src,None)
        key_obj,desc_obj=orb.detectAndCompute(obj,None)
        matcher=cv2.BFMatcher_create(cv2.NORM_HAMMING)
        all_match=matcher.match(desc_src,desc_obj)
        dist_list=list([m.distance for m in all_match])
        good_match=[[good] for good in all_match if good.distance<30]
        
        similarity=len(good_match)/len(key_obj)
        
        if similarity>0.4:
            index_arr.append(i-1)
            print("  delete {} : {}".format(i-1,i))
        obj=src
        
c=0
for g in index_arr:
    img_arr.pop(g-c)
    c+=1                                                                       

if len(sys.argv)<3:
	SAVE_PATH="PDF_SAVE"
else:
	SAVE_PATH=args[2]
f=1
while True:
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        break
    else:
        SAVE_PATH=SAVE_PATH+'_{}'.format(f)
        f+=1
    
for i in range(len(img_arr)):
    cv2.imwrite(os.path.join(SAVE_PATH,'img{}.png'.format(i)),img_arr[i])