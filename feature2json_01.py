# -*- coding: utf-8 -*-
#!/usr/bin/python
########################################
##  author: maciu665
########################################
## extracting features from a
## series of images
## and storing their locations in json
########################################

#IMPORTY
import cv2
import os
from pathlib import Path
import random, string
import sys
import json

global punkt, nooks, oks, ims, halfsize
global imw, imh
nooks = 0
oks = 0
halfsize = 32

###
###

def click(event, x, y, flags, param):
    global punkt, nooks, oks
    if event == cv2.EVENT_LBUTTONDOWN:# or event == cv2.EVENT_RBUTTONDOWN:
  	     punkt = [x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        [px,py] = punkt

        minx = (px-halfsize)/imw
        maxx = (px+halfsize)/imw
        miny = (py-halfsize)/imw
        maxy = (py+halfsize)/imw

        print(minx,miny,maxx,maxy)
        newset[1].append([minx,miny,maxx,maxy])

        #imCrop = image[py-50:py+50, px-50:px+50]
        #cv2.imwrite( os.path.join(okdir,getrname()+".png"), imCrop )
        oks += 1
        print(oks,ims)
    '''
    elif event == cv2.EVENT_RBUTTONUP:
        [px,py] = punkt
        imCrop = image[py-50:py+50, px-50:px+50]
        cv2.imwrite( os.path.join(nookdir,getrname()+".png"), imCrop )
        nooks += 1
        print(oks,nooks)
    '''


imdir = "/home/maciej/DANE/CFD/272/"
outdir = "/home/maciej/DANE/CFD/272/WIR"



#print(getrname())
pliki = []
tpliki = os.listdir(imdir)
for i in tpliki:
    pliki.append(os.path.join(imdir, i))
print(len(pliki))
pnum = 0
ims = 1
plik = pliki[pnum]
print(plik)
image = cv2.imread(plik)

(imw,imh,imc) = image.shape
print(imw,imh,imc)



sys.exit()

###############

cv2.namedWindow("image")
cv2.setMouseCallback("image", click)


sets = []

newset = [plik,[]]

while True:
    # display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        if len(newset[1]):
            sets.append(newset[:])
        break
    elif key == ord("n"):
        print("next")
        if len(newset[1]):
            sets.append(newset[:])




        pnum += 1
        newset = [pliki[pnum],[]]
        print("newset",newset[0])
        image = cv2.imread(pliki[pnum])
        ims += 1

print(sets)

jdata = []

for imset in sets:
    tann = []
    for impt in imset[1]:
        tann.append({"label":["Wir"],"notes":"","points":[{"x":impt[0],"y":impt[1]},{"x":impt[2],"y":impt[3]}],"imageWidth":imw,"imageHeight":imh})
    jdata.append({'content':imset[0],"annotation":tann[:],"extras":None})


with open('/home/maciej/DANE/CFD/272/test.txt', 'w') as outfile:
    for jd in jdata:
        json.dump(jd, outfile)
        outfile.write('\n')


cv2.destroyAllWindows()
