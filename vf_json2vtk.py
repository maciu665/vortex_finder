import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import urllib
import os
import sys
from PIL import Image
import cv2
import math
import vtk

import json
#print(os.getcwd())

os.chdir("/home/maciej/DANE/UDEMY/DL01/retina_faces")


df = pd.read_json('lica1_-2-8.json', lines=True)
#df.head()
print(df.shape)
#print(df)
seria = df["series"][0]
resolution = df['resolution'][0][:2]
print("RES", resolution)
skeys = list(seria.keys())
print("SKEYS",len(skeys))
print(skeys[0])

k = skeys[0]

img = cv2.imread(k)
p = img.shape
print(p)
rows,cols,chan = img.shape

pix0 = img[0][0]

bx = [0,cols]
by = [0,rows]

midcol = int(cols/2)
midrow = int(rows/2)
#bounds
for i in range(midcol):
    tpix = img[midrow,i]
    acomp = tpix == pix0
    if not acomp.all():
        bx[0] = i
        break

for i in range(cols-1,0,-1):
    tpix = img[midrow,i]
    acomp = tpix == pix0
    if not acomp.all():
        bx[1] = i
        break

for i in range(midrow):
    tpix = img[i,midcol]
    acomp = tpix == pix0
    if not acomp.all():
        by[0] = i
        break

for i in range(rows-1,0,-1):
    tpix = img[i,midcol]
    acomp = tpix == pix0
    if not acomp.all():
        by[1] = i
        break

print(bx, by, bx[1]-bx[0], by[1]-by[0])
mpp = 5/(bx[1]-bx[0])
print("m per pixel",mpp)

korekta = [7,5]
corner = [2.5,5.0]

slicexs = []
slicepts = []

for i in range(0,1000):
    slicexs.append(-2+0.01*i)
    tslicepts = []
    k = skeys[i]
    for p in seria[k]:
        #mx my + korekta - offsets + 0.5 pixel
        mx,my = p["mx"]+korekta[0]-bx[0]+0.5,p["my"]+korekta[1]-by[0]+0.5
        px,py = mx*mpp,my*mpp
        tpt = [corner[0]-px,corner[1]-py]
        tslicepts.append(tpt[:])
    slicepts.append(tslicepts[:])
    #print(tslicepts)
#print(slicexs)
lpts = slicepts[:]
'''
def dist(p0,p1):
    return math.sqrt(math.pow(p1[0]-p0[0],2)+math.pow(p1[1]-p0[1],2))

def get_next_pt(lpt,lnum):
    pdist = 1000
    fpt = None
    next_layer_pt_ok = 0
    for n in range(len(lpts[lnum+1])):
        #print("n",n)
        tpdist = dist(lpt,lpts[lnum+1][n])
        if tpdist <= pdist:
            pdist = tpdist
            fpt = n
            #print("ppdist,)
    if fpt != None and pdist < 0.06:
        opdist = 1000
        ofpt = None
        for m in range(len(lpts[lnum])):
            otpdist = dist(lpts[lnum+1][fpt],lpts[lnum][m])
            if otpdist <= opdist:
                opdist = otpdist
                ofpt = m
        if ofpt != None:
            if opdist > pdist:
                next_layer_pt_ok = 1

    if next_layer_pt_ok:
        return fpt, pdist
    else:
        return -1,-1

pt_sets = [[]]
cnt = 0

while 1:
    go_on = 0
    for i in range(len(slicexs)):
        if len(lpts[i]):    #at least one point in layer
            go_on = 1
            #print("\n\nlen", len(lpts[i]))
            cl = i

            #print("pop0")
            #print(lpts[cl])
            #print(lpts[cl+1])

            lpt = lpts[cl].pop(0)   #first point
            #print(lpt)

            while 1:
                ### closest point in next layer
                #print("cl",cl)
                next_pt = get_next_pt(lpt,cl)
                #print("lpt, cl, npt", lpt, cl, next_pt,)
                #if next_pt[0] != -1:
                #    print(lpts[cl+1][next_pt[0]])
                #if cnt:
                    #cnt += 1
                #if cnt > 2:
                    #sys.exit()
                if next_pt[0] != -1:
                    #print("ok")
                    pt_sets[-1].append([slicexs[cl],lpt[0],lpt[1]])
                    cl += 1
                    lpt = lpts[cl].pop(next_pt[0])
                    #print(pt_sets[-1],lpt,next_pt)
                    #print()
                    #cnt = 1
                    #if len(pt_sets[-1]) >= 3:
                    #    sys.exit()
                else:
                    pt_sets.append([])
                    break

            #print(pt_sets[-1])
            #print("len", len(lpts[i]), len(pt_sets))
            #sys.exit()
    if not go_on:
        break

#sys.exit()
for pt_set in pt_sets:
    if len(pt_set) > 10:
        print(pt_set)


print(len(pt_sets))
'''
#
#sys.exit()

pts = vtk.vtkPoints()
pid = 0
for i in range(len(slicexs)):
    x = slicexs[i]
    if len(slicepts[i]):
        for pt in slicepts[i]:
            pts.InsertNextPoint([x,pt[0],pt[1]])

print(pts)

u = vtk.vtkUnstructuredGrid()
u.SetPoints(pts)
uw = vtk.vtkXMLUnstructuredGridWriter()
uw.SetFileName("test.vtu")
uw.SetInputData(u)
uw.Update()

sys.exit()

##
spts = vtk.vtkPoints()
spid = 0

su = vtk.vtkUnstructuredGrid()
su.Allocate(1, 1)

for p in range(len(pt_sets)):
    #print("pt_set",p)
    pt_set = pt_sets[p]
    if len(pt_set):
        npid = spts.InsertNextPoint(pt_set[0])
        pointIds = vtk.vtkIdList()
        for e in range(len(pt_set)-1):
            npid = spts.InsertNextPoint(pt_set[e+1])
            pointIds.InsertId(0, npid-1)
            pointIds.InsertId(1, npid)
            su.InsertNextCell(3, pointIds)

su.SetPoints(spts)
#print(su)

suw = vtk.vtkXMLUnstructuredGridWriter()
suw.SetFileName("linie.vtu")
suw.SetInputData(su)
suw.Update()

'''
print(skeys[500])
print(seria[skeys[500]])
k = skeys[500]
img = cv2.imread(k)

for p in seria[skeys[500]]:
    mx,my = p["mx"]+korekta[0],p["my"]+korekta[1]
    print(mx,my)
    #cv2.rectangle(img,(mx-1,my-1),(mx+1,my+1))
    color = (255,0,255)
    img = cv2.rectangle(img,(int(mx-1),int(my-1)),(int(mx+1),int(my+1)), color, 1)

cv2.imwrite("test.png", img)
'''
