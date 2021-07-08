from matplotlib import pyplot as plt
import numpy as np
import cv2
import glob
from mpl_toolkits import mplot3d
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d.axes3d import *
from matplotlib import cm
import time
import sys
from ajuste import *
from Newtoninter import *

 #inicio = time.time()
z11 = np.array([])
z22 = np.array([])
x11 = np.array([])
y = np.array([])


#fovea = ([cv2.imread("01.png"), cv2.imread("02.png"), cv2.imread("03.png"), cv2.imread("04.png"), cv2.imread("05.png"), cv2.imread("06.png"), cv2.imread("07.png"), cv2.imread("08.png"), cv2.imread("09.png"), cv2.imread("10.png"), cv2.imread("11.png"), cv2.imread("12.png"), cv2.imread("13.png"), cv2.imread("14.png")])
#fovea = [cv2.imread(file) for file in glob.glob("*.png")]
fovea = [cv2.imread(file) for file in glob.glob("C:/Users/PC/Documents/Proyecto/Proyecto/FOVEA/Fovea/LA23/P17/*.png")]

for i in range(len(fovea)):
    s0=0
    z = np.array([])
    zz = np.array([])
    x = np.array([])
    xx = np.array([])
    fovea[i] = fovea[i][0:396 , 550:950 ]
    plt.imshow(fovea[i])
    #plt.show()
    fovea[i] = np.uint8(fovea[i])
    fovea[i] = cv2.cvtColor(fovea[i], cv2.COLOR_BGR2RGB)
    fovea[i] = cv2.cvtColor(fovea[i], cv2.COLOR_RGB2GRAY)
    for r in range(1,len(fovea[i])):
        s0 = s0 + int(fovea[i][r-1,0])
        s1 = s0/(r)
        if int(fovea[i][r,0])-s1>90:
            c=r
            x=np.append(x,0)
            xx=np.append(xx,0)
            z=np.append(z,(len(fovea[i])-r)*2.52)
            zz=np.append(zz,r)
            z22=np.append(z22,(len(fovea[i])-r)*2.52)
            for j in range(r+1,len(fovea[i])):
                           fovea[i][j,0]=0
            for j in range(r):
             fovea[i][j,0]=0
            break
    for r in range(len(fovea[i][0])-1):
        k=40
        for j in range(c-4,c+10):
            u0=0
            for jj in range(c-15,c-1):
             u0 = u0+int(fovea[i][jj-1,r+1])
             u1 = u0/(jj-c+16) 
            if u1>20:
                k=70
            if int(fovea[i][j,r])-u1>k:
                c=j
                x=np.append(x,(r+1)*2.48)
                xx=np.append(xx,(r+1))
                z=np.append(z,(len(fovea[i])-j)*2.52)
                zz=np.append(zz,j)
                z22=np.append(z22,(len(fovea[i])-j)*2.52)
                for jj in range(j+1,len(fovea[i])):
                    fovea[i][jj,r]=0
                for jj in range(j):
                    fovea[i][jj,r]=0    
                break 
            if j==c+9:
              x=np.append(x,(r+1)*2.48)
              xx=np.append(xx,(r+1))
              z=np.append(z,(len(fovea[i])-c)*2.52)
              zz=np.append(zz,c)
              z22=np.append(z22,(len(fovea[i])-c)*2.52)
              for jj in range(c+1,len(fovea[i])):
                    fovea[i][jj,r]=0
              for jj in range(c):
                    fovea[i][jj,r]=0  
    #plt.imshow(fovea[i])
    #plt.show()

    zplt = np.array([],float)

    m=50
    f=polyFit(x,z,m)
    f0=polyFit(xx,zz,m)

    for x1 in x:
      p=f[0]
      for ii in range(1,m+1):
       p=f[ii]*x1**ii+p
      zplt = np.append(zplt,p)
      x11=np.append(x11,x1)
      z11=np.append(z11,p)
      y=np.append(y,(i*14)*2.48)
    #plt.plot (x,zplt,'k-')#,linewidth=0.5)#,x,z,'r-',linewidth=0.2)
    #plt.ylim([400,900])
    #plt.show()

    #plt.plot (x,z,'b-',linewidth=0.5)
    #plt.ylim([400,900])
    #plt.show()

    #plt.plot (x,zplt,'r-',x,z,'b-',linewidth=0.2)
    #plt.ylim([400,900])
    #plt.show()

    zz1=np.array([],float)

    for x1 in xx:
      p=f0[0]
      for ii in range(1,m+1):
       p=f0[ii]*x1**ii+p
      zz1=np.append(zz1,p)
    plt.plot (xx,zz1,'r-',linewidth=1)#,x,z,'r-',linewidth=0.2)
    #plt.ylim([400,900])
    plt.show()


x22 = np.array([])
z33 = np.array([])
y22 = np.array([])

for i in range(800):
    x22 = np.append(x22,x11[i])
    z33 = np.append(z33,z11[i])
    y22 = np.append(y22,y[i])

for i in range(4800,5600):
    x22 = np.append(x22,x11[i])
    z33 = np.append(z33,z11[i])
    y22 = np.append(y22,y[i])


#print(x11)
xi = np.linspace(min(x11), max(x11))
yi = np.linspace(min(y), max(y))
X, Y = np.meshgrid(xi, yi)
 
# interpolación

Z = griddata((x11, y), z11, (xi[None,:], yi[:,None]), method='cubic')
   

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter3D(x11, y, z11, cmap='Greens')
plt.show(block=False)
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,linewidth=1, antialiased=True)
plt.plot
plt.show(block=False)
#plt.show()


xii = np.linspace(min(x22), max(x22))
yii = np.linspace(min(y22), max(y22))
X1, Y1 = np.meshgrid(xii, yii)
 
# interpolación

Z1 = griddata((x22, y22), z33, (xii[None,:], yii[:,None]), method='linear')
   

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X1, Y1, Z1, rstride=1, cstride=1, cmap=cm.jet,linewidth=1, antialiased=True)
plt.plot
plt.show(block=False)
plt.show()

vol = 0
base = (xi[1]-xi[0])*(yi[1]-yi[0])

for i in range(len(Z)):
    for j in range(len(Z[1])):
        vol = vol + base*(Z[i,j]-Z1[i,j])

vol = abs(vol)
#print(vol)
h = abs(Z-Z1)
#print(np.amax(h))
prop = np.array([np.amax(h),vol])
print(prop)

#np.savetxt('correc1.csv', [prop], delimiter=',', fmt='%s')
#with open("carac2.csv", "ab") as f:
#    np.savetxt(f, [prop], delimiter=',', fmt='%s',newline='\n')
#fin = time.time()
#print(fin-inicio)