import pygame
from random import randint
from time import sleep
import numpy as np
import random
import os
director='C:\\Users\\ZEUS TASK\\desktop'
os.chdir(director)
from GANZER import *
pygame.init()
#Some Dimensions Windows
Height_Windows=450
Width_Windows=550
#Responsivity
rcoef=0.02
a=Width_Windows*rcoef
b=Height_Windows*rcoef
c=Width_Windows//3
d=Height_Windows//2
#Some Dimension Domain
AW=[a,c-a]
AH=[b,Height_Windows-b]
BW=[c+a,Width_Windows-a]
BHU=[b,d-b]
BHD=[d+b,Height_Windows-b]
# # #Display An Image(imgh)
def surf(example_z,id=0):
    if id==1:
        nda=np.random.randint(255, size=(128,128,3))
        return pygame.surfarray.make_surface(nda)
    try:
        imgh=USEME(False,example_z)
        return pygame.surfarray.make_surface(np.array(imgh))
    except:        
        imgh=USEME(False,[[0]*10])
        return pygame.surfarray.make_surface(np.array(imgh))
#Center the Image
XI=[c//2-64,Height_Windows//2-64]
#Some Colors
Windows_Colors=(205,200,100)
Cursors_Colors=(93, 104, 100)
Lines_Colors  =(0, 0, 237)
Texts_Colors  =(69,125, 127)
#Create My Surface
Windows=pygame.display.set_mode((Width_Windows,Height_Windows))
pygame.display.set_caption("LEAF MAKER")
#Text Situation
pygame.font.init()
font = pygame.font.SysFont("monospace", 23,bold=False, italic=True)
def textsurface(text):
    return font.render(text, False, Texts_Colors)
#My Functions 
def Creat_One_Cursor(posx,posy,posyi,posyf,taillex,tailley):
    if   posy<=posyi:posy=posyi
    if   posy>=posyf:posy=posyf
    pygame.draw.rect(Windows,Cursors_Colors,(posx,posy,taillex,tailley))
    tx=taillex//2
    cx=posx+tx
    #O_VER
    pygame.draw.line(Windows, Lines_Colors, (cx, posyi), (cx, posyf))
    #M_HOR
    zx=tx//2
    ux=zx//2
    W=posyf-posyi
    num=16
    frac=W/num
    for i in range(num+1):
        if i%2==0:pygame.draw.line(Windows, Lines_Colors, (cx-zx, posyi+i*frac), (cx+zx, posyi+i*frac))
        else:pygame.draw.line(Windows, Lines_Colors, (cx-ux, posyi+i*frac), (cx+ux, posyi+i*frac))
    return posx,posy,(posy-posyi)/W
def includity(a,B):
    c,d=B[0],B[1] 
    return a>=c and a<=d
def Clicked(x,y,width,height,prev_mouse_pos):
    cur_x,cur_y=prev_mouse_pos
    A=[x,x+width];B=[y,y+height]
    return includity(cur_x,A) and includity(cur_y,B)

def Creat_Many_Cursor(num,posxi,posxf,posyi,posyf,Lposy,taillex,tailley):
    W=posxf-posxi
    frac=int(W/num)
    Lposx=[posxi+i*frac for i in range(num)]
    if Lposy==None:Lposy=[posyi]*num
    Lpx =[]
    Lpy =[]
    Coef=[]
    for posx,posy in zip(Lposx,Lposy):
        Ht=Creat_One_Cursor(posx,posy,posyi,posyf,taillex,tailley)
        Lpx.append(Ht[0]);Lpy.append(Ht[1]);Coef.append(Ht[2])
    return posyi,posyf,Lpx,Lpy,Coef

def The_Clicked_One(Lpx,Lpy,Vect):
    t=-1
    for x,y in zip(Lpx,Lpy):
        t+=1
        if Clicked(x,y,taillex,tailley,prev_mouse_pos):
            Vect[t]=1;return Switch_One(Vect),True
def Switch_One(Vect):
    for i in range(len(Vect)):
        if Vect[i]==1:return i

def Creat_Ups_Cursor(num,BW,BHU,Lposy,taillex,tailley):
    posxi,posxf=BW[0],BW[1]
    posyi,posyf=BHU[0],BHU[1]
    return Creat_Many_Cursor(num,posxi,posxf,posyi,posyf,Lposy,taillex,tailley)
def Creat_Downs_Cursor(num,BW,BHD,Lposy,taillex,tailley):
    posxi,posxf=BW[0],BW[1]
    posyi,posyf=BHD[0],BHD[1]
    return Creat_Many_Cursor(num,posxi,posxf,posyi,posyf,Lposy,taillex,tailley)
#Some Variables
num,Lposyu,Lposyd=5,None,None
taillex,tailley=43,7
Vectu=[0]*num
Vectd=[0]*num
#Define The Game Loop 
Running=True
Action=False
OLD=[[0]*10]
Smurf=surf(OLD,1)
text="Noise"
Lurf=textsurface(text)
Lurf2=textsurface("0")
while Running:
    for event in pygame.event.get():
        if event.type==pygame.QUIT:Running=False
        keys=pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            example_z=[[0]*10]
            Lposyu=[randint(posyiu,posyfu) for i in range(num)] 
            Lposyd=[randint(posyid,posyfd) for i in range(num)]
            example_z=[Coefu+Coefd]
            Action=True 
            print(example_z)
        elif keys[pygame.K_UP]:
            print("Traitement...")
            Action=True
        elif event.type == pygame.MOUSEBUTTONDOWN:
            RESPU=The_Clicked_One(Lpxu,Lposyu,Vectu)
            try:
                xtu=RESPU[0]
                if RESPU[1]:
                    Lurf=textsurface("Pending...")
                    print(RESPU[0],"TOUCHED!")
            except:
                RESPD=The_Clicked_One(Lpxd,Lposyd,Vectd)
                if RESPD==None:continue
                xtd=RESPD[0]
                if RESPD[1]:
                    Lurf=textsurface("Pending...")
                    print(RESPD[0],"TOUCHED!")
        elif event.type == pygame.MOUSEMOTION:
            prev_mouse_pos=pygame.mouse.get_pos()
            if pygame.mouse.get_pressed()[0]:
                try:
                    if RESPU[1]:
                        _,Lposyu[xtu]=prev_mouse_pos
                        example_z=[Coefu+Coefd]
                        print(example_z)
                except:    
                    if RESPD==None or Lposyd==None:continue
                    elif RESPD[1]:
                        _,Lposyd[xtd]=prev_mouse_pos
                        example_z=[Coefu+Coefd]
                        print(example_z)
        elif event.type == pygame.MOUSEBUTTONUP:Vectu=[0]*num;Vectd=[0]*num
    Windows.fill(Windows_Colors)
    posyiu,posyfu,Lpxu,Lposyu,Coefu=Creat_Ups_Cursor(num,BW,BHU,Lposyu,taillex,tailley)
    posyid,posyfd,Lpxd,Lposyd,Coefd=Creat_Downs_Cursor(num,BW,BHD,Lposyd,taillex,tailley)
    if Action:
        ti=time.clock()
        Smurf=surf(example_z)
        tf=time.clock()
        text="New Image!"
        text2="  "+str(tf-ti)[:5]+"  "
        print(text)
        print(text2)
        Lurf=textsurface(text)
        Lurf2=textsurface(text2)
        Action=False
    Windows.blit(Smurf, (XI[0], XI[1]))
    Windows.blit(Lurf,(XI[0],XI[1]//2.9))
    Windows.blit(Lurf2,(XI[0],XI[1]//1.5))
    pygame.display.update()
pygame.quit()
