import csv,os
import math 
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
from scipy.optimize import curve_fit
from scipy.optimize import fminbound
from scipy import integrate
from scipy.interpolate import griddata
import json
from scipy import interpolate
import itertools
import re

print("""
* * * * * * * * * * * * * * * * * * * *  *
*                                        *
*                                        *
*           内波振幅反演算法             *     
*          Author:  songhao              *   
*           Version: demo1               *  
*           Data: 2020-5-11              *   
*                                        *   
*                                        *   
* * * * * * * * * * * * * * * * * * * *  *

""")
file='D:/new/data.json'
with open(file,'r') as oj:
    c=json.load(oj)
c000=c.get('c000'); ct1=c.get('ct1'); ct2=c.get('ct2');ct3=c.get('ct3');cs1=c.get('cs1'); cs2=c.get('cs2'); cp1=c.get('cp1'); cp2=c.get('cp2'); cp3=c.get('cp3'); ctp=c.get('ctp'); ct3p=c.get('ct3p'); ctp2=c.get('ctp2'); ct2p2=c.get('ct2p2'); ctp3=c.get('ctp3'); cst=c.get('cst'); cst2=c.get('cst2'); cstp=c.get('cstp'); cs2tp=c.get('cs2tp'); cs2p2=c.get('cs2p2'); g=c.get('g');
####test.m 模块
print("""请输入以下信息 ： """)

a4=input('Enter the deep: ');
a5=input('Enter the H: ');
a6=input('Enter the A: ');
print("""运算中...... """)
xi=[i for i in range(-int(a4),1,1)]              #从-a4 取到0 ，步长为1
pi=math.pi
h=[] #深度
x=[] #密度
s=[] #盐度
t=[] #温度
with open('D:/new/2.csv',encoding="utf-8") as f:
    reader = csv.reader(f)
    header_row = next(reader)
    datas = []
    for row in reader:
        h.append(float(row[0]))             #读取水深
        x.append(float(row[1])+1000)         #读取密度
        s.append(float(row[3]))             #读取盐度
        t.append(float(row[2]))             #读取温度
f.close()
p=np.zeros(len(x))
rhop=[]
i=1
xx=[]
hh=[]
for i in range(len(x)): 
    xxx=(x[i]+x[i-1])/2
    i+=i
    xx.append(xxx)                      #密度差
j=1
for j in range(len(h)-1): 
    hhh=(h[j+1]-h[j])                   #高度差
    j+=j
    hh.append(hhh)                      #!高度差只有55个数值
for ii in range(0,len(p)): 
    p[ii]=p[ii-1]+hh[ii-1]*xx[ii-1]*g       #压强
pp=[i*0.00001 for i in p]                 #压强单位换算
for i in range(0,len(x)):
    rhop1=(x[i]-x[i-1])/(p[i]-p[i-1])       
    rhop.append(rhop1)                  #密度对压强求导
N=[]
Ct=[]
Cs=[]
Cp=[]
Cstp=[]
cc=[]
for T,S,P in zip(t,s,pp):                    
    ctt=ct1*T+ct2*T**2+ct3*T**3
    Ct.append(ctt)
    css=cs1*S+cs2*S**2
    Cs.append(css)
    cpp=cp1*P+cp2*P**2+cp3*P**3
    Cp.append(cpp)
    cstpp=ctp*T*P+ct3p*T**3*P+ctp2*T*P**2+ct2p2*T**2*P**2+ctp3*T*P**3+cst*S*T+cst2*S*T**2+cstp*S*T*P+cs2tp*S**2*T*P+cs2p2*S**2*P**2
    Cstp.append(cstpp)
cc=[c000+i+j+k+m for i,j,k,m in zip(Ct,Cs,Cp,Cstp)]                #计算声速
ccc=[1.0/(i**2) for i in cc]    
n=[i-j for i,j in zip(rhop,ccc)]
nn=[g*math.sqrt(i) for i in n]
def func1(x,a,b,c):
    return (a*b**2)/(4*(c+x)**2+b**2)
popt,pcov=curve_fit(func1,h,nn,maxfev=50000)                        #曲线拟合
#####Kdv solution.m 模块
##计算c1，c2，和cf
x1=[i for i in np.arange(-1,0,0.01)]      #从-1到0，步长为0.01，取值为列表x1的元素
a1=popt[0]
a2=popt[1]
a3=popt[2]
Hp =float(a2)             
dHp=float(a3)
Nm=float(a1)
H =float(a5)
A =float(a6)
c1=(dHp/H)/2
c2=(Hp/H)
dhp=(dHp/H)
hp=(Hp/H)
nm_2=(Nm*Nm*H)/9.81
aaa=A/H

###积分计算 gama
def f(x):
    return (math.atan((x+c2)/c1)-math.atan(c2/c1))
h1=c1*f(-1)
hh1 = f(-1)
zz1=[]
for i in x1:
    zz1.append(f(i))
zzz1=[h1*i for i in zz1]
z1=[c1/i for i in zzz1]
c7=[2*i+2*c2 for i in x1]                   #80-88行求omiga-diff和w 
c8=[-2*i  for i in c7]
c9=[(((c2+i)/c1)**2+1)**3*(c1**2) for i in x1]
omiga_diff =[-i/j for i in c8 for j in c9]
c10=[math.sin(i*pi) for i in z1]
c11=[(((c2+i)/c1)**2+1) for i in x1]
w=[math.sqrt(i)*j for i in c11 for j in c10]
####找到最大的本征值     
# ###！！！！计算速度开始下降
def func (x2):
    return -math.sqrt(((x2+c2)/c1)**2+1)* math.sin((math.atan((x2+c2)/c1)-math.atan(c2/c1))/h1*c1*pi)   
x_max= fminbound(func,-1,0)
c0 = math.sqrt(-func(x_max))
w_2=[i**2 for i in w]
w_3=[i**3 for i in w]
omiga_w=[i*j for i in omiga_diff for j in w_3]
gama_down = 0.0001*np.trapz(w_2)
gama_up = 0.0001*np.trapz(omiga_w) 
cccc0=c0
#####计算bet0
bet0=(pi/h1)**2-1/c1**2
gama=-bet0*gama_up/gama_down/c0

#####积分计算alpha
c12=[i**2 for i in c11]
omiga_org_2=[1/i for i in c12]
omiga_org_w_2=[i*j for i in omiga_org_2 for j in w_2]
alpha_up=0.0001*np.trapz(omiga_org_w_2)
alpha=alpha_up/gama_down

##计算速度
Cf0 = Nm*H/math.sqrt(bet0)
cf = Cf0*(1-A*gama/3/alpha/H/bet0)

##计算水平特征尺度
ll=6*H*H*H/A/gama
if ll<0:
    L=math.sqrt(-ll)
else:
    L=math.sqrt(ll)
l=L/H

###画出归一化本征函数

# figure;
# set(gca,'ticklength',[0 0]);
# set(gca,'xtick',[]);
# set(gca,'ytick',[]);
# hold on;
# AdsorptionAxes = gca;
# AdAxesEdit = axes('Position',get(AdsorptionAxes,'Position'),'XAxisLocation','top','YAxisLocation','left');
# set(AdAxesEdit,'XTick',[],'YTick',[]);
# hold on;
# w_C0 = w/c0;
# plot(w_C0,x,'k','LineWidth',1.0);
# annotation('arrow',[0.1305 0.1305],[0.2 0]);
# annotation('arrow',[0.8 1],[0.925 0.925]);
# % hold on;
# % data = load('LHYPHI.txt');
# % plot(data(:,2)./10.4424,data(:,1)./300,'r')

##边界归一化本征值
w_bound_1= math.sqrt(((-0.01+c2)/c1)**2+1)*math.sin((math.atan((-0.01+c2)/c1)-math.atan(c2/c1))/h1*c1*pi)/c0
w_bound_2= math.sqrt(((-0.99+c2)/c1)**2+1)*math.sin((math.atan((-0.99+c2)/c1)-math.atan(c2/c1))/h1*c1*pi)/c0



####rho.m模块
x3 =float(-H*0.01)
x4 =float(-H*0.09)
d1 = math.atan(2*Hp/dHp)
k1 = -Nm**2*dHp/4/9.81
c1 = 2*dHp*Hp/(dHp**2+4*Hp**2)
ss=float(x[0]) 
def fun2(x):
    a1 = 2*dHp*(Hp+x)/(dHp**2+4*(Hp+x)**2)
    b1 = math.atan(2*(Hp+x)/dHp)
      
    return ss*k1*(a1-c1+b1-d1) 
y1=fun2(x3)
y2=fun2(x4)-y1

f1 = open('D:/new/42.txt','r')  #设置文件对象
lines1 = f1.readlines()
x5=[]
y5=[]
z5=[]
f1.close()
for line1 in lines1:
    a = float(line1.split(' ')[1])         #读取第二列      
    b = float(line1.split(' ')[2])         #读取第三列
    c = float(line1.split(' ')[8])         #读取第九列                   
    x5.append(a)   
    y5.append(b)
    z5.append(c)                         
xx0=np.linspace(float(min(x5)),float(max(x5)),200)    #从min（x5）到max（x5）取200个数
yy0=np.linspace(float(min(y5)),float(max(y5)),300)
xxx0=np.array([xx0 for i in range(300)])
yyy0=np.array([yy0 for i in range(200)]) 
coords = list(itertools.product(xx0,yy0))
yyyy0=yyy0.T
values=np.array(z5)
x,y=np.meshgrid(xx0,yy0)   #产生二维数组
# xii = np.concatenate(xii,axis=0)
# points = np.concatenate(points,axis=0)
# xx,yy = np.meshgrid(xx0,yy0)
zzz0=griddata(coords,values,(x,y),method='linear')    #将输入点设置为n维单纯形，并在每个单形上线性插值。
c0=plt.contour(xxx0,yyyy0,zzz0)
# plt.figure(figsize=(10,4))
#从等值线中提取坐标点和属性值

plt.close()
def get_contour_verts(cn):

    contours = []

    idx = 0

    # for each contour line

    #print(cn.levels)

    for cc,vl in zip(cn.collections,cn.levels):

        # for each separate section of the contour line

        for pp in cc.get_paths():

            paths = {}

            paths["id"]=idx

            paths["type"]=0

            paths["value"]=float(vl) # vl 是属性值

            xy = []

            # for each segment of that section

            for vv in pp.iter_segments():

                xy.append([float(vv[0][0]),float(vv[0][0])]) #vv[0] 是等值线上一个点的坐标，是 1 个 形如 array[12.0,13.5] 的 ndarray。

            paths["coords"]=xy

            contours.append(paths)

            idx +=1

    return contours

a=get_contour_verts(c0)
b=[]
for i in range(len(a)):
    a[i]=a[i].get('value')
    b.append(a[i])

c=[b[i+1]-b[i] for i in range(len(b)-1)]
A=max(c)*300

print("\n输出结果如下： ")
print("Nm= ",popt[0])
print("dHp= ",popt[1])
print("Hp= ",-popt[2])
print("c1= ",c1)
print("c2= ",c2)
print("cf= ",cf)
print("w_bound_1= ",w_bound_1)
print("w_bound_2= ",w_bound_2)
print("y1= ",y1)
print("y2= ",y2)
print("d1= ",d1)
print("A= ",A)

data=open("D:/new/output.txt",'w+') 
print("\n输出结果如下： ",file=data)
print("Nm= ",popt[0],file=data)
print("dHp= ",popt[1],file=data)
print("Hp= ",-popt[2],file=data)
print("c1= ",c1,file=data)
print("c2= ",c2,file=data)
print("cf= ",cf,file=data)
print("w_bound_1= ",w_bound_1,file=data)
print("w_bound_2= ",w_bound_2,file=data)
print("y1= ",fun2(x3),file=data)
print("y2= ",fun2(x4)-fun2(x3),file=data)
print("d1= ",d1,file=data)
print("A= ",A,file=data)
print("c0= ",c0,file=data)
data.close()

f = open("E:/Desktop/run02.gfs","r+")   
a=''
lines = f.readlines() 
def replace(i,c):
    c=str(c)
    b=re.sub(r"\d+\.\d+",c,lines[i])
    lines[i]=b
    return lines[i]
bb=replace(24,cccc0)
cc=replace(25,c1)
dd=replace(26,dhp)
ee=replace(27,l)
ff=replace(28,aaa)
gg=replace(53,cf)

d='static double cf ='+str(cf)
bb=re.sub(r"static double cf = (\d+\.\d+)", d, lines[23])
lines[23]=bb
c='static double c2 ='+str(c2)
b1=re.sub(r"static double c2 = (\d+\.\d+)", c, lines[25])
lines[25]=b1
d='static double hp ='+str(hp)
bb=re.sub(r"static double hp = (\d+\.\d+)", d, lines[26])
lines[26]=bb
d='static double nm_2 ='+str(nm_2)
bb=re.sub(r"static double nm_2 = (\d+\.\d+)", d, lines[27])
lines[27]=bb
d='static double hh  = '+str(hh1)
bb=re.sub(r"static double hh  = -(\d+\.\d+)", d, lines[28])
lines[28]=bb

d='0.01-a*sech(x/l)*sech(x/l)*'+str(w_bound_1)
bb=re.sub(r"0\.01-a\*sech\(x/l\)\*sech\(x/l\)\*(\d+\.\d+)", d, lines[58])
lines[58]=bb
d='0.09-a*sech(x/l)*sech(x/l)*'+str(w_bound_2)
bb=re.sub(r"0\.99-a\*sech\(x/l\)\*sech\(x/l\)\*(\d+\.\d+)", d, lines[58])
lines[58]=bb
d='nm_2*dhp/4*'+str(ss)
bb=re.sub(r"nm_2\*dhp\/4\*(\d+)",d,lines[58])
lines[58]=bb
d='-atan('+str(-c1)
bb=re.sub(r"-atan\((\d+)",d,lines[58])
lines[58]=bb
d=')-'+str(d1)
bb=re.sub(r"\)-(\d+\.\d+)",d,lines[58])
lines[58]=bb
d='))-'+str(y1)
bb=re.sub(r"\)\)-(\d+\.\d+)",d,lines[58])
lines[58]=bb
d=')/'+str(y2)
bb=re.sub(r"\)/(\d+\.\d+)",d,lines[58])
lines[58]=bb

for line in lines:
    a+=line
f.close()
data=open("E:/Desktop/run03.gfs",'w+') 
print(a,file=data)











