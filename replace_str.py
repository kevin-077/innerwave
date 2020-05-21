import re
f = open("/Users/Kev/Desktop/run02.gfs","r+")   
a=''
lines = f.readlines() 
#cf 在23行 
# #c0 在24行
# c='static double c0 ='+str(555.123)
# b=re.sub(r"static double c0 = (\d+\.\d+)",c,lines[24])
# lines[24]=b
# #c2 在25行

# c=str(0.444444)
# b1=re.sub(r"\d+\.\d+",c,lines[53])
# lines[53]=b1
def replace(i,c):
    c=str(c)
    b=re.sub(r"\d+\.\d+",c,lines[i])
    lines[i]=b
    return lines[i]
c1=0.33333
cf=0.11111
c0=0.22222
dhp=0.44444
l=0.55555
aaa=0.6666
w_bound_1=0.77777
bb=replace(24,c0)
cc=replace(25,c1)
dd=replace(26,dhp)
ee=replace(27,l)
ff=replace(28,aaa)
gg=replace(53,cf)

d='static double cf ='+str(0.32423532523)
bb=re.sub(r"static double cf = (\d+\.\d+)", d, lines[23])
lines[23]=bb
c='static double c2 ='+str(0.3265436345)
b1=re.sub(r"static double c2 = (\d+\.\d+)", c, lines[25])
lines[25]=b1
d='static double hp ='+str(0.7775667567)
bb=re.sub(r"static double hp = (\d+\.\d+)", d, lines[26])
lines[26]=bb
d='static double nm_2 ='+str(0.335435252)
bb=re.sub(r"static double nm_2 = (\d+\.\d+)", d, lines[27])
lines[27]=bb
d='static double hh  = '+str(0.6756724)
bb=re.sub(r"static double hh  = -(\d+\.\d+)", d, lines[28])
lines[28]=bb

d='0.01-a*sech(x/l)*sech(x/l)*'+str(0.99999)
bb=re.sub(r"0\.01-a\*sech\(x/l\)\*sech\(x/l\)\*(\d+\.\d+)", d, lines[58])
lines[58]=bb
d='0.09-a*sech(x/l)*sech(x/l)*'+str(0.88888)
bb=re.sub(r"0\.99-a\*sech\(x/l\)\*sech\(x/l\)\*(\d+\.\d+)", d, lines[58])
lines[58]=bb
for line in lines:
    a+=line
f.close()
data=open("/Users/Kev/Desktop/run03.gfs",'w+') 
print(a,file=data)



