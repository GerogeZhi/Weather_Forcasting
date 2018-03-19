import re
import requests
from bs4 import BeautifulSoup
import time

count,i=0,0
while(count<50):

    try:
        r=requests.get('https://book.douban.com/subject/1084336/comments/hot?p=' + str(i+1))
        r.status_code
    except Exception as err:
        print(err)
        break
    i += 1
    soup=BeautifulSoup(r.text,"lxml")
    comments=soup.find_all('p',"comment-content")
    for item in comments:
        count+=1
        print(item.string)
        if count==50:
            break
    pattern_s=re.compile('<span class="user-stars allstar(.*?) rating"')
    points=re.findall(pattern_s,r.text)
    y=[]
    for i in points:
        y.append(int(i))
    summer=sum(y)
    time.sleep(5)
    
if count==50:
    print(summer/len(points))
    





import requests
import re
def retrieve_djf_list():
    rest=requests.get('http://money.cnn.com/data/dow30/')
    patterns=re.compile('class="wsod_symbol">(.*?)<\/a>.*<span.*">(.*?)<\/span>.*\n.*class="wsod_stream">(.*?)<\/span>')
    in_text=re.findall(patterns,rest.text)
    return in_text
dji_list=retrieve_djf_list()
print(dji_list)


import requests
import re
rest=requests.get('http://money.cnn.com/data/dow30/')
pattern_p=re.compile('class="wsod_symbol">(.*?)<\/a>.*<span.*">(.*?)<\/span>.*\n.*class="wsod_stream">(.*?)<\/span>')
what_list=re.findall(pattern_p,rest.text)


shit=pd.DataFrame(what_list)
shit.columns=['code','name','lasttrade']
dates=pd.date_range('20170920',periods=7)
shit.index=dates
y=[]
for i in shit.lasttrade:
    y.append(float(i))
shit['lasttrade']=y
shit[(shit.index >= '2017-05-04')&(shit.lasttrade>=70.0)]

status=np.sign(np.diff(shit.lasttrade))
status[np.where(status==1)].size

temp=shit.sort_values(by='lasttrade',ascending='False')
temp[:3]

import matplotlib.pyplot as plt
shit.plot(kind='bar',stacked=True)



def countchar(s):
    lst=[0]*26
    for i in range(len(s)):
        if s[i]>'a' and  s[i]<'z':
            lst[ord(s[i])-ord('a')]+=1
    return lst


s='Hope is a good thing'
s=s.lower()
lst=countchar(s)

alist=['hello','world']
' '.join(alist)

def insert_line(lines):
    lines.insert(0,"Blowin' in the wind\n")
    lines.insert(1,"Bob Dylan\n")
    lines.append("\n 1962 by Warner Bros")
    return ''.join(lines)

with open('Blowing in the wind.txt','r+') as f:
    lines=f.readlines()
    string=insert_line(lines)
    print(string)
    f.seek(0)
    f.write(string)

sum=0
for i in range(2000):
    sum=sum+5*(10**i)
    yu=sum%84


#dictory
updata#直接更新
fromkeys#批量处理keys
something.get('')#获取
set(names)#去掉重复的

import requests
kw={'q':'python dict'}
r=requests.get('https://cn.bing.com/',params=kw)
print(r.text)


system={'zhiying':'zhiying','yingzi':'yingzi123','tom':'tom123'}

def newusers():
    print('your name')
    name=input()
    print('your password')
    password=input()
    system[name]=password
    print('wellcome newone')
def oldusers():
    print('your name')
    name=input()
    print('your password')
    password=input()
    if system[name]==password:
        print(name,'wellcome back')
    else:
        print('login incorrect')
def login():
    print('Are you already a client,say Y/N')
    clinet=input()
    if clinet == 'Y':
        oldusers()
    elif clinet == 'N':
        print('please enter the name')
        newusers()
    else:
        print('what r u doing')
if __name__=='__main__':
     login()    


import pandas as pd

music_data = [("the rolling stones","Satisfaction"),("Beatles","Let It Be"),("Guns N' Roses","Don't Cry"),("Metallica","Nothing Else Matters")]
music=pd.DataFrame(music_data,columns=('singer','song_name'),index=range(1,5))


shit=requests.get('https://api.douban.com/v2/movie/subject/1291546')
data=shit.json()




import numpy as np
from datetime import date
firstdat=date.fromtimestamp(1464010200)
dates=pd.date_range('20170920',periods=7)
datesdf=pd.DataFrame(np.random.randn(7,3),index=dates,columns=list('abc'))




import numpy as np
from sklearn.cluster import KMeans
X=np.array([list1,list2,list3])
kmeans=KMeans(n_cluster=2).fit(X)
pred=kmeans.predict(X)


from sklearn import datasets
from sklearn import svm
clf=svm.SVC(gamma=0.001,C=100.)
digits=datasets.load_digits()
clf.fit(digits.data[:-1],digits.target[:-1])
clf.predict(digits.data[-1])



import matplotlib.pyplot as plt
import numpy as np
x=np.linspace(-np.pi,np.pi,256)
s=np.sin(x)
c=np.cos(x)
plt.plot(x,s)
plt.plot(x,c)



import scipy as sp
listA=sp.ones(500)
listA[100:300]=-1
f=sp.fft(listA)
plt.plot(f)
plt.plot(listA)

#pillow opencv skimage python图像处理库
from PIL import Image
im1=Image.open('outdoor test.png')
print(im1.size,im1.format,im1.mode)
Image.open('outdoor test.png').save('2.png')
im2=Image.open('2.png')
size=(288,180)
im2.thumbnail(size)
out=im2.rotate(45)
im1.paste(out,(50,50))


from nltk.corpus import gutenberg
allwords=gutenberg.words('shakespeare-hamlet.txt')
len(allwords)
len(set(allwords))
allwords.count('Hamlet')
A=set(allwords)
longwords=[w for w in A if len(w)>12]

from nltk.probability import *
fd2=FreqDist([sx.lower() for sx in allwords if sx.isslpha()])
print(fd2.B())
print(fd2.N())
fd2.tabulate()
fd2.plot(20)
fd2.plot(20,cumulative=True)






class Dog(object):
    def greet(self):
       print('hi')     
dog=Dog()
dog.greet()
#self 这表明什么呢 这表示调用这个方法的对象自身 在调用时不需要实参跟它对应 


class Dog(object):
    def setname(self,name):
        self.name=name
    def greet(self):
        print('hi, I am called %s.'%self.name)
if __name__=='__main__':
    '''理解为程序入口 类似于main'''
    dog=Dog()
    dog.setname('paul')
    dog.greet()




class Dog(object):
    counter=0
    def __init__(self,name):
        self.name=name
        Dog.counter+=1
    def greet(self):
        print('hi, I am %s,my number is %d'%(self.name,Dog.counter))
if __name__=='__main__':
    '''创建完对象后，python自动调用地第一个方法为__init__()'''
    dog=Dog('tom')
    dog.greet()


class roster(object):
    teacher=""
    students=[]
    def __init__(self,tn='mayun'):
        self.teacher=tn
    def add(self,sn):
        self.students.append(sn)
    def remove(self,sn):
        self.students.remove(sn)
    def print_all(self):
        print('Teacher:',self.teacher)
        print('students:',self.students)



#sub class子类 (is-a parents)
#class SubClassName(ParentClass1[,ParentClass2,...]):
#    class_suite

class BarkingDog(Dog):
    def greet(self):
        print('woof! I am %s, my number is %d' %(self.name,Dog.counter))
if __name__=='__main__':
    dog=BarkingDog('Zoe')
    dog.greet()

'''Python类的成员属性和方法都是public的 但是你可以通过访问控制符来限定成员函数的访问 
有两个访问控制符 一个是双下划线 另一个是单下划线 简单来说 
双下划线就是 限定属性和方法在类内部可见 
单下划线 就是限定属性和方法在模块内可见 也就是不能通过类似的方式 被其他模块导入'''

















