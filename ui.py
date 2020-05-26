from tkinter import *
sh=Tk() #创建Tk对象
sh.title("innnerwave") #设置窗口标题
sh.geometry("300x300") #设置窗口尺寸
l1=Label(sh,text="Hp") #标签
user_text=Entry() #创建文本框
l1.pack() 
user_text.pack()
sh.geometry("300x300") #设置窗口尺寸
l2=Label(sh,text="H") #标签
use_text=Entry() #创建文本框
l2.pack()
use_text.pack()
#指定包管理器放置组件
def number():
    user=user_text.get() #获取文本框内容
    usr=use_text.get()
    print(user,usr)
    top =Toplevel()
    top.title('calculate')
    v1 = StringVar()
    e1 = Entry(top,textvariable=v1,width=10)
    e1.grid(row=1,column=0,padx=1,pady=1)

    Button(top, text='计算中').grid(row=1,column=1,padx=1,pady=1)
Button(sh,text="登录",command=number).pack() #command绑定获取文本框内容方法
sh.mainloop() 