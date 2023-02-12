from tkinter import*
from tkinter import ttk, messagebox

import pandas as pd
from PIL import Image
from PIL import ImageTk
import numpy as np
import Decision_Tree
import Neural_network
import preprocess_Dataset


def distinct():
    #path_china = 'C:/Users/Senpai/Documents/SP2/china.csv'
    #path_ksa = 'C:/Users/Senpai/Documents/SP2/SA_Aqar.csv'
    path_china = 'china.csv'
    path_ksa = 'SA_Aqar.csv'
    data_ = preprocess_Dataset.Dataset_(path_china, path_ksa)
    data_.read_()
    data_.district()
    data_.district_KSA.tolist()
    data_.district_china.tolist()
    x = np.concatenate((data_.district_KSA, data_.district_china))
    values = []
    for i in x:
        values.append(i)
    return values

root = Tk()
root.geometry("1000x700")
root.title("Housing Price Prediction")
root.configure(background='white')
Tops = Frame(root,bg="white",width = 1600,height=50,relief=SUNKEN)
Tops.pack(side=TOP)

f1 = Frame(root,width = 900,height=700,bg="white",relief=SUNKEN)
f1.pack(side=LEFT)

#-----------------INFO TOP------------
lblinfo = Label(Tops, font=( 'aria' ,30, 'bold' ),text="Housing Price Prediction",fg="steel blue",bg="white",bd=10,anchor='w')
lblinfo.grid(row=0,column=0)

def Ref():
    if years.get() == '' or txtdistrict.get()=='' or size.get()=='' or bathroom.get()=='' or kitchen.get()=='' or txtmodel.get()==''or txtelevator.get()=='':
        messagebox.showinfo("showinfo", "please enter all information")
        return
    vyears     = int(years.get())
    if vyears > 5:
        vyears=1
    else:
        vyears=0

    if txtelevator.get()=="Yes":
        velevator  = 1
    else:
        velevator = 0
    value = txtdistrict.get()
    vdistrict  = distinct().index(value)
    vsize      = float(size.get())
    vkitchen   = int(kitchen.get())
    vbathroom  = int(bathroom.get())
    vlivingRoom= int(livingRoom.get())
    if txtmodel.get() == "Decision Tree":
        ml_ = Decision_Tree.machine_learning(random_state=0)
        ml_.Build2_m()
        X_test = [vyears, vbathroom, vdistrict, velevator, vkitchen, vlivingRoom, vsize]
        output_ = ml_.predict([X_test])

    else:
        nn_ = Neural_network.neural_network()
        nn_.build2_model()
        X_test = [vyears, vbathroom, vdistrict, velevator, vkitchen, vlivingRoom, vsize]
        output_ = float(nn_.predict([X_test]))
    print(output_)
    price.set(float("{0:.4f}".format(abs(output_))))
 #  Property','bathRoom','district','elevator','kitchen','livingRoom','size'

years     = StringVar()
elevator  = StringVar()
size      = StringVar()
kitchen   = StringVar()
bathroom  = StringVar()
district  = StringVar()
livingRoom= StringVar()
price     = StringVar()
country   = StringVar()

lblbathroom = Label(f1, font=( 'aria' ,16, 'bold' ),bg="white",text=" No.bathroom.",fg="steel blue",bd=10,anchor='w')
lblbathroom.grid(row=0,column=0)
txtbathroom = Entry(f1,font=('ariel' ,16,'bold') ,textvariable=bathroom, bd=6,insertwidth=4,bg="powder blue" ,justify='center')
txtbathroom.grid(row=0,column=1)

lblkitchen = Label(f1, font=( 'aria' ,16, 'bold' ),bg="white",text="kitchen",fg="steel blue",bd=10,anchor='w')
lblkitchen.grid(row=1,column=0)
txtkitchen = Entry(f1,font=('ariel' ,16,'bold'),textvariable=kitchen, bd=6,insertwidth=4,bg="powder blue" ,justify='center')
txtkitchen.grid(row=1,column=1)

lblSize = Label(f1, font=( 'aria' ,16, 'bold' ),bg="white",text="Size",fg="steel blue",bd=10,anchor='w')
lblSize.grid(row=2,column=0)
txtSize = Entry(f1,font=('ariel' ,16,'bold'),textvariable=size, bd=6,insertwidth=4,bg="powder blue" ,justify='center')
txtSize.grid(row=2,column=1)


lblelevator = Label(f1, font=( 'aria' ,16, 'bold' ),bg="white",text="elevator",fg="steel blue",bd=10,anchor='w')
lblelevator.grid(row=3,column=0)
txtelevator = ttk.Combobox(f1,font=('ariel' ,16,'bold'),values=['Yes','No'] ,state= 'readonly',justify='center')
txtelevator.grid(row=3,column=1)

lblyears = Label(f1, font=( 'aria' ,16, 'bold' ),bg="white",text="years",fg="steel blue",bd=10,anchor='w')
lblyears.grid(row=4,column=0)
txtyears = Entry(f1,font=('ariel' ,16,'bold'),bg="powder blue", textvariable=years , bd=6,insertwidth=4 ,justify='center')
txtyears.grid(row=4,column=1)

lblcountry = Label(f1, font=( 'aria',16, 'bold' ),bg="white",text="Country",fg="steel blue",bd=10,anchor='w')
lblcountry.grid(row=5,column=0)
txtcountry= ttk.Combobox(f1,font=('ariel' ,16,'bold'),values=['China','KSA'],state= 'readonly',justify='right')
txtcountry.grid(row=5,column=1)

lbldistrict = Label(f1, font=( 'aria' ,16, 'bold' ),bg="white",text="District",fg="steel blue",bd=10,anchor='w')
lbldistrict.grid(row=6,column=0)
txtdistrict = ttk.Combobox(f1,font=('ariel' ,16,'bold'),values=distinct(),state= 'readonly' ,justify='center')
txtdistrict .grid(row=6,column=1)

lbllivingRoom = Label(f1, font=( 'aria' ,16, 'bold' ),bg="white",text="livingRoom",fg="steel blue",bd=10,anchor='w')
lbllivingRoom.grid(row=7,column=0)
txtlivingRoom = Entry(f1,font=('ariel' ,16,'bold'),bg="powder blue", textvariable=livingRoom , bd=6,insertwidth=4 ,justify='center')
txtlivingRoom.grid(row=7,column=1)



lblmodel= Label(f1, font=( 'aria' ,16, 'bold' ),bg="white",text="Choice of model",fg="steel blue",bd=10,anchor='w')
lblmodel.grid(row=8,column=0)
txtmodel= ttk.Combobox(f1,font=('ariel' ,16,'bold'),values=['Decision Tree','Neural Network'],state= 'readonly',justify='right')
txtmodel.grid(row=8,column=1)

#-----------------------------------------buttons------------------------------------------
btnpredict=Button(f1,padx=16,pady=8, bd=10 ,fg="black",font=('ariel' ,16,'bold'),width=10, text="predict", bg="powder blue",command=Ref)
btnpredict.grid(row=9, column=0)

btnprice= Label(f1,padx=16,pady=8, bd=10 ,fg="black",font=('ariel' ,16,'bold'),textvariable = price,width=10, text="0.00 ", bg="powder blue")
btnprice.grid(row=9, column=1)
#--------------------------------------------------------------------------
#img = Image.open("house.jpg")
#img = ImageTk.PhotoImage(img)
#panel = Label(root, bg="white",image = img)
#panel.pack(side = "right", fill = "both", expand = "no")

root.mainloop()