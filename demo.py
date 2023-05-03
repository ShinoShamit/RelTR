from tkinter import *
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
import argparse
import math
import glob
import tensorflow as tf
import os
from test import pred_main

a = Tk()
a.iconbitmap("Project_Extra/icon.ico")
a.title("Scene Graph Generation")
a.geometry("740x600")
a.minsize(740,600)
a.maxsize(740,600)

def get_output():


    list_box.insert(1, "Loading Image")
    list_box.insert(2, "")
    list_box.insert(3, "Preprocessing")
    list_box.insert(4, "")
    list_box.insert(5, "Load Trained Model")
    list_box.insert(6, "")
    list_box.insert(7, "Prediction")

    pred_main(path)
    print("\n")
    print("Result saved to Output folder")

def Check():
    global f
    f.pack_forget()

    f = Frame(a, bg="white")
    f.pack(side="top", fill="both", expand=True)

    global f1
    f1 = Frame(f, bg="#97BC62")
    f1.place(x=0, y=0, width=500, height=690)
    f1.config()

    input_label = Label(f1, text="INPUT", font="arial 16", bg="#97BC62")
    input_label.place(x=220, y=20)    # input_label.pack(anchor=CENTER)

    upload_pic_button = Button(
        f1, text="Upload Picture", command=Upload, bg="#EA5D81")
    upload_pic_button.place(x=210, y=100)

    global label
    label=Label(f1,bg="#97BC62")

    f3 = Frame(f, bg="#2C5F2D")
    f3.place(x=500, y=0, width=240, height=690)
    f3.config()

    name_label = Label(f3, text="Process", font="arial 14", bg="#2C5F2D")
    name_label.pack(pady=20)

    global list_box
    list_box = Listbox(f3, height=12, width=31)
    list_box.pack()

    enhance_button = Button(
        f3, text="Generate", command=get_output, bg="#F1E577")
    enhance_button.place(x=90, y=300)


def Upload():

    global path
    label.config(image='')
    list_box.delete(0,END)
    path = askopenfilename(title='Open a file',
                           initialdir='Input',
                           filetypes=(("JPG", "*.jpg"), ("JPEG", "*.jpeg"),("PNG", "*.png")))
    image = Image.open(path)
    global imagename
    imagename = ImageTk.PhotoImage(image.resize((300,300)))
    # label = Label(f1, image=imagename)
    label.config(image=imagename)
    label.image = imagename
    label.pack(anchor=CENTER,pady=180)


def Home():
    global f
    f.pack_forget()

    f = Frame(a, bg="Aquamarine")
    f.pack(side="top", fill="both", expand=True)

    front_image = Image.open("Project_Extra/home1.jpg")
    front_photo = ImageTk.PhotoImage(front_image.resize((a.winfo_width(), a.winfo_height()), Image.ANTIALIAS))
    front_label = Label(f, image=front_photo)
    front_label.image = front_photo
    front_label.pack()

    home_label = Label(f, text="Scene Graph Generation",
                       font="arial 30", bg="white")
    home_label.place(x=150, y=250)


f = Frame(a, bg="Aquamarine")
f.pack(side="top", fill="both", expand=True)
front_image1 = Image.open("Project_Extra/home1.jpg")
front_photo1 = ImageTk.PhotoImage(front_image1.resize((740, 600), Image.ANTIALIAS))
front_label1 = Label(f, image=front_photo1)
front_label1.image = front_photo1
front_label1.pack()

home_label = Label(f, text="Scene Graph Generation",
                   font="arial 30", bg="white")
home_label.place(x=150, y=250)

m = Menu(a)
m.add_command(label="Homepage", command=Home)
checkmenu = Menu(m)
m.add_command(label="Test", command=Check)
a.config(menu=m)


a.mainloop()
