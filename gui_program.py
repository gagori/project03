import tkinter as tk
from tkinter import filedialog as fd
import os
from PIL import Image, ImageTk
from text_face_detection_complete import *
import pymongo

window = tk.Tk()





#몽고디비에 저장하기 위한 함수 만들기
def insert_data(text, name):
    myclient = pymongo.MongoClient('mongodb+srv://root:1234@cluster0.eoohb.mongodb.net/iddb?retryWrites=true&w=majority&tls=true&tlsAllowInvalidCertificates=true')
    mydb = myclient['iddb']
    id_info =mydb['info']
    info_dict = {
        "filename": name, 
        "id_info": text
    }
    
    data =id_info.insert_one(info_dict)
    print('data.insert_ids')



def showImage():
    file_name = fd.askopenfilename(initialdir=os.getcwd(), title="Select Image File", filetypes=(("JPG File", "*.jpg"),("All File","*.*")))
    
    file_name = rectangle_detect(file_name)+'.jpg'
    file_named = f'{file_name}'
    print(file_named)

    text = id_info(file_named)
    print(text)
    insert_data(text, file_named)

    img =Image.open(file_named)
    img.thumbnail((450,400))
    img = ImageTk.PhotoImage(img)
    lbl.configure(image=img)
    lbl.image = img





frm = tk.Frame(window)
frm.pack(side=tk.BOTTOM, padx=15,pady=15)

lbl = tk.Label(window)
lbl.pack()

btn = tk.Button(frm, text="비식별화", command=showImage)
btn.pack(side=tk.LEFT)


btn = tk.Button(frm, text="종료", command=lambda: exit()) # 파이썬 빠져나가는 함수.
btn.pack(side=tk.LEFT, padx=10)

window.title("ID DETECTOR")
window.geometry("640x450")




window.mainloop()

print("Window Close")




