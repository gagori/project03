from PIL.Image import NONE
from flask import Flask, redirect, request, render_template
import pymongo
from bson.objectid import ObjectId
from text_face_detection_complete import *


myclient = pymongo.MongoClient('mongodb+srv://root:1234@cluster0.eoohb.mongodb.net/iddb?retryWrites=true&w=majority&tls=true&tlsAllowInvalidCertificates=true')
app = Flask(__name__)
app.debug = True # 웹에 오류메시지 뜨게함.

#몽고디비에 저장하기 위한 함수만들기
def insert_data(text, name, original_name):
    print(myclient)
    mydb = myclient['iddb']
    id_info = mydb['info']
    info_dict = {
        'original_name':original_name,
        'filename': name,
        'id_info': text
    }
    data = id_info.insert_one(info_dict)
    print(data)


@app.route('/', methods=['GET','POST'])
def index():
    # 몽고디비 조회해오기
    mydb = myclient['iddb'] 
    id_info = mydb['info']
    contents=id_info.find()
    # print(contents)
    return render_template('index.html', data=contents)

@app.route('/edit/<id>', methods=['GET','POST'])
def edit(id):
    if request.method == 'POST':
        mydb = myclient['iddb'] 
        id_info = mydb['info']
        id_info.update_one({"_id":ObjectId(id)}, {"$set":{"id_info":request.form['id_info']}})
        return redirect('/')

    else:
        # 몽고디비 조회해오기
        mydb = myclient['iddb'] 
        id_info = mydb['info']
        contents=id_info.find_one({"_id":ObjectId(id)})
        # print(contents)
        return render_template('edit.html', data=contents)

@app.route('/de_identification', methods=['GET','POST'])
def upload():
    if request.method =='POST':
        filename=request.form['filename']
        print(filename)
        file_name = rectangle_detect(f'static/img/{filename}')
        print(file_name)
        text = id_info(f'static/img/{filename}')
        insert_data(text, file_name, filename)
        return redirect('/')

@app.route('/search_result')
def search_result():
    search = request.args.get('search')
    print(search)
    mydb = myclient['iddb'] 
    id_info = mydb['info']
    contents = id_info.find({"original_name":search})
    contents_list = []
    for i in contents:
        contents_list.append(i)
    print(contents_list)
    if contents_list:
        return render_template('search_result.html', data=contents_list)
    else:
        return render_template('search_result.html', data=None)



        
if __name__ =='__main__':
    app.run()