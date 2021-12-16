from PIL.Image import NONE
from flask import Flask, redirect, request, render_template
import pymongo
from bson.objectid import ObjectId
from text_face_detection_complete import *
from datetime import datetime

myclient = pymongo.MongoClient('mongodb+srv://yjlee:admin1@de-identification.9zsqz.mongodb.net/iddb?retryWrites=true&w=majority&tls=true&tlsAllowInvalidCertificates=true')
app = Flask(__name__)
app.debug = True # 웹에 오류메시지 뜨게함.

#몽고디비에 저장하기 위한 함수만들기
def insert_data(text, name, original_name, info_type_name):
    print(myclient)
    mydb = myclient['iddb']
    id_info = mydb['info']
    info_dict = {
        'original_name':original_name,
        'filename': name,
        'id_info': text,
        'info_type_name' : info_type_name,
        'create_at': (datetime.now()).strftime("%Y-%m-%dT%H:%M:%S")
    }
    data = id_info.insert_one(info_dict)


@app.route('/', methods=['GET','POST'])
def index():
    # 몽고디비 조회해오기
    mydb = myclient['iddb'] 
    id_info = mydb['info']
    search = request.args.get('search')
    list_contents = []
    if search == None or search == " ":
        contents=id_info.find().sort("create_at", -1)
        list_contents = list(contents)

        for i in list_contents :
            i['show_id'] = str(i['_id'])
    else :
        contents = id_info.find({"original_name":search}).sort("create_at", -1)
        for i in contents:
            list_contents.append(i)


    return render_template('index.html', data=list_contents)

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
        infoTypeName = getType(f'static/img/{filename}')
        file_name = rectangle_detect(f'static/img/{filename}')
        print(file_name)
        text = id_info(f'static/img/{filename}')
        insert_data(text, file_name, filename, infoTypeName)
        return redirect('/')
        
if __name__ =='__main__':
    app.run(port=5001)