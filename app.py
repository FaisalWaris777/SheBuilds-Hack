from flask import Flask,render_template,redirect,request
import pickle
import cv2
import time
from keras import models 
import os
import base64
from PIL import Image
import base64
import io

app = Flask(__name__)

@app.route("/heart_disease")
def heart_disease():
    return render_template("heart_disease.html")

@app.route('/heart_disease_result', methods=['POST'])
def heart_disease_result():
    age = request.form['age']
    cpt = request.form['chest-pain-type']
    mhra = request.form['maximum-heart-rate-achieved']
    eia = request.form['exercise-induced-angina']
    oldpeak = request.form['oldpeak']
    nomv = request.form['number-of-major-vessels']
    thal = request.form['thal']
    user_input=[[age,cpt,mhra,eia,oldpeak,nomv,thal]]
    # print(age)
    model=""
    with open('Heart_Disease_Prediction','rb') as file:
      model = pickle.load(file)
    output=model.predict(user_input)
    if output>0.7:
     output="Yes, you are having Heart Disease"
    else:
        output="No, you are not having Heart Disease"
    # return("hello")  
    return render_template("heart_disease_result.html",result=output)

@app.route("/brain_tumor")
def brain_tumor():
    return render_template("brain_tumor.html")

@app.route('/brain_tumor_result', methods=['POST'])
def brain_tumor_result():
    name = request.form['name']
    age = request.form['age']
    # img = request.form["image"]
    img='1 no.jpeg'
    path="brain_tumor_dataset/"+str(img)
    img=cv2.imread(path)
    img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100,100))
    user_input = img.reshape(1,10000)
    model = models.load_model("Brain_tumor_Detection.h5")    
    output=int(model.predict(user_input)[0])
    if output==1:
     output="Yes, you are having Brain Tumor"
    else:
        output="No, you are not having Brain Tumor"
    im = Image.open(path)
    data = io.BytesIO()
    im.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())
    return render_template("brain_tumor_result.html",result=output)

@app.route("/diabetes")
def get_diabetes():
    return render_template("diabetes.html")

@app.route('/diabetes_result', methods=['POST'])
def diabetes():
    age = request.form['age']
    preg = request.form['pregnancies']
    glu = request.form['glucose']
    bpress = request.form['bloodpressure']
    skthick = request.form['skinthickness']
    insu = request.form['insulin']
    bm = request.form['bmi']
    dpgf = request.form['diabetespedigreefunction']
    user_input=[[preg,glu,bpress,skthick,insu,bm,dpgf,age]]
    print(age)
    
    #result # 0- Not diabetes
            # 1- diabetes
    
            
    diab=pickle.load(open('Diabetes.pkl','rb'))
    output=diab.predict(user_input)
    #print(output[0])
    # return("hello")
    if output[0]==0:
        output="You are not diabetic. "
    else:
        output="You are diabetic."
    return render_template("diabetes_result.html",result=output)


@app.route("/alzheimer")
def get_alzheimer():
    return render_template("alzheimer.html")

@app.route('/alzheimer_result', methods=['POST'])
def alzheimer():
    # for single test data
    name = request.form['name']
    age = request.form['age']
    img = request.form["image"]
    path="datasetR/"+str(img)
    img = Image.open(path)#just putting image_directory or file does not work for google colab, interesting. 
                                    #preserve aspect ratio
    # img = Image.open('verymild_1007.jpg')
    # img = Image.open('mild_100.jpg')
    # img = Image.open('moderate_48.jpg')
    
    #result # 0- Non Demented
            # 1- Very Mild Demented
            # 2- Mild Demented
            # 3- Moderate Demented
            
    
    width = 256
    height = 256
    new_size = (width,height)
    img = img.resize(new_size)
    array_temp = np.array(img)
    shape_new = width*height
    img_wide = array_temp.reshape(1, shape_new)
    
    alz=pickle.load(open('Alzheimer.pkl','rb'))
    output=alz.predict(img_wide)
    print("your result is here")
    print(output[0])
    # return("hello")
    im = Image.open(path)
    data = io.BytesIO()
    im.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())
    return render_template("alzheimer_result.html",img_data=encoded_img_data.decode('utf-8'))
    return render_template("alzheimer_result.html",result=output[0])



@app.route("/pneumonia")
def get_pneumonia():
    return render_template("pneumonia.html")

@app.route('/pneumonia_result', methods=['POST'])
def pneumonia():
    # for single test data
    # single test data
    # img_arr = cv2.imread('person1008_virus_1691.jpeg', cv2.IMREAD_GRAYSCALE)
    img_arr = cv2.imread('datasetR/IM-0133-0001.jpeg', cv2.IMREAD_GRAYSCALE)
    img_size = 200
    resized_arr = cv2.resize(img_arr, (img_size, img_size))
    # data.append([resized_arr, class_num])
    resized_arr=np.array(resized_arr).reshape(-1, img_size, img_size, 1)
    print(resized_arr.shape)

    #result # 0- pneumonia
        # 1- normal
    
    pn=pickle.load(open('pneumonia.pkl','rb'))
    output=pn.predict(resized_arr)
    print("your result is here")
    print(output[0])
    #return("hello")  
    return render_template("pneumonia_result.html",result=output[0])

@app.route("/breast_cancer")
def breast_cancer():
    return render_template("breast_cancer.html")

@app.route('/breast_cancer_result', methods=['POST'])
def breast_cancer_result():
    mr = float(request.form['mean radius'])
    mp = float(request.form['mean perimeter'])
    ma = float(request.form['mean area'])
    mc = float(request.form['mean compactness'])
    mcy = float(request.form['mean concavity'])
    mcp = float(request.form['mean concave points'])
    user_input=[[mr,mp,ma,mc,mcy,mcp]]
    print(user_input)
    # model=""
    # with open('Breast_Cancer_Model','rb') as file:
    #   model = pickle.load(file)  
    model = models.load_model("Breast_Cancer_Prediction.h5")
    output=model.predict(user_input)[0]
    if output>0.7:
     output="Yes, you are having breast cancer"
    else:
        output="No, you are not having breast cancer"
    return render_template("breast_cancer_result.html",result=output)    

if __name__ == "__main__":
    app.run(debug=True)
