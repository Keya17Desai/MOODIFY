from flask import Flask, render_template, url_for, redirect,flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
import cv2
import numpy as np
import tensorflow as tf
import statistics as st


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/camera', methods = ['GET', 'POST'])
def camera():
    i=0

    GR_dict={0:(0,255,0),1:(0,0,255)}
    model = tf.keras.models.load_model('final_model.h5')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    output=[]
    cap = cv2.VideoCapture(0)
    while (i<=30):
        ret, img = cap.read()
        faces = face_cascade.detectMultiScale(img,1.05,5)

        for x,y,w,h in faces:

            face_img = img[y:y+h,x:x+w] 

            resized = cv2.resize(face_img,(224,224))
            reshaped=resized.reshape(1, 224,224,3)/255
            predictions = model.predict(reshaped)

            max_index = np.argmax(predictions[0])

            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'neutral', 'surprise')
            predicted_emotion = emotions[max_index]
            output.append(predicted_emotion)
            
            
            
            cv2.rectangle(img,(x,y),(x+w,y+h),GR_dict[1],2)
            cv2.rectangle(img,(x,y-40),(x+w,y),GR_dict[1],-1)
            cv2.putText(img, predicted_emotion, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        i = i+1

        cv2.imshow('LIVE', img)
        key = cv2.waitKey(1)
        if key == 27: 
            cap.release()
            cv2.destroyAllWindows()
            break
    print(output)
    cap.release()
    cv2.destroyAllWindows()
    final_output1 = st.mode(output)
    return render_template('buttons.html',final_output=final_output1)

@app.route('/buttons', methods = ['GET','POST'])
def buttons():
    return render_template("buttons.html")

@app.route('/songs/surprise', methods = ['GET', 'POST'])
def songsSurprise():
    return render_template("songsurprise.html")

@app.route('/songs/angry', methods = ['GET', 'POST'])
def songsAngry():
    return render_template("songangry.html")

@app.route('/songs/sad', methods = ['GET', 'POST'])
def songsSad():
    return render_template("songsad.html")

@app.route('/songs/disgust', methods = ['GET', 'POST'])
def songsDisgust():
    return render_template("songdisgust.html")

@app.route('/songs/happy', methods = ['GET', 'POST'])
def songsHappy():
    return render_template("songhappy.html")

@app.route('/songs/fear', methods = ['GET', 'POST'])
def songsFear():
    return render_template("songfear.html")

@app.route('/songs/neutral', methods = ['GET', 'POST'])
def songsNeutral():
    return render_template("songneutral.html")

@app.route('/movies/surprise', methods = ['GET', 'POST'])
def moviesSurprise():
    return render_template("moviessurprise.html")

@app.route('/movies/angry', methods = ['GET', 'POST'])
def moviesAngry():
    return render_template("moviesangry.html")

@app.route('/movies/sad', methods = ['GET', 'POST'])
def moviesSad():
    return render_template("moviessad.html")

@app.route('/movies/disgust', methods = ['GET', 'POST'])
def moviesDisgust():
    return render_template("moviesdisgust.html")

@app.route('/movies/happy', methods = ['GET', 'POST'])
def moviesHappy():
    return render_template("movieshappy.html")

@app.route('/movies/fear', methods = ['GET', 'POST'])
def moviesFear():
    return render_template("moviesfear.html")

@app.route('/movies/neutral', methods = ['GET', 'POST'])
def moviesNeutral():
    return render_template("moviesneutral.html")

if __name__ =="__main__":
    app.run(debug=True)