
#!/usr/bin/env python
from flask import Flask, render_template, Response
import cv2
import imutils
import requests
import threading


video = cv2.VideoCapture(0)
# List https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video.set(cv2.CAP_PROP_FPS, 3)
video.set(cv2.CAP_PROP_CONTRAST,11)

#video.set(cv2.CV_CAP_PROP_FPS, 10)
faceCascade = cv2.CascadeClassifier("app/haarcascade_frontalface_default.xml")


def gen():    
    while True:        
        rval, frame = video.read()        
        #frame = imutils.resize(frame, width=400)
        frame=find_faces(frame)
        cv2.imwrite('app/tmp/buff.jpg', frame)
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + open('app/tmp/buff.jpg', 'rb').read() + b'\r\n')

def envioimg(image):
    url = 'https://pydoc.com.br/facedetect/SearchFaceStudyView/?apikey=bee560126e09153f3aa9daba8b84a9e7'
    files = {'conparison': open(image, 'rb').read()}
    data={'key':'a6f3e2679ce82c866924a55585ce8c1f'}
    return requests.post(url, files=files, data=data).text


def find_faces(frame):
    inc=0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_rect = faceCascade.detectMultiScale(gray,  1.1, 5,minSize=(30, 30))    
    if len(faces_rect) == 0:
        inc =0
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
        inc =inc+1
    if inc ==2:
        path_image="app/tmp/"+str(w) + str(h) + '_faces.jpg'
        cv2.imwrite(path_image, frame)
        
        #print(envioimg(str(w) + str(h) + '_faces.jpg'))

    return frame


def create_app():
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        """Video streaming home page."""
        return render_template('index.html')
   
    @app.route('/stream')
    def video_feed():
        return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')
    
    return app


#if __name__ == '__main__':
#    app.run(host='0.0.0.0', debug=False, threaded=False)
