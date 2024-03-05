from flask import Flask, Response, render_template, send_file # pip install flask 
import cv2, base64 # pip install opencv-python 
from ultralytics import YOLO # pip install ultralytics
from io import BytesIO

app = Flask(__name__) # Make instance of Flask 
captureCamera = cv2.VideoCapture(0)
model = YOLO('best.pt')
frames = []



def getRes(frame):
    result = model(frame, show_boxes=True)[0]
    return result

def getImage(video): 
    frame = cv2.imencode('.jpg', video)[1].tobytes()
    toImg =  BytesIO(frame)
    img = base64.b64encode(toImg.read()).decode('utf-8')
    return img

@app.route('/')
def index():
    return render_template('pages/index.html')

def accessCamera(videocam): 


    while True:

        accessible, frame = captureCamera.read()
        
        if not accessible:
            break

        else:

            result = model(frame, show_boxes=True)[0]
            cv2.rectangle(frame, (150, 50), (460, 370), (42, 219, 151), 1)

        for r in result:
            boundary_box = [(r.boxes.xyxy[0][i]).item() for i in range(4)]   
            confidence = (r.boxes.conf[0]).item()
            lbls = (r.boxes.cls).item()

            x1, y1, x2, y2 = map(int, boundary_box)
            frames.append(frame)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 144, 30), 1)
        
        frame = cv2.imencode('.jpg', frame)[1].tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break
            

@app.route('/home')
def showCamera():
    global captureCamera
    return Response(accessCamera(captureCamera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/results')
def page():
    global captureCamera
    result = getRes(captureCamera.read()[1])
    img = getImage(captureCamera.read()[1])
    return render_template('pages/page.html', result=result, img=img)

    
if __name__ == "__main__":
    app.run(debug=True)





