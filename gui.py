from tkinter import *
from PIL import ImageTk, Image
from itertools import count
import cv2
from PIL import Image
import torch
import threading
# import screeninfo
import base64
import requests
import time
import json                    
import unidecode
import argparse
from gtts import gTTS
import playsound
from datetime import datetime
import os
import numpy as np

from caffe.ultra_face_opencvdnn_inference import path, inference, net as net_dnn


# ********************************** Face recognition variables **********************************
parser = argparse.ArgumentParser(description='Face Recognition')
parser.add_argument('-db', '--debug', default='False',
        type=str, metavar='N', help='Turn on debug mode')

args = parser.parse_args()
debug = False
if args.debug == 'True':
	debug = True

url = 'http://192.168.1.62:5051/'
# url = 'http://localhost:5051/'

api_list = [url + 'FaceRec', url + 'FaceRec_DREAM', url + 'FaceRec_3DFaceModeling']
request_times = [10, 10, 10]
api_index = 0

secret_key = "6fdf703e-1f6c-4196-9e85-a20133eb6337"
# secret_key = "057ca076-de61-44ec-a619-195758453416"

window_name = 'Hệ thống phần mềm AI nhận diện khuôn mặt VKIST'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

temp_boxes = []
predict_labels = []
queue = []

crop_image_size = 120
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 0.8
fontcolor = (0,255,0)

extend_pixel = 50
minimum_face_size = 60

box_size = 250
n_box = 1

webcam = cv2.VideoCapture(0)

print('webcam.get(cv2.CAP_PROP_FRAME_WIDTH): ' + str(webcam.get(cv2.CAP_PROP_FRAME_WIDTH)))
print('webcam.get(cv2.CAP_PROP_FRAME_HEIGHT): ' + str(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# print('Window size: w = ', window_size[0], ' h = ', window_size[1])
frame_width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
x_dis_image = int((frame_width - box_size * n_box) / (n_box + 1))
y_dis_image = int((frame_height - box_size) / 2)


temp_id = -2
temp_name = 'Unknown'
time_appear = time.time()
max_time_appear = 5
prev_frame_time = 0
new_frame_time = 0

cur_time = 0
max_times = 3

take_photo_state = False


if not os.path.exists(path + 'sounds/'):
    os.makedirs(path + 'sounds/')


def remove_accent(text):
    return unidecode.unidecode(text)

def set_temp_value(new_id, new_name, is_reset):
	global temp_id, temp_name
	temp_id = new_id
	temp_name = new_name
	if is_reset and debug:
		print("--------------- Reset temp value")
	else:
		if debug:
			print("+++++++++++++++ Update temp value")

def check_first_time_appear(cur_id, cur_name, temp_id_):
	if cur_id != -1:
		if cur_id != temp_id_:
			if debug:
				print('cur_id: ' + str(cur_id) + ' temp_id: ' + str(temp_id_))
			set_temp_value(cur_id, cur_name, False)
			return True
		else:
			return False

def say_hello(content):
    unsign_content = remove_accent(content).replace(" ", "_")
    if not os.path.isfile(unsign_content + ".mp3"):
        if debug:
            print("Creating " + unsign_content + ".mp3 file")
        tts = gTTS(content, tld = 'com.vn', lang='vi')
        tts.save(path + 'sounds/' + unsign_content + ".mp3")
    
    playsound.playsound(path + 'sounds/' + unsign_content + ".mp3", True)
def face_recognize(frame):
    cur_hour = str(datetime.now()).split(" ")[1].split(":")[0]

    global predict_labels, time_appear, max_time_appear, temp_id, temp_name, cur_time, api_index, max_times

    _, encimg = cv2.imencode(".jpg", frame)
    img_byte = encimg.tobytes()
    img_str = base64.b64encode(img_byte).decode('utf-8')
    new_img_str = "data:image/jpeg;base64," + img_str
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain', 'charset': 'utf-8'}

    payload = json.dumps({"secret_key": secret_key, "img": new_img_str})

    seconds = time.time()
    response = requests.post(api_list[api_index], data=payload, headers=headers, timeout=100)

    try:
        print('Server response', response.json())
        for id, bb, name, profileID, generated_face_id in zip(response.json()['result']['id'], response.json()['result']['bboxes'], response.json()['result']['identities'], response.json()['result']['profilefaceIDs'], response.json()['result']['3DFace'] ):
            response_time_s = time.time() - seconds
            print("Server's response time: " + "%.2f" % (response_time_s) + " (s)")
            bb = bb.split(' ')
            if check_first_time_appear(id, name, temp_id):
                time_appear = time.time()
                max_time_appear = 10

                non_accent_name = remove_accent(temp_name)
                if id > -1:
                    front_string = "Xin chào "
                    if int(cur_hour) > 15:
                        front_string = "Tạm biệt "

                name_parts = temp_name.split(' - ')[0].split(' ')
                content = ''
                if non_accent_name.find(' Thi ') > -1 and len(name_parts) < 4:
                    print('\n' + front_string + name_parts[-1] + ' ' + name_parts[0] + '\n')  
                    content = front_string + name_parts[-1] + ' ' + name_parts[0]
                    # say_hello(front_string + name_parts[-1] + ' ' + name_parts[0])
                else:
                    print('\n' + front_string + name_parts[-2] + ' ' + name_parts[-1] + '\n') 
                    content = front_string + name_parts[-2] + ' ' + name_parts[-1]
                    # say_hello(front_string + name_parts[-2] + ' ' + name_parts[-1])

                faceI = cv2.resize(frame[int(float(bb[1])): int(float(bb[3])), int(float(bb[0])): int(float(bb[2]))], (crop_image_size, crop_image_size))
                predict_labels.append([non_accent_name, faceI, content, profileID, generated_face_id])
            else:
                cur_time += 1
                if cur_time >= max_times:
                    temp_id = -2
                    temp_name = 'Unknown'

    except requests.exceptions.RequestException:
        print(response.text)

    return
# ********************************** Tkinder variables **********************************
x_dis = 10
y_dis = 10

window_size_x = 1200
window_size_y = 700

button_size_x = 120
button_size_y = 20

distance_x = 20
distance_y = 30

image_zone_size_x = window_size_x
image_zone_size_y = 600

button_zone_size_x = window_size_x
button_zone_size_y = 90

def mode_1():
    global api_index
    print('\nACTIVATE MODE 1\n')
    take_photo_btn["state"] = DISABLED
    api_index = 0

def mode_2():
    global api_index
    print('\nACTIVATE MODE 2\n')
    take_photo_btn["state"] = DISABLED
    api_index = 1

def take_photo():
    global take_photo_state
    print('\nSend image to 3D face modeling server\n')
    take_photo_state = True

def mode_3():
    global api_index, take_photo_state
    print('\nACTIVATE MODE 3\n')
    take_photo_btn["state"] = NORMAL
    api_index = 2
    take_photo_state = False

root = Tk()
root.title(window_name)
root.geometry(str(window_size_x) + 'x' + str(window_size_y))

# # Create image zone
# image_zone = Frame(root, width = image_zone_size_x, height = image_zone_size_y, bg="white")
# image_zone.grid()
# # Create a label in the image zone
# lmain = Label(image_zone)
# lmain.grid()

# Create image zone
image_zone = Canvas(root, width = image_zone_size_x, height = image_zone_size_y, bg="white")
image_zone.grid()

image_id = None  # default value at start (to create global variable)

# Create button zone
button_zone = Frame(root, width = button_zone_size_x, height = button_zone_size_y, bg='#bbbcbd')
button_zone.place(x= 0, y = image_zone_size_y + y_dis)

var = IntVar()
# Create a Button
mode_1_btn = Radiobutton(button_zone, text = 'Chế độ 1', variable=var, value=1, bd = '5', command = mode_1)
mode_1_btn.place(x = x_dis, y = y_dis * 2)
mode_1_btn.select()

mode_2_btn = Radiobutton(button_zone, text = 'Chế độ 2', variable=var, value=2, bd = '5', command = mode_2)
mode_2_btn.place(x = button_size_x + x_dis, y = y_dis * 2)

mode_3_btn = Radiobutton(button_zone, text = 'Chế độ 3', variable=var, value=3,  bd = '5', command = mode_3)
mode_3_btn.place(x = button_size_x * 2 + x_dis, y = y_dis * 2)

take_photo_btn = Button(button_zone, text = 'Chụp ảnh!', bd = '5', command = take_photo)
take_photo_btn.place(x = button_size_x * 3 + x_dis, y = y_dis * 2)

take_photo_btn["state"] = DISABLED

# Capture from camera
cap = cv2.VideoCapture(0)

count = 0
# tm = cv2.TickMeter()

# function for video streaming
def video_stream():
    global count, predict_labels, temp_boxes, prev_frame_time, new_frame_time, queue, api_index, request_times, take_photo_state
    global image_id
    # tm.start()
    count += 1

    frame_show = np.ones((window_size_y, window_size_x, 3),dtype='uint8') * 255    

    ret, orig_image = webcam.read()


    final_frame = orig_image.copy()

    # for i in range(0, n_box):
    #     final_frame = cv2.rectangle(final_frame,(int((x_dis + box_size) * i) + x_dis, y_dis), (int((x_dis + box_size) * i) + x_dis + box_size, y_dis + box_size),(255,0,0), 10)

    # temp_boxes, _, probs = predictor.predict(orig_image[y_dis: y_dis + box_size, x_dis: x_dis + box_size], candidate_size / 2, threshold)
    temp_boxes, _, probs = inference(net_dnn, orig_image)

    for i, boxI in enumerate(temp_boxes):
        x1, y1, x2, y2 = int(boxI[0]), int(boxI[1]), int(boxI[2]), int(boxI[3])
        # if ((x2 - x1) * (y2 - y1)) / (box_size * box_size) > 0.2:
        final_frame = cv2.rectangle(final_frame,(x1, y1), (x2, y2),(0,255,0), 2)

    if api_index < 2 or (api_index == 2 and take_photo_state):
        if (count % request_times[api_index]) == 0:
            for i, boxI in enumerate(temp_boxes):
                x1, y1, x2, y2 = int(boxI[0]), int(boxI[1]), int(boxI[2]), int(boxI[3])
                queue = [t for t in queue if t.is_alive()]
                if len(queue) < 3:
                    # queue.append(threading.Thread(target=face_recognize, args=(orig_image,)))
                    queue.append(threading.Thread(target=face_recognize, args=(orig_image[y1:y2, x1:x2],)))
                    queue[-1].start()
                count = 0
            take_photo_state = False

    frame_show[:frame_height, :frame_width,:] = final_frame

    # image_names = ['Capture Image(s)', 'Profile Image(s)', '3D Image(s)']


    image_names = ['Ảnh tức thời', 'Ảnh hồ sơ', 'Ảnh 3D']
    for i, name_I in enumerate(image_names):
        # cv2.putText(frame_show, '{0}'.format(name_I), (frame_width + distance_x + (crop_image_size + distance_x) * i, int(distance_y * 0.5)), fontface, 0.4, (0, 0, 0))
        image_zone.create_text(frame_width + distance_x + (crop_image_size + distance_x) * i + 60, int(distance_y * 0.5), text=name_I, fill="black", font=('Helvetica 15 bold'))

    image_name_y = 30

    for i, labelI in enumerate(predict_labels):
        if frame_width + distance_x + crop_image_size < window_size_x and int((crop_image_size + distance_y) * i) + distance_y + crop_image_size < window_size_y:
            cv2.putText(frame_show, '{0}'.format(labelI[0]), (frame_width + distance_x, int((crop_image_size + distance_y) * i) + int(distance_y / 1.5)  + image_name_y), fontface, fontscale, (100, 255, 0))
            frame_show[int((crop_image_size + distance_y) * i) + distance_y + image_name_y: int((crop_image_size + distance_y) * i) + distance_y + image_name_y + crop_image_size, frame_width + distance_x: frame_width + distance_x + crop_image_size, :] = labelI[1]

            if labelI[3] != '':
                cur_url = url + 'images/' + secret_key + '/' + labelI[3]
                cur_profile_face = np.array(Image.open(requests.get(cur_url, stream=True).raw))
                cur_profile_face = cv2.resize(cur_profile_face, (crop_image_size, crop_image_size))
                cur_profile_face = cv2.cvtColor(cur_profile_face, cv2.COLOR_BGR2RGB)
                frame_show[int((crop_image_size + distance_y) * i) + distance_y + image_name_y: int((crop_image_size + distance_y) * i) + distance_y + image_name_y + crop_image_size, frame_width + distance_x * 2 + crop_image_size: frame_width + distance_x * 2 + crop_image_size * 2, :] = cur_profile_face
            
            if labelI[4] != '':
                cur_url = url + 'images/' + secret_key + '/' + labelI[4]
                cur_generated_face = np.array(Image.open(requests.get(cur_url, stream=True).raw))
                cur_generated_face = cv2.resize(cur_generated_face, (crop_image_size, crop_image_size))
                cur_generated_face = cv2.cvtColor(cur_generated_face, cv2.COLOR_BGR2RGB)
                frame_show[int((crop_image_size + distance_y) * i) + distance_y + image_name_y: int((crop_image_size + distance_y) * i) + distance_y + image_name_y + crop_image_size, frame_width + distance_x * 3 + crop_image_size * 2: frame_width + distance_x * 3 + crop_image_size * 3, :] = cur_generated_face


    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = str(int(fps))

    cv2.putText(frame_show, '{0} fps'.format(fps), (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)
    
    # text = Text(root)

    # text.insert(INSERT, "Xin chào")
    # text.insert(END, "...")
    # text.place(x = frame_width + distance_x + (crop_image_size + distance_x) * 0, y = int(distance_y * 0.5))


    cv2image = cv2.cvtColor(frame_show, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)

    # convert to Tkinter image
    imgtk = ImageTk.PhotoImage(image=img)
    # image_zone.create_image(0, 0, anchor=NW, image=imgtk)
    
    # solution for bug in `PhotoImage`
    image_zone.photo = imgtk

    # check if image already exists
    if image_id:       
        # replace image in PhotoImage on canvas
        image_zone.itemconfig(image_id, image=imgtk)
    else:
        # create first image on canvas and keep its ID
        image_id = image_zone.create_image((0,0), image=imgtk, anchor='nw')

        # resize canvas
        image_zone.configure(width=imgtk.width(), height=imgtk.height())

    root.after(1, video_stream) 

    if len(predict_labels) > 3:
        temp_boxes = []
        predict_labels = []

video_stream()
root.mainloop()
cap.release()