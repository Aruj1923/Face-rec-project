# from django.shortcuts import render

# def home(request):
#     return render(request,'index.html')
# # Create your views here.
# import face_recognition_models
import face_recognition
import numpy as np
# from django.http import HttpResponse
from django.shortcuts import render
from .models import *
# from django.core.mail import EmailMessage
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
# import cv2
# import tqdm
import os
import time
import random
import threading
from django.http import JsonResponse
import face_recognition
import pandas as pd
from django.contrib import messages
from django.http import HttpResponse
import cv2
from numba import jit, cuda 
import numpy as np


@gzip.gzip_page
 

# def input_val(request):

#     return render(request , '')


# cam = VideoCamera()
# global stream_stopped 

# stream_stopped = False

# def my_view(request):
#   global stream_stopped
#   stream_stopped = False  # Reset flag on each request

#   response = StreamingHttpResponse(gen(VideoCamera()), content_type="multipart/x-mixed-replace;boundary=frame")
#   return response

# def closeAllWindows(request):
#     cv2.destroyAllWindows()
#     return HttpResponse("AllWindows closed")

def home(request):
#     if request.method =='POST':
# #         cam
#         return JsonResponse({'message':'Video stream stopped'})
    # f_name = 'no'
    # f_name =  request.POST.get('fname')
    # names = request.POST.get('lname')
    global offname
    
    offname = True
    if request.method == 'POST':
        offname = request.POST.get('offname')
    # l_name =
    # response = request.POST.get('response') 
    # response = input('If you are unknown then say Yes or No')
    # res = ['Yes','Y','y','yes']
    # while f_name in res:
        
    #     # name = input('Enter file name with that you want to save your photo.')
    #     if names == None:
    #         break
    #     try:
    #         os.makedirs('D:/Users/asus/OneDrive/Desktop/Photos/'+names, mode=0o777, exist_ok=False)
    #     except:
    #         print('We have already have your directory')
    #     print('Be ready we take your photo in next 15 seconds ')
    #     time.sleep(15)
    #     videoCaptureObject = cv2.VideoCapture(0)
    #     result = True
    #     while(result):       
    #         ret,frame = videoCaptureObject.read()
    #         for i in range(5):
                
    #             rLL = chr(random.randint(ord('a'), ord('z')))
    #             rUL = chr(random.randint(ord('A'), ord('Z')))
    #             r = str(random.randint(1,100))
    #             if r not in os.listdir('D:/Users/asus/OneDrive/Desktop/Photos/'+names):
    #                 cv2.imwrite("D:/Users/asus/OneDrive/Desktop/Photos/"+names+'/'+rLL+r+rUL+'.jpg',frame)
    #                 break
    #         result = False
    #     videoCaptureObject.release()
    #     cv2.destroyAllWindows()
    #     # response = input('You Want another entry')
    #     # response = 'no'
    #     f_name = 'no'
        
    # input_W = input('you want to continue with laptop camera then enter l else enter O')
    # input_W = 'l'
    # global input_W 
    # global enter
    # input_W =  request.POST.get('input_W')
    # enter = request.POST.get('enter')
    



    return render(request, 'ok.html') #,context={'fname':f_name, 'lname':names , 'offname':offname}) #  ,'input_W':input_W , 'enter':enter   #, 'response':response

def video_cap(request):
    # cv2.destroyAllWindows()
    global stream_stopped
    stream_stopped = False
     
    global offname 
    offname = "True"
    if request.method == 'POST':
        offname = request.POST.get('offname')
    input_W = 'l'
    inp = ['l','L']
    
    if input_W in inp:
    # Old way only for laptop camera
        
        video_capture = cv2.VideoCapture(0)
    # else :
    #     # enter = input('Please enter your web link like http://192.168.137.81:8080/video please ensure it ends with /video')
    #     if (not enter) == True:
    #         print('As you did`nt enter we take default input and processing you laptop camera')
    #         video_capture = cv2.VideoCapture(0)
    #     else :
    #         # print('Thanks for providing link we are under process')
    #         # for i in tqdm (range (1000),  desc="Loading…",  ascii=False, ncols=75):
    #         #     time.sleep(0.01)
    #         # print("Complete.")
    #         # New way for web cam too
    #         # video_capture = cv2.VideoCapture('http://192.168.137.81:8080/video')
    # 
    #         video_capture = cv2.VideoCapture(enter)
    
    try:
        # if request.method == 'POST':
        #     stream_stopped.set()  # Set the event on POST request
        #     return HttpResponse("Stream stopped.")
        # else:
            cam = VideoCamera()
            return  StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame") #,context ={'offname':offname} )
        # return render(request, 'app1.html' , context={'reqt':reqt})
    except:
        # cv2.destroyAllWindows()
        # video_cap()
        # VideoCamera.release()
        pass
        

        # cam = VideoCamera()
        # return  StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    return render(request, 'ok.html')





def contact(request):

    
    return render(request , "Contact.html")


# def front(request):

#     return render(request , 'front.html')



def form_page(request):
    f_name = 'no'
    f_name =  request.POST.get('fname')
    names = request.POST.get('lname')
    # global offname
    
    # offname = True
    # if request.method == 'POST':
    #     offname = request.POST.get('offname')
    # l_name =
    # response = request.POST.get('response') 
    # response = input('If you are unknown then say Yes or No')
    res = ['Yes','Y','y','yes']
    while f_name in res:
        
        # name = input('Enter file name with that you want to save your photo.')
        if names == None:
            break
        try:
            os.makedirs('D:/Users/asus/OneDrive/Desktop/Photos/'+names, mode=0o777, exist_ok=False)
        except:
            print('We have already have your directory')
        print('Be ready we take your photo in next 15 seconds ')
        time.sleep(15)
        videoCaptureObject = cv2.VideoCapture(0)
        result = True
        while(result):       
            ret,frame = videoCaptureObject.read()
            for i in range(5):
                
                rLL = chr(random.randint(ord('a'), ord('z')))
                rUL = chr(random.randint(ord('A'), ord('Z')))
                r = str(random.randint(1,100))
                if r not in os.listdir('D:/Users/asus/OneDrive/Desktop/Photos/'+names):
                    cv2.imwrite("D:/Users/asus/OneDrive/Desktop/Photos/"+names+'/'+rLL+r+rUL+'.jpg',frame)
                    break
            result = False
        videoCaptureObject.release()
        cv2.destroyAllWindows()
        # response = input('You Want another entry')
        f_name = 'no'
   
    return render(request,"newform.html")


# global l
# l = []

# def attandence(request, face_names):
    
#     l.append(face_names)

#     if len(l) != 0:
#         l_p = pd.Series(l)
#         l_p = l_p.value_counts().sort_values(ascending = False)
        
#         list(dict(l_p).keys())[0]

#         return render(request)
#     else:
#         return render(request)
    



#to capture video class
class VideoCamera(object):
    
    def __init__(self):
        
        
        folder_dir = "D:/Users/asus/OneDrive/Desktop/Photos"
        global my_data
        global known_face_encodings 
        global known_face_names 
        my_data={}
        known_face_encodings = []
        known_face_names = []
        import face_recognition
        for images,i in zip(os.listdir(folder_dir),range(1000)):
            for j in os.listdir(folder_dir+'/'+images):
         
        
 
           #  check if the image ends with png
                if (j.endswith(".jpg")):
            
                    my_data[f'images{i}' ] = face_recognition.load_image_file(folder_dir+'/'+images+'/'+j)
                    my_data[f'images{i}_encoding'] = face_recognition.face_encodings(my_data[f'images{i}'] ,model='cnn')[0] 
                    known_face_encodings.append(my_data[f'images{i}_encoding'])
                    my_data[f'name{i}'] = images.replace('.jpg','')
                    known_face_names.append(my_data[f'name{i}'])
                elif (j.endswith(".jpeg")):
                    my_data[f'images{i}'] = face_recognition.load_image_file(folder_dir+'/'+images+'/'+j)
                    my_data[f'images{i}_encoding'] = face_recognition.face_encodings(my_data[f'images{i}'],model='cnn')[0] 
                    known_face_encodings.append(my_data[f'images{i}_encoding'])
                    my_data[f'name{i}'] = images.replace('.jpeg','')
                    known_face_names.append(my_data[f'name{i}'])




        # for images,i in zip(os.listdir(folder_dir),range(1000)):
 
        #     # check if the image ends with png
        #     if (images.endswith(".jpg")):
        #         my_data[f'images{i}'] = face_recognition.load_image_file('D:/Users/asus/OneDrive/Desktop/Photos/'+images)
        #         my_data[f'images{i}_encoding'] = face_recognition.face_encodings(my_data[f'images{i}'])[0]
        #         known_face_encodings.append(my_data[f'images{i}_encoding'])
        #         my_data[f'name{i}'] = images.replace('.jpg','')
        #         known_face_names.append(my_data[f'name{i}'])
        #     elif (images.endswith(".jpeg")):
        #         my_data[f'images{i}'] = face_recognition.load_image_file('D:/Users/asus/OneDrive/Desktop/Photos/'+images)
        #         my_data[f'images{i}_encoding'] = face_recognition.face_encodings(my_data[f'images{i}'])[0]
        #         known_face_encodings.append(my_data[f'images{i}_encoding'])
        #         my_data[f'name{i}'] = images.replace('.jpeg','')
        #         known_face_names.append(my_data[f'name{i}'])






        # # Load a sample picture and learn how to recognize it.
        # obama_image = face_recognition.load_image_file("C:/Users/shreya computer/Pictures/Camera Roll/WIN_20240303_11_35_12_Pro.jpg")
        # obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

        # # Load a second sample picture and learn how to recognize it.
        # biden_image = face_recognition.load_image_file("E:/SONAL DOCUMENT/Untitled.jpg")
        # biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

        # # Create arrays of known face encodings and their names
        # known_face_encodings = [
        #     obama_face_encoding,
        #     biden_face_encoding
        # ]
        # known_face_names = [
        #     "Aruj",
        #     "Sonal"
        # ]

        # Initialize some variables
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        th1 = threading.Thread(target=self.update, args=())
        # th2 = threading.Thread(target=self.update, args=())

        th1.start()
        # th2.start()

        # th1.join()
        # th2.join()

    def __del__(self):
        self.video.release()
        # cv2.destroyAllWindows()

    def get_frame(self):
        image = self.frame
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
    


    # @jit(target_backend='cuda') 
    def update(self):
        global face_names
        global face_locations
        global face_encodings
        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True
        
        while True:
            (self.grabbed, self.frame) = self.video.read()


            def process_frame(process_this_frame):
        

                global face_names
                global face_locations
                # Only process every other frame of video to save time
                if process_this_frame:
                    # Resize frame of video to 1/4 size for faster face recognition processing
                    small_frame = cv2.resize(self.frame, (0, 0), fx=0.25, fy=0.25)

                    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                    # rgb_frame = frame[:, :, ::-1]
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    # Find all the faces and face encodings in the current frame of video
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                    face_names = []
                    for face_encoding in face_encodings:
                        # See if the face is a match for the known face(s)
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        name = "Unknown"

                        # # If a match was found in known_face_encodings, just use the first one.
                        # if True in matches:
                        #     first_match_index = matches.index(True)
                        #     name = known_face_names[first_match_index]

                        # Or instead, use the known face with the smallest distance to the new face


                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        # if matches[best_match_index]:
                        #     name = known_face_names[best_match_index]
                        

                        if face_distances[best_match_index] < 0.5:  # Example threshold for 70% accuracy
                         # If the distance is less than the threshold, it's a potential match
                             if matches[best_match_index]:
                                 name =  known_face_names[best_match_index]
                             else:
                                 name = "Unknown (Low Confidence)"
                        else:
                             name = "Unknown"
                        
                        face_names.append(name)
                    # print(face_names)

                process_this_frame = not process_this_frame
            # try:
            t1 = threading.Thread(target=process_frame , args = (process_this_frame,))
            t1.start()
            # except:
            #     break
            
            # Display the results



            # @jit( nopython=True ,  target_backend='cuda')
            def display(frame,face_locations , face_names):
             
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
            
                    # print('Display the results')

                    #  Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                    # Draw a label with a name below the face
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            # try: 

            
            p1 = threading.Thread(target=display , args = (self.frame,face_locations,face_names))
            p2 = threading.Thread(target=display , args = (self.frame,face_locations,face_names))
            p3 = threading.Thread(target=display , args = (self.frame,face_locations,face_names))
            p4 = threading.Thread(target=display , args = (self.frame,face_locations,face_names))
            p5 = threading.Thread(target=display , args = (self.frame,face_locations,face_names))

            p1.start()
            p2.start()
            p3.start()
            p4.start()
            p5.start()

            p1.join()
            p2.join()
            p3.join()
            p4.join()
            p5.join()
            
            if offname == "False":
                # self.offname = "True"
                # p1.stop()
                # p2.stop()
                # p3.stop()
                # p4.stop()
                # p5.stop()
                break
            else:
                pass 
        VideoCamera.__del__()
        cv2.destroyAllWindows()

                # Display the resulting image
                # resised = cv2.resize( self.frame ,(900,500))
                # cv2.imshow('Video', resised)

                # Hit 'q' on the keyboard to quit!
        #     if cv2.waitKey(0) | 0xFF == ord('q'):
        #         break
        # cv2.destroyAllWindows()

                # Release handle to the webcam
            # video_capture.release()s
            # cv2.destroyAllWindows() 

    def stop(self):
        self.video.release()
        # cv2.destroyAllWindows()


# cam = VideoCamera().stop()

# def stop_vedio(request):
#     if request.method =='POST':
#         cam
#         return JsonResponse({'message':'Video stream stopped'})
#     else:
#         cv2.destroyAllWindows()
#         return JsonResponse({'message': 'Video stopped successfully'})

stream_stopped = threading.Event()

def gen(camera):
    # offname= False
    while not stream_stopped:
        # offname= False
        frame = camera.get_frame()
        # if offname == "False":
        # # Handle potential errors (e.g., camera disconnected)
        #    break
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        # if offname == False:
            
        #     # offname = True
        #     break
        





face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5, minSize=(20, 20))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces


'''

def face_rec():
    
     
    import face_recognition
    import cv2
    import numpy as np
    import os
    from os import listdir
    import pandas as pd
    import time
    from tqdm import tqdm
    import random
    import string

    # This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
    # other example, but it includes some basic performance tweaks to make things run a lot faster:
    #   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
    #   2. Only detect faces in every other frame of video.

    # PLEASE NOTE : This  example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
    # OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
    # specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

    # Get a reference to webcam #0 (the default one)


    # For Take new data into database

    response = input('If you are unknown then say Yes or No')
    res = ['Yes','Y','y','yes']
    while response in res:
        
        name = input('Enter file name with that you want to save your photo.')
        try:
            os.makedirs('D:/Users/asus/OneDrive/Desktop/Photos/'+name, mode=0o777, exist_ok=False)
        except:
            print('We have already have your directory')
        print('Be ready we take your photo in next 15 seconds ')
        time.sleep(15)
        videoCaptureObject = cv2.VideoCapture(0)
        result = True
        while(result):       
            ret,frame = videoCaptureObject.read()
            for i in range(15):
                
                rLL = chr(random.randint(ord('a'), ord('z')))
                rUL = chr(random.randint(ord('A'), ord('Z')))
                r = str(random.randint(1,100))
                if r not in os.listdir('D:/Users/asus/OneDrive/Desktop/Photos/'+name):
                    cv2.imwrite("D:/Users/asus/OneDrive/Desktop/Photos/"+name+'/'+rLL+r+rUL+'.jpg',frame)
                    break
            result = False
        videoCaptureObject.release()
        cv2.destroyAllWindows()
        response = input('You Want another entry')
    input_W = input('you want to continue with laptop camera then enter l else enter O')
    inp = ['l','L']

    if input_W in inp:
    # Old way only for laptop camera
        video_capture = cv2.VideoCapture(0)
    else :
        enter = input('Please enter your web link like http://192.168.137.81:8080/video please ensure it ends with /video')
        if (not enter) == True:
            print('As you did`nt enter we take default input and processing you laptop camera')
            video_capture = cv2.VideoCapture(0)
        else :
            print('Thanks for providing link we are under process')
            for i in tqdm (range (1000),  desc="Loading…",  ascii=False, ncols=75):
                time.sleep(0.01)
            print("Complete.")
            # New way for web cam too
            # video_capture = cv2.VideoCapture('http://192.168.137.81:8080/video')
            video_capture = cv2.VideoCapture(enter)

    # Auto pick photos from directory name and path for image encoding.

    # get the path/directory
    folder_dir = "D:/Users/asus/OneDrive/Desktop/Photos"
    my_data={}
    known_face_encodings = []
    known_face_names = []

    for images,i in zip(os.listdir(folder_dir),range(1000)):
        for j in os.listdir(folder_dir+'/'+images):
       
        
 
       #  check if the image ends with png
            if (j.endswith(".jpg")):
            
                my_data[f'images{i}' ] = face_recognition.load_image_file(folder_dir+'/'+images+'/'+j)
                my_data[f'images{i}_encoding'] = face_recognition.face_encodings(my_data[f'images{i}'])[0] 
                known_face_encodings.append(my_data[f'images{i}_encoding'])
                my_data[f'name{i}'] = images.replace('.jpg','')
                known_face_names.append(my_data[f'name{i}'])
            elif (j.endswith(".jpeg")):
                my_data[f'images{i}'] = face_recognition.load_image_file(folder_dir+'/'+images+'/'+j)
                my_data[f'images{i}_encoding'] = face_recognition.face_encodings(my_data[f'images{i}'])[0] 
                known_face_encodings.append(my_data[f'images{i}_encoding'])
                my_data[f'name{i}'] = images.replace('.jpeg','')
                known_face_names.append(my_data[f'name{i}'])




    # for images,i in zip(os.listdir(folder_dir),range(1000)):
 
    #     # check if the image ends with png
    #     if (images.endswith(".jpg")):
    #         my_data[f'images{i}'] = face_recognition.load_image_file('D:/Users/asus/OneDrive/Desktop/Photos/'+images)
    #         my_data[f'images{i}_encoding'] = face_recognition.face_encodings(my_data[f'images{i}'])[0]
    #         known_face_encodings.append(my_data[f'images{i}_encoding'])
    #         my_data[f'name{i}'] = images.replace('.jpg','')
    #         known_face_names.append(my_data[f'name{i}'])
    #     elif (images.endswith(".jpeg")):
    #         my_data[f'images{i}'] = face_recognition.load_image_file('D:/Users/asus/OneDrive/Desktop/Photos/'+images)
    #         my_data[f'images{i}_encoding'] = face_recognition.face_encodings(my_data[f'images{i}'])[0]
    #         known_face_encodings.append(my_data[f'images{i}_encoding'])
    #         my_data[f'name{i}'] = images.replace('.jpeg','')
    #         known_face_names.append(my_data[f'name{i}'])






    # # Load a sample picture and learn how to recognize it.
    # obama_image = face_recognition.load_image_file("C:/Users/shreya computer/Pictures/Camera Roll/WIN_20240303_11_35_12_Pro.jpg")
    # obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

    # # Load a second sample picture and learn how to recognize it.
    # biden_image = face_recognition.load_image_file("E:/SONAL DOCUMENT/Untitled.jpg")
    # biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

    # # Create arrays of known face encodings and their names
    # known_face_encodings = [
    #     obama_face_encoding,
    #     biden_face_encoding
    # ]
    # known_face_names = [
    #     "Aruj",
    #     "Sonal"
    # ]

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Only process every other frame of video to save time
        if process_this_frame:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            # rgb_frame = frame[:, :, ::-1]
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        resised = cv2.resize( frame ,(900,500))
        cv2.imshow('Video', resised)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

    
    '''