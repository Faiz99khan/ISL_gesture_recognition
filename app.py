from flask import Flask, render_template, Response, redirect, url_for, request
#from flask_ngrok import run_with_ngrok
import os
import time
import cv2

import torch
from torch.autograd import Variable
import cv2
from PIL import Image
from spatial_transforms_new import *
from utils import Queue
import time
import torch.nn.functional as F

from load_model import Opt, load_models

from IPython.display import display,Audio
from playsound import playsound


import ffmpeg
def check_rotation(path_video_file):
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)

    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotateCode = None
  #  print(meta_dict['streams'][0]['tags'].keys())
    if 'rotate' in meta_dict['streams'][0]['tags'].keys():      
        if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
            rotateCode = cv2.ROTATE_90_CLOCKWISE
        elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
            rotateCode = cv2.ROTATE_180
        elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
            rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE

    return rotateCode

opt=Opt(path_det='./detector.pth',path_clf='./classifier.pth')
print('loading models...')
detector,classifier=load_models(opt)
print('models loaded successfully!')

audio_files_path='./audio_files/mp3/'

key_to_audio={}
key_to_file={}

def load_audio_files(path):
    global key_to_audio,key_to_file
    import os
    str_to_num={'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10}
    if not os.path.isdir(path):
       print('audio_files_directory does not exist')
       return
    files_lst=os.listdir(path)

    #key_to_audio={}
    for el in files_lst:
      ind=str_to_num[el[:-4]]
      key_to_audio[ind]=Audio(path+el,autoplay=True)
      key_to_file[ind]=path+el

load_audio_files(audio_files_path)

class Prediction():
  def __init__(self,rotate_code=None):
    self.root_path='.'

    self.no_cuda=True
    self.top2_diff_threshold=.75
    self.freq_threshold=3
    self._6_freq_threshold=3#6
    self.pred_cls_queue_size=12

    self.n_classes_clf=11
    self.n_classes_det=2
    self.det_queue_size=4
    self.clf_queue_size=4
    self.sample_duration=16
    self.sample_duration_det=8
    self.det_strategy='ma'
    self.clf_strategy='ma'


    self.mean=[114.7748, 107.7354, 99.475]

   # self.interpolation=Image.BICUBIC
    self.interpolation=Image.BILINEAR



    self.pred_cls_queue=(np.ones(self.pred_cls_queue_size)*10).tolist()

    self.clf_selected_queue = np.zeros(self.n_classes_clf, )
    self.det_selected_queue = np.zeros(self.n_classes_det, )
    self.myqueue_det = Queue(self.det_queue_size, n_classes=self.n_classes_det)
    self.myqueue_clf = Queue(self.clf_queue_size, n_classes=self.n_classes_clf)

    self.num_frame = 0
    self.clip = []
    self.org_frms=[]

    self.detector=detector
    self.classifier=classifier       
    if not self.no_cuda:
      self.detector,self.classifier=self.detector.cuda(),self.classifier.cuda()

    self.detector.eval()
    self.classifier.eval()
        
    self.infer_extract_roi=InferenceExtractRoi(self.no_cuda)

    self.spatial_transform = Compose([
        FinalSize(interpolation=self.interpolation),
        ToTensor(1), Normalize(self.mean, [1, 1, 1])
    ])

    self.spatial_transform.randomize_parameters()

    self.rotate_code=rotate_code

    self.key_to_audio=key_to_audio
    self.key_to_file=key_to_file
    
    self.final_prediction=10
    self.prev_prediction=10
    self.count_0=0
    
    self.tmp_inputs2=0

    self.prev_best1=10
    self.top2_diff_avg=[]
    self.pred_cls=[]
    self.outputs_clf=0             #### temp
    self.tmp_count=0

  def  __call__(self,frame):
      self.tmp_count+=1
      t1 = time.time()
      if self.rotate_code is not None:
        frame=cv2.rotate(frame,self.rotate_code)
      if self.num_frame == 0:
          cur_frame=frame#cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
          for i in range(16):
              self.org_frms.append(cur_frame)
      self.org_frms.pop(0)
      _frame=frame#cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
      self.org_frms.append(_frame)
      x1,y1,x2,y2=self.infer_extract_roi(self.org_frms[0])
    #  st=time.time()
      self.clip = [self.spatial_transform(img[y1:y2,x1:x2].copy()) for img in self.org_frms]
      # print('time taken by transformation',time.time()-st)

      im_dim = self.clip[0].size()[-2:]
      try:
          test_data = torch.cat(self.clip, 0).view((self.sample_duration, -1) + im_dim).permute(1, 0, 2, 3)
      except Exception as e:
          pdb.set_trace()
          raise e
      inputs = torch.cat([test_data],0).view(1,3,self.sample_duration,112,112)
      self.num_frame += 1



      with torch.no_grad():
          
          inputs = Variable(inputs)
          inputs_det = inputs[:, [2,1,0], -self.sample_duration_det:, :, :]  ###
          inputs_det=inputs_det.cpu()
          outputs_det = self.detector(inputs_det)
          outputs_det = F.softmax(outputs_det, dim=1)
          outputs_det = outputs_det.cpu().numpy()[0].reshape(-1, )
          # enqueue the probabilities to the detector queue
          self.myqueue_det.enqueue(outputs_det.tolist())

          self.myqueue_det.dequeue()        ####

          if self.det_strategy == 'raw':
              self.det_selected_queue = outputs_det
          elif self.det_strategy == 'median':
              self.det_selected_queue = self.myqueue_det.median
          elif self.det_strategy == 'ma':
              self.det_selected_queue = self.myqueue_det.ma
          elif self.det_strategy == 'ewma':
              self.det_selected_queue = self.myqueue_det.ewma
          prediction_det = np.argmax(self.det_selected_queue)

          prob_det = self.det_selected_queue[prediction_det]

          #print(self.tmp_count,'\n',outputs_det,'\t',outputs_det.argmax())
         # print(self.tmp_count,'\n',self.det_selected_queue,'\t',prediction_det)

          if prediction_det==0:
             self.count_0+=1
          else:
             self.count_0=0

          if self.count_0>=50:
             prediction_det=0
          else:
             prediction_det=1
          
         # print('final_prediction_det:',prediction_det)
          

          #### State of the detector is checked here as detector act as a switch for the classifier
          if prediction_det==1:                                                                      ####
              inputs_clf = inputs[:, :, :, :, :]
              inputs_clf = torch.Tensor(inputs_clf.numpy()[:,:,::1,:,:])
            #  st=time.time()
              outputs_clf = self.classifier(inputs_clf)
            #  print('actual time by clf',time.time()-st)
              #self.outputs_clf=outputs_clf

              outputs_clf = F.softmax(outputs_clf.data, dim=1)
            #   s=time.time()
              outputs_clf=outputs_clf.detach().cpu()
          #    print('on cpu',time.time()-s)
              outputs_clf = outputs_clf.numpy()[0].reshape(-1, )
            #   print('cpu conversion',time.time()-s)

              # Push the probabilities to queue
              #print(outputs_clf.shape,outpuu)
              self.myqueue_clf.enqueue(outputs_clf.tolist())

              self.myqueue_clf.dequeue()                 #####

              #passive_count = 0
              #best2,best1=outputs_clf.argsort()[-2:][::1]
            # top2_diff.append(float(outputs_clf[best1]-outputs_clf[best2]))
            # preds.append(outputs_clf)
            #  best1_lst.append(outputs_clf[best1])
            #  pred_cls.append(best1)
              #tmp=outputs_clf.argmax()
          #    print('\n',k,'class:',best1,'\tbest1:',outputs_clf[best1],'\tdiff: ',outputs_clf[best1]-outputs_clf[best2])
              
              if self.clf_strategy == 'raw':
                  self.clf_selected_queue = outputs_clf
              elif self.clf_strategy == 'median':
                  self.clf_selected_queue = self.myqueue_clf.median
              elif self.clf_strategy == 'ma':
                  self.clf_selected_queue = self.myqueue_clf.ma
              elif self.clf_strategy == 'ewma':
                  self.clf_selected_queue = self.myqueue_clf.ewma
              
              
              
              best2,best1=self.clf_selected_queue.argsort()[-2:][::1]
              top2_diff=float(self.clf_selected_queue[best1]-self.clf_selected_queue[best2])
              
            #  print('best1:',best1,'\ttop2_diff:',top2_diff)
              self.top2_diff_avg.append(top2_diff)
              self.pred_cls.append(best1)
            #  self.prev_best1=best1
              
            #   print('whole clafication part',time.time()-st)
              
              st=time.time()
              if top2_diff>self.top2_diff_threshold:
              #  print('prediction class: ',best1)
                self.pred_cls_queue.insert(0,best1)
                # self.prev_best1=best1
              else:
              #  print('prediction class: ',10)
                self.pred_cls_queue.insert(0,10)
              self.pred_cls_queue.pop()
              

              self.post_processing(best1)
           
      elapsedTime = time.time() - t1
      fps = "(Playback) {:.1f} FPS".format(1/elapsedTime)
      # print(fps,'\nspf',elapsedTime)                       ####
      # print()
    #  cv2.putText(frame, fps, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 0, 255), 1, cv2.LINE_AA)
      return self.final_prediction
       
  def post_processing(self,best1):
      count=1
      for i in range(self.pred_cls_queue_size-1):
        if self.pred_cls_queue[i]!=self.pred_cls_queue[i+1]:
            break
        count+=1


      if count>=self.freq_threshold:
        if self.pred_cls_queue[0]==6 and False:
          if count>=self._6_freq_threshold:
           # print('prediction class: ',self.pred_cls_queue[0])
            self.final_prediction=self.pred_cls_queue[0]
            if self.pred_cls_queue[0] not in [10,self.prev_prediction]:
              self.play(self.pred_cls_queue[0])
              self.prev_prediction=self.pred_cls_queue[0]
          else:
           # print('prediction class: ',10)
            self.final_prediction=10 
        else:
         # print('prediction class: ',self.pred_cls_queue[0])
          self.final_prediction=self.pred_cls_queue[0]
          if self.pred_cls_queue[0] not in [10,self.prev_prediction]:
              self.play(self.pred_cls_queue[0])
              self.prev_prediction=self.pred_cls_queue[0]

        if self.pred_cls_queue[0]==10 and count>=6:     
          self.prev_prediction=10
      else:
       # print('prediction class: ',10)
        self.final_prediction=10   
  
  def play(self,gesture):
    #display(self.key_to_audio[gesture])#,display_id='0')
    playsound(self.key_to_file[gesture])


class VideoStream:
    def __init__(self, path=""):
        """
        Constructor that returns a video camera input.
        """
        if path == "":
            self.norm_video = cv2.VideoCapture(0)
            self.real_time = True
        else:
            self.norm_video = cv2.VideoCapture(path)
            self.process_video = cv2.VideoCapture(path)
            self.real_time = False

        self.is_norm_first_time = True
        self.is_process_first_time = True
        self.prev_norm_frame = 0
        self.prev_process_frame = 0
        self.queue = []
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        if not self.real_time:
           self.rotate_code=check_rotation(path)
        else:
           self.rotate_code=None
        #print(self.rotate_code)
        self.pred=Prediction(None)

    def __del__(self):
        """
        Class destructor.
        """
        if self.real_time:
            self.norm_video.release()
        else:
            self.norm_video.release()
            self.process_video.release()

    def get_frame_col(self):
        """
        Camera input for processing.
        Returns a color image.
        """
        success, frame = self.norm_video.read()
        if self.rotate_code is not None:
           frame=cv2.rotate(frame,self.rotate_code)
        if self.real_time:
            self.queue.append(frame)
       # print('in real-time:',success) 
        time.sleep(.030)
        if success:
            frame = cv2.resize(frame, (640, 480))
            ret, frame = cv2.imencode('.jpg', frame)
            self.prev_norm_frame = frame
            return frame.tobytes()
        else:
            return self.prev_norm_frame.tobytes()

    def gesture_recog(self):

        if not self.real_time:
          success, frame = self.process_video.read()
          if self.rotate_code is not None:
             frame=cv2.rotate(frame,self.rotate_code)
        elif len(self.queue)==0:
          success, frame = self.norm_video.read()
          if self.rotate_code is not None :
             frame=cv2.rotate(frame,self.rotate_code)
        else:
          if len(self.queue):
              success=True
              frame=self.queue.pop(0)
          else:
              success=False
        
       # print('in gesture recog:',success) 
        if success:
            gesture=self.pred(frame)
          #  print('inside gesture recog',gesture)
            frame=cv2.resize(frame, (640, 480))
            if gesture!=10:
               cv2.putText(frame, str(gesture), (640-120,120),
                            self.font, 0.006*640, (255, 180, 10), 8, cv2.LINE_AA)
            ret, jpeg = cv2.imencode('.jpg', frame)
            self.prev_process_frame = jpeg
            return jpeg.tobytes()
        else:
            return self.prev_process_frame.tobytes()


app = Flask(__name__)
#run_with_ngrok(app)
stream = VideoStream()


# Home Page
@app.route('/')
def homepage():
    global stream
    stream.norm_video.release()
    if not stream.real_time:
      stream.process_video.release()
    return render_template('homepage.html')


# Real_time Recognition Page
@app.route('/real_time', methods=['POST'])
def real_time():
    global stream
    stream = VideoStream(path="")
    return render_template('real-time.html')


# Video_based Recognition Page
@app.route('/video_based', methods=['POST'])
def video_based():
    global stream
    passed_path = request.form['video_path']
    is_file=os.path.isfile(passed_path)
    if is_file:
        stream = VideoStream(path=passed_path)
        return render_template('video-based.html')
    else:
        return redirect(url_for("homepage"))


# Video_feed Generator Function
def gen(feed_type):
    global stream, app
    while True:
        if feed_type == 'normal_video':
            frame = stream.get_frame_col()
        else:
            frame = stream.gesture_recog()#detect_faces()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# Video_feed URL
@app.route('/video_feed/<string:video>')
def video_feed(video):
    return Response(gen(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


if __name__ == '__main__':
    app.run()
