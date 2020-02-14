from dronekit import connect, Command, LocationGlobal,VehicleMode
from pymavlink import mavutil
import time, sys, argparse, math
import threading
import os
import cv2
import numpy as np
import importlib.util

"""
Not needed for now

from statistics import mean
import cv2
import numpy as np
import picamera
"""

#################################### Connection String and vehicle connection
"""
# For Jmavsim
connection_string       = '127.0.0.1:14540'

print ("Connecting")
vehicle = connect(connection_string, wait_ready=True)
"""
connection_string       = "/dev/ttyAMA0"
#Bağlantıyı yapıyoruz
print ("Connecting")
vehicle = connect(connection_string,baud=57600, wait_ready=True)

#objeleri tanımlıyoruz
##################################### Objects
class LandSiteObject:
    def __init__(self,xCoordinate,yCoordinate,flagName,letter):
        self.xCoordinate = xCoordinate
        self.yCoordinate = yCoordinate
        self.flagName = flagName
        self.letter = letter

class FlagObject:
    def __init__(self,landSiteLetter,landOrder,flagName):
        self.landSiteLetter = landSiteLetter
        self.landOrder = landOrder
        self.flagName = flagName

class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        (self.grabbed, self.frame) = self.stream.read()

        self.stopped = False

    def start(self):
        threading.Thread(target=self.update,args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return

            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
#ayar kısmımız
##################################### SETUP
stm = FlagObject('N',1,"stm")
metu = FlagObject('N',2,"metu")
ort = FlagObject('N',3,"ort")
landingfield = FlagObject('N',4,"landingfield")

A = LandSiteObject(3.25,3.25,"N",'A')
B = LandSiteObject(3.25,-3.25,"N",'B')
C = LandSiteObject(-3.25,-3.25,"N",'C')
D = LandSiteObject(-3.25,3.25,"N",'D')

land_sites = [A,B,C,D]
flag_objects = [stm,metu,ort,landingfield]


x,y,z =0,0,0
detected_flag_name = "empty for start"

center_of_object = 320,240 
"""
Obje ekranda sola doğru kaydığı zaman 320 olan yani ilk değer azaldı.
x bizim için ilerisi demek olduğundan eksenler ters döndürülecek
y,x = center_of_object
şimdi x,y olarak baktığımızda obje sola doğru kaydığında y değeri azalıyor.
sol taraf 0 en sağ taraf 640 demek (y ekseni)
obje yukarı doğru giderken x azalıyor
en üst taraf 0 en alt taraf 480 demek ( x ekseni)
sol üst köşe 0,0 sağ alt köşe 480,640
kamera düz bir şekilde drone altına takılırsa eğer
üst kısım ilerisi alt kısım geriyi sağ sağı ve sol solu temsil eder.
cisim kamerada üstte yani droneun ilerisinde gözüküyor ise drone ileri gitmelidir ortalamak için.
x ekseninde hareket orta nokta olan 240 dan ne kadar az ise o kadar ileri.
240-0 arası ise ileri
y ekseni için cisim kamerada eğer sağ tarafta ise
drone sağa gitmelidir.
320 y ekseninin ortası olduğuna göre
320 den ne kadar fazla ise o kadar sağa gitmelidir.cisim sağa gittikçe y ekseni artar çünkü.
"""
number_of_detection = 0

pixel_square_of_image = 0

pixel_square_needed = 172800 # 3/4 x 3/4

number_of_being_sure = 3 #how many detections in a row to be sure

vision_altitude = 6 #meter

distance_tolerance = 0.10 # meter

threshold_for_tf = 0.6 # %60

time_to_takeoff_again = 3 #second

drive_with_meter = 0b111111111000
drive_with_speed = 0b111111000111

drive_type = drive_with_meter #initial

###################################### FUNCTIONS
#drone a sürekli konum basan thread
def setpoint_buffer():
    global x,y,z,drive_type
    while True:
        msg=vehicle.message_factory.set_position_target_local_ned_encode(
            0,
            0,0,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            drive_type,
            x,y,z,
            x,y,z,
            0,0,0,
            0,0)
        vehicle.send_mavlink(msg)
        vehicle.flush()
        time.sleep(0.3)

parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=threshold_for_tf)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='640x480')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

pkg = importlib.util.find_spec('tensorflow')
if pkg is None:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

if use_TPU:
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

CWD_PATH = os.getcwd()

PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

if labels[0] == '???':
    del(labels[0])

if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

frame_rate_calc = 1
freq = cv2.getTickFrequency()

videostream = VideoStream(resolution=(imW,imH),framerate=30).start()

#Sürekli görüntü işleyen thread
def tf_buffer():
    global videostream,freq,frame_rate_calc,input_std,input_mean,floating_model,width,height,output_details,input_details,interpreter,use_TPU,labels,PATH_TO_LABELS,PATH_TO_CKPT,CWD_PATH,GRAPH_NAME,pkg,MODEL_NAME,LABELMAP_NAME
    global min_conf_threshold,resW,resH,imW,imH,args,detected_flag_name,number_of_detection,center_of_object,pixel_square_of_image
    while True:

        frame1 = videostream.read()

        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects

        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
            
                # ekranda gördüğü cismin kaç pixel kare alan kapladığını hesaplıyoruz.
                pixel_square_of_image = (ymax - ymin) * (xmax - xmin)
                #objenin orta noktasını x,y olarak hesaplıyoruz
                center_of_object = int((xmin+xmax)/2),int((ymin+ymax)/2)
                #objenin adını object name içine alıyoruz
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                """
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                """
                #object name'i detected flag name olarak alıyoruz.
                detected_flag_name = object_name
                #yapılan tanımlama (detect) sayısını her tanımlama sonrası arttırıyoruz.
                #bu sayede globalde tuttuğumuz detected flag name güncellenmiş mi anlayabileceğiz.
                number_of_detection += 1

        if cv2.waitKey(1) == ord('q'):
            break    

def land():#Dümdüz iniş moduna geçiren fonksiyon.aktif edildiğinde direk inişe geçiyor.
    print("Land !")
    """
    while vehicle.mode != "LAND":
        vehicle._master.set_mode_px4('LAND',None,None)
        print ("Trying land")
        time.sleep(0.3)
    """
    print("alet iniş moduna geçti")
    time.sleep(2)
    print ("Landed!")

def tryArming():#arm etmeye çalıştığımız fonskiyon.
    while True:
        vehicle._master.mav.command_long_send(
        1, # autopilot system id
        1, # autopilot component id
        400, # command id, ARM/DISARM
        0, # confirmation
        1, # arm!
        0,0,0,0,0,0 # unused parameters for this command
        )
        print("Trying arming")
        time.sleep(0.3)
        if (vehicle.armed == True):
            break
    print ("Armed :",vehicle.armed)

#setpoint threadi başlatıyor.
def startThread():
    t = threading.Thread(target=setpoint_buffer)

    t.daemon = True

    t.start()
#görüntü işleme threadi başlatıyor.
def startTfThread():
    tf = threading.Thread(target=tf_buffer)

    tf.daemon = True

    tf.start()
#offboard moda geçmemizi sağlıyor.
def startOffboardMode():
    while vehicle.mode != "OFFBOARD":
        vehicle._master.set_mode_px4('OFFBOARD',None,None)
        print ("Trying offboard")
        time.sleep(0.3)
#drone'un o anki durumunu ekrana bastırıyor.
def print_status():
    # Display basic vehicle state
    print (" Type: %s" % vehicle._vehicle_type)
    print (" Armed: %s" % vehicle.armed)
    print (" System status: %s" % vehicle.system_status.state)
    print (" GPS: %s" % vehicle.gps_0)
    print (" Alt: %s" % vehicle.location.global_relative_frame.alt)
#drone kalkmadan önceki tüm hazırlığı tek çatı altına topluyoruz.
def readyToTakeoff():
    global drive_with_meter,drive_type
    #drone a verdiğimiz x,y,z yi metre olarak kullanmak için drive type meter olarak seçiliyor.
    drive_type = drive_with_meter
    #standart çalıştırma komutları
    print_status()
    tryArming()
    startOffboardMode()
    print_status()

    return True

def tryDisArming():#aleti disarm etmeye motorları durdurmaya çalışan kod
    #disarm
    while True:
        vehicle._master.mav.command_long_send(
        1, # autopilot system id
        1, # autopilot component id
        400, # command id, ARM/DISARM
        0, # confirmation
        0, # arm!
        0,0,0,0,0,0 # unused parameters for this command
        )
        print("Trying Disarming")
        time.sleep(0.3)
        if (vehicle.armed == False):
            break
    print ("Armed :",vehicle.armed)

    return True

def shutDownTheMotors():#disarm etmeden önce ve sonra drone durumunu basan fonksiyon.
    #stop motors afer landing
    print_status()
    tryDisArming()
    print_status()
    return True

def goToLocation(xTarget,yTarget,altTarget):#verilen lokasyona gidilmesini sağlıyor.
    global startYaw,x,y,z,distance_tolerance
    #goto desired locaiton
    #ileri sağ frame'i kuzey doğu frame'e çeviriyoruz sonra bunu x,y global değişkeni olarak gönderiyoruz.
    print(xTarget)
    print(yTarget)
    print(altTarget)
    print("x,y,z konumuna gidiyoruz. yani")
    x,y = bodyToNedFrame(xTarget,yTarget,startYaw)
    z = -altTarget
    print(x)
    print(y)
    print(z)
    print("Ned konumuna gidiyoruz.")
    #sürekli konuma varıp varmadığımızı kontrol ediyoruz.
    #konuma varana kadar bekliyorz.
    """
    while not atTheTargetYet(xTarget,yTarget,altTarget):
        time.sleep(0.1)
    """
    print("normalde konuma gitmemizi bekliyoruz ancak şimdi 2 sn bekleyip onay gönderiyorum ve konuma varmışız gibi yapıyouz.")
    time.sleep(2) #konuma gittiğimiz zamanmış gibi biraz bekliyoz orjinal kodda yok bu satır.
    return True

def defineTheFlag(landSiteLetter):
    #define the flag and write name of it to the landSite
    global detected_flag_name,number_of_detection,number_of_being_sure

    sure_counter = 0

    temp_flag_name = detected_flag_name#geçici olarak bayrağı ve tanımlama sayısını alıyoruz.
    temp_number = number_of_detection#bayrağı almamızın sebebi ard arda gelişini kontrol etmek.detection sayısı da görüntünün güncelliği için.

    while True:
        if number_of_detection > temp_number:#önce hafızadaki detection yeni mi ona bakıyoruz eğer yeni ise devam değilse bekle
            if detected_flag_name == temp_flag_name:#eğer alınan yeni gönrütü hafızaya alınan geçici görüntü ile aynı ise counter 1 artar.
                sure_counter +=1
                temp_number = number_of_detection#number_of_detection > temp_number kontrolü için temp olarak alınır.
                if sure_counter >= number_of_being_sure:#counter belli bir (number_of_being_sure) sayının üzerindeyse bayrağı kesin olarak alırız.
                    break
            else:#eğer aldığımız yeni detection temp ile aynı bayrak değilse.temp i aldığımız yeni bayrak yaparız.
                sure_counter = 0
                temp_flag_name = detected_flag_name#bu sefer yeni bayrak temp olur ve sonraki döngüde bunun number_of_being_sure bu kadar ard arda çıkması beklenir.
                temp_number = number_of_detection#yine güncelliği kontrol etmek için temp number
        time.sleep(0.1)



    #after being sure of detection:bayrak kesinleştikten sonra temp de tuttuğumuz ama artık emin olduğumuz bayrağı
    for land in land_sites:
        if(land.letter == landSiteLetter):
            land.flagName = temp_flag_name#bulunduğumuz land'in flag name kısmına yazarız
    
    for flag in flag_objects:
        if(flag.flagName == temp_flag_name):
            flag.landSiteLetter = landSiteLetter#aynı zamanda bayrak objelerinden bulduğumuz bayrağa hangi land site da olduğumuzu söyleriz.
    
    return True 

def landWithVision(flagName):
    #land with vision to the flag
    global center_of_object,detected_flag_name,number_of_detection,x,y,z,drive_type,drive_with_meter,drive_with_speed,pixel_square_of_image,pixel_square_needed,startYaw

    
    temp_number = number_of_detection#yine görüntü güncelliği için alıyouz.

    drive_type = drive_with_speed#artık hız ile sürüyoruz.
    x,y,z = 0,0,0#başlangıç hızı 0

    while True:
        if number_of_detection > temp_number:#eğer görüntü güncel ise,değilse 0.01 sn bekle
            if detected_flag_name == flagName:#gördüğü bayrakta bizim ineceğimiz bayrak ise
                temp_number = number_of_detection
                yCenter,xCenter = center_of_object # y,x = center_of_object
                #gerekli hesaplamalar ile orta noktaya olan mesafeye göre hız değerleri belirliyoruz.
                xPix = (240 - xCenter) * 0.004 #Max speed is 1 m/s
                yPix = (yCenter - 320) * 0.003 #Max speed is 1 m/s
                #hız değerlerini NED frame'e çeviriyoruz ve yayınlıyoruz.
                x,y = bodyToNedFrame(xPix,yPix,startYaw)
                z = 0.2 # m/s down speed
                print("x,y,z hızımız: (z pozitif çünkü ned frame için aşağı demek.değerler x,y,z cinsinden.Ned cinsinden olanalr kodda x,y,z değişkenleri ile global yayınlanıyor)")
                print(xPix)
                print(yPix)
                print(z)
                if pixel_square_of_image >= pixel_square_needed:#eğer ekrandaki görüntü yeteri kadar büyümüş ise yaklaştık demek.inişe geçmiş için döngüden çık.
                    #go to land mode
                    print("ekranda yeteri kadar çok pixel kapladı.İnişe geçiyorum.")
                    x,y,z = 0,0,0
                    break
            else:#eğer yanlış,farklı bir bayrak görmüş ise hızı sıfırla ve yeni detection bekle.
                temp_number = number_of_detection
                print("gördüğüm bayrak ineceğim bayrak değil o yüzden bekliyorum.")
                #ignore wrong detections and wait with zero speed.
                x,y,z = 0,0,0
        else:#eğer hiç bayrak göremiyor isek hızı sıfırla ve yeni detection bekle.
            #if there is no detection set all the speed to zero
            print("hiç bayrak göremiyorum hızımı sıfırladım.")
            x,y,z = 0,0,0
        time.sleep(0.01)

    print("Lande geç")
    land()#inişe geçiyoruz.
    drive_type = drive_with_meter#kalkarken yine metre kullanıcağımız için güvenlik amaçlı tekrar metreye dönüyoruz.
    #after landing
    print("iniş sonrası motorları kapıyorum")
    #shutDownTheMotors()#motorları kapatıyoruz.
    return True

def rtl():#görev bitince eve dönmemizi ve türk bayrağına göre inmemizi sağlıyor.
    print("eve dönüyorum")
    goToLocation(0,0,-vision_altitude)
    print("türk bayrağına bakarak iniş yapıcam.")
    landWithVision("turkishflag")
    print("eve dönme bitti.")
    #return to launch
    return True

def bodyToNedFrame(xBody,yBody,yawBody):#X;Y,Z koordinat düzlemini NED düzlemine çeviriyor.(drone NED kullanıyor.)
    xNed =  (xBody * math.cos(yawBody) ) - ( yBody * math.sin(yawBody) )
    yNed =  (xBody * math.sin(yawBody) ) + ( yBody * math.cos(yawBody) )
    return xNed,yNed

def atTheTargetYet(xTarget,yTarget,zTarget):#hedefe varıp varmadğımızı kontrol ediyor.bunu XYZ düzleminde kontrol ediyor.
    global startYaw,distance_tolerance
    #goto desired locaiton
    #bu fonskiyonu bu scriptte kullanmadık.onun yerine 2 sn bekle dedik.test için.
    xTarget,yTarget = bodyToNedFrame(xTarget,yTarget,startYaw)#istediğimiz xyz NED e çevrilir.
    zTarget = -zTarget

    north = vehicle.location.local_frame.north#drone konumu canlı olarak ned olarak alınır.
    east = vehicle.location.local_frame.east
    down = vehicle.location.local_frame.down
    if (abs(xTarget-north) < distance_tolerance):#sonra bu iki ned değerleri arasındaki fark kontrol edilir.
        if(abs(yTarget-east) < distance_tolerance):
            if (abs(zTarget-down) < distance_tolerance):
                print("Target point reached")#add x y z tolerans yeteri kadar az ise true döner
                return True
    print("Not at target yet")#değilse false
    return False

#bugünlük home position almıyoruz.drone bağlantı yapıp beklesin sadece
"""
##################################### START
home_position_set = False
#Create a message listener for home position fix
@vehicle.on_message('HOME_POSITION')
def listener(self, name, home_position):
    global home_position_set
    home_position_set = True


while not home_position_set:
    print ("Waiting for home position...")
    time.sleep(1)
"""
print("setpoint thread başlatıyorum")
startThread()
print("görüntü işleme thread başlatıyorum")
startTfThread()
print("kalkış hazırlıklarını yapıyorum (aslında yapmıyom)")
#readyToTakeoff()

startYaw = vehicle.attitude.yaw
print("kalkış yapıyom")
goToLocation(0,0,vision_altitude) # take off

i = 1
###################################### LOOP

while True:
    print("Land Order = ",i)

    if (i == 5):#So we should RTL
        print("son aşamaya geldik.Eve dön türk bayrağına in.")
        rtl()
        break

    for flag in flag_objects:
        if (flag.landOrder == i):
            if(flag.landSiteLetter == 'N'):
                for land in land_sites:
                    if (land.flagName == 'N'):
                        print("arayıpta bulamadığımız bayrağın sırası ve adı:")
                        print(i)
                        print(flag.flagName)
                        print("sıradaki bayrağın iniş yeri belli değil.Bir sonraki içi boş olan  iniş yeri olan")
                        print(land.letter)
                        print("lokasyonuna gidiyoruz..")
                        goToLocation(land.xCoordinate,land.yCoordinate,vision_altitude)
                        print("lokasyona vardık.Şimdi bu lokasyonda bulunan bayrağı tanımlamaya çalışıcağız.")
                        defineTheFlag(land.letter)
                        print("bayrağı tanımladık.Döngü başa dönecek.")
                        
            else:
                for land in land_sites:
                    if (flag.landSiteLetter == land.letter):
                        print("ineceğimiz bağrağın iniş yerini biliyoruz.Bayrak adı:")
                        print(flag.flagName)
                        print("iniş yerinin adı:")
                        print(land.letter)
                        print("Şimdi bu iniş yerine gidiyoruz.")
                        goToLocation(land.xCoordinate,land.yCoordinate,vision_altitude)
                        print("iniş yerine vardık.Şimdi kameraya indiğimiz bayrağın adını verip buna göre hız ile ineceğiz.")
                        landWithVision(flag.flagName)
                        print("iniş bitti motorlar kapandı ve biraz bekleyeceğiz.")
                        time.sleep(time_to_takeoff_again)
                        print("bekleme bitti şimdi hazırlıklarımızı yapıp tekrar kalkmak için herşeyi baştan çalıştıracağız")
                        #readyToTakeoff()
                        print("sonraki bayrağa geçmek için land order olan i arttırılıyor.Sonraki konuma gitmek için kalkış yapıcağız.O yüzden takeoff vermemize gerek yok.")
                        i += 1