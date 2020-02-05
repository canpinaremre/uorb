from dronekit import connect, Command, LocationGlobal,VehicleMode
from pymavlink import mavutil
import time, sys, argparse, math
import threading
"""
Not needed for now

from statistics import mean
import cv2
import numpy as np
import picamera
"""
#################################### Connection String and vehicle connection
"""
connection_string       = '127.0.0.1:14540'

print ("Connecting")
vehicle = connect(connection_string, wait_ready=True)
"""
connection_string       = "/dev/ttyAMA0"

print ("Connecting")
vehicle = connect(connection_string,baud=57600, wait_ready=True)


##################################### Objects
class LandSiteObject:
    def __init__(self,lat,lon,flagName,letter):
        self.lat = lat
        self.lon = lon
        self.flagName = flagName
        self.letter = letter

class FlagObject:
    def __init__(self,landSiteLetter,landOrder,flagName):
        self.landSiteLetter = landSiteLetter
        self.landOrder = landOrder
        self.flagName = flagName


##################################### SETUP
STM = FlagObject('N',1,"STM")
ODTU = FlagObject('N',2,"ODTU")
ORT = FlagObject('N',3,"ORT")
LAND = FlagObject('N',4,"LAND")

A = LandSiteObject(2,2,"N",'A')
B = LandSiteObject(-2,2,"N",'B')
C = LandSiteObject(-2,-2,"N",'C')
D = LandSiteObject(2,-2,"N",'D')

land_sites = [A,B,C,D]
flag_objects = [STM,ODTU,ORT,LAND]


x,y,z =0,0,0

vision_altitude = 6 #meter

drive_with_meter = 0b111111111000
drive_with_speed = 0b111111000111

drive_type = drive_with_meter #initial

###################################### FUNCTIONS

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

def land():
    print("Land !")
    while vehicle.mode != "LAND":
        vehicle._master.set_mode_px4('LAND',None,None)
        print ("Trying land")
        time.sleep(0.3)
    time.sleep(2)
    print ("Landed!")

def tryArming():
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


def startThread():
    t = threading.Thread(target=setpoint_buffer)

    t.daemon = True

    t.start()

def startOffboardMode():
    while vehicle.mode != "OFFBOARD":
        vehicle._master.set_mode_px4('OFFBOARD',None,None)
        print ("Trying offboard")
        time.sleep(0.3)

def print_status():
    # Display basic vehicle state
    print (" Type: %s" % vehicle._vehicle_type)
    print (" Armed: %s" % vehicle.armed)
    print (" System status: %s" % vehicle.system_status.state)
    print (" GPS: %s" % vehicle.gps_0)
    print (" Alt: %s" % vehicle.location.global_relative_frame.alt)



def takeOff(desiredAltitude):
    #takeoff to desired altitude
    return True

def goToLocation(lat,lon,alt):
    #goto desired locaiton
    return True

def defineTheFlag(landSiteLetter):
    #define the flag and write name of it to the landSite
    return True 

def landWithVision(flagName):
    #land with vision to the flag
    return True

def rtl():
    #return to launch
    return True





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

print_status()

tryArming()
startThread()
startOffboardMode()


takeOff(8) #Takeoff to 8 meters
i = 1
###################################### LOOP

while True:
    print("Land Order = ",i)

    if (i == 5):#So we should RTL
        rtl()
        break

    for flag in flag_objects:
        if (flag.landOrder == i):
            if(flag.landSiteLetter == 'N'):
                for land in land_sites:
                    if (land.flagName == 'N'):
                        goToLocation(land.lat,land.lon,vision_altitude)
                        defineTheFlag(land.letter)
            else:
                for land in land_sites:
                    if (flag.landSiteLetter == land.letter):
                        goToLocation(land.lat,land.lon,vision_altitude)
                        landWithVision(flag.flagName)
                        #sleep for some time
                        takeOff(vision_altitude)
                        i += 1


vehicle.close()