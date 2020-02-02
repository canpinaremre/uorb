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

def TakeOff(desitedAltitude):
    #takeoff to desired altitude
    return True

def GoToLocation(lat,lon,alt):
    #goto desired locaiton
    return True

def DefineTheFlag(landSiteLetter):
    #define the flag and write name of it to the landSite
    return True 

def LandWithVision(flagName):
    #land with vision to the flag
    return True

def RTL():
    #return to launch
    return True

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

TakeOff(8) #Takeoff to 8 meters

#LOOP:
i = 1
while True:
    print("Land Order = ",i)

    if (i == 5):#So we should RTL
        RTL()
        break

    for flag in flag_objects:
        if (flag.landOrder == i):
            if(flag.landSiteLetter == 'N'):
                for land in land_sites:
                    if (land.flagName == 'N'):
                        GoToLocation(land.lat,land.lon,8)
                        DefineTheFlag(land.letter)
            else:
                for land in land_sites:
                    if (flag.landSiteLetter == land.letter):
                        GoToLocation(land.lat,land.lon,8)
                        LandWithVision(flag.flagName)
                        #sleep for some time
                        TakeOff(8)
                        i += 1
