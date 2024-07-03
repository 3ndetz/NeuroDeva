import os
import time
thisfolder = os.path.dirname(os.path.realpath(__file__))
print("started",thisfolder)
os.startfile(thisfolder+'/StartSheepChat.bat')
#os.startfile(thisfolder+'/VTubeNoSteamStart.bat')
time.sleep(10)
print(1)
time.sleep(10)
print("exiting...")