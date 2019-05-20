#!/usr/bin/python
import numpy as np
import Acqu as aq
#import A2G4
#import AcquDetector as aqdet
#import CrystalBall as cb
import argparse 
import time
#import Timepix
#import igraph as ig
#import cairo
#import plotly.plotly as py
#import plotly.graph_objs as go
import ROOT
from rootpy.plotting import histogram, Hist2D,Hist,  Canvas

#canvas      = Canvas()
adchist     = Hist(10000,0,10000,name='adcs')
adchist2d   = Hist2D(500,0,500,68000,0,68000,name='adcvalues')
taggerchan  = Hist(366,0,366)
#taggertime  = Hist(10240,0,10240)
tpxXY1      = Hist2D(256,0,256,256,0,256)
#tpxXY2      = Hist2D(256,0,256,256,0,256)
tpxTime      = Hist(10000,0,1)
#tpxtimeDiff = Hist(2000,-1000,1000)
#tpxtimeDiff2 = Hist2D(2000,-1000,1000,200,0,40000000)
#adccompare  = Hist2D(1000,0,10000,1000,0,10000)
#adcevent    = Hist2D(10000,0,10000,7000,11000,18000)
#adcevent2   = Hist2D(10000,0,10000,1000,0,65536)

#taggtpxDiff  = Hist(20000,-4E8,4E8)
#taggtpxDiff  = Hist(10000000,-1E8,1E8)
#taggtpxDiff2 = Hist2D(1000,-4E8,4E8,1000,0,5E8)
#taggtpxComp2 = Hist2D(1000,-4E8,4E8,1000,0,5E8)

#taggtpxDiff2 = Hist2D(100000,-3E8,4E8,200,0,3E8)
#taggtpxComp2 = Hist2D(10000,-4E5,4E5,1000,0,5E8)

taggerchantime  = Hist2D(2000,-1000,1000,366,0,366,name="taggerchantime")
cbchanenergy    = Hist2D(100,0,1200,720,0,720,name="cbchanenergy")
cbchantime      = Hist2D(2000,-1000,1000,720,0,720,name="cbchantime")
tapschanenergy  = Hist2D(100,0,1200,384,0,384,name="tapschanenergy")
tapschantime    = Hist2D(2000,-1000,1000,384,0,384,name="tapschantime")

def main():       
    global adchist                                             
    parser = argparse.ArgumentParser()
    parser.add_argument("fileName", help="AcquDAQ data file")  #Add args and opts
    args = parser.parse_args()                                 #parse them
    
    flist = [args.fileName]                      #make a list of all files to be processed, somehow
    outFile = ROOT.TFile('/scratch/test.root','recreate') 
    #do init stuff in here
    #...
    for file in flist:                                         #for each file in the list
        start  = time.time()
        aq.openFile(file)
        print aq.fileInfo
        #print aq.fileTrailer
        #aqdet.LoadDetectors(['aux/CrystalBallEnergy.json','aux/CrystalBallTime.json'])
        #aqdet.LoadDetectors(['aux/CrystalBallEnergy.json','aux/TAPSEnergy.json'])
        #aqdet.LoadDetectors(['aux/CrystalBallEnergyNew.json'])#,'/home/simong/AcquPy/aux/tagger.json'])#,'aux/TAPSEnergy.json'])
        #aqdet.LoadDetectors(['/home/simong/AcquPy/aux/tagger.json','/home/simong/AcquPy/aux/CrystalBallEnergy.json','/home/simong/AcquPy/aux/TAPSEnergy.json'])
        #aqdet.LoadDetectors(['/home/simong/AcquPy/aux/tagger.json','/home/simong/AcquPy/aux/CrystalBallEnergy.json','/home/simong/AcquPy/aux/CrystalBallTime.json','/home/simong/AcquPy/aux/TAPSEnergy.json','/home/simong/AcquPy/aux/TAPSTime.json'])
        #aqdet.LoadDetectors(['/home/simong/AcquPy/aux/taggerNew.json','/home/simong/AcquPy/aux/clock.json'])
        middle = time.time()
        print('Open file: ',middle-start)
        #aq.runFunction(fillTP3,0,20000)
        aq.runFunction(fillADCHist,0,100000)
        print aq.eventNo
        #aq.runEPICSFunction(printEpics)
        #aq.listEpicsPVs()
        #aq.runEPICSFunction(getEpics)
        #aq.runEPICSFunction(dumpTimepix)
        
        #aq.runFunction(fillTagger,0,10)
        #aq.runFunction(fillTagger)     
        end = time.time()
        print('Process:   ',end-middle)
        #aq.runFunction(ADCEventTrend,100000)
        #aq.runFunction(taggerTimepix,0,6000)
    #####A2G4.openFile("/w/work1/home/simong/Simulation/G4Out/GP_GP_20982.root")
    #tapschantime.Write()
    #tapschanenergy.Write()
    #cbchantime.Write()
    #cbchanenergy.Write()
    #taggerchantime.Write()
    #taggerchan.Write()
    #adccompare.Write()
    adchist.Write()
    adchist2d.Write()
    #tpxtimeDiff2.Write()
    #tpxtimeDiff.Write()
    #tpxXY1.Write()
    #tpxTime.Write()
    #tpxXY2.Write()
    #adcevent.Write()
    #adcevent2.Write()
    #taggtpxDiff.Write()
    #taggtpxDiff2.Write()
    #taggtpxComp2.Write()

       
def fillTagger():
    np.set_printoptions(threshold=np.nan)
    aqdet.Calibrate(aq.adcArray)
    graph = aqdet.GetGraph('CB',['Energy','Time'])
    print graph.vs['Energy']
    #print graph.vs['Time']
    #CBgraph = aqdet.detgraphs['CB'].subgraph(data['CB']['channel'])
    #print graph
    #cb.Build_Clusters(graph)
    
    #channellist = data['CB']['channel'].tolist()
    #valuelist = [int(x) for x in data['CB']['value']]
    #subgraph = aqdet.detgraphs['CB'].subgraph(channellist)
    #print('CB')
    #print(data['CB'].community_optimal_modularity())
    #print(data['CB'].clusters())
    #valuelist = [int(x) for x in data['CB'].vs['value']]
    #valuelist = list(range(data['CB'].vcount()))
    #data['CB'].vs['label'] = valuelist
    #ig.plot(CBgraph,keep_aspect_ratio=True,bbox=(0,0,1200,1200),layout=CBgraph.vs['coordinates'])
    #ig.plot(data['CB'],keep_aspect_ratio=True,bbox=(0,0,1200,1200),layout=data['CB'].vs['coordinates'])
    #ig.plot(data['CB'],keep_aspect_ratio=True,bbox=(0,0,600,600),layout=data['CB'].vs['coordinates'])


    #if(data['TAPS'].vcount()):
    #    print('TAPS')
    #    print(data['TAPS'].vs['coordinates'])
    #    valuelist = [int(x) for x in data['TAPS'].vs['value']]
    #    data['TAPS'].vs['label'] = valuelist
        #ig.plot(data['TAPS'],keep_aspect_ratio=True,rescale=False,layout=data['TAPS'].vs['coordinates'])
    #cb.Build_Clusters(data['CB'])
    #print(data)
    #print(aqdet.frame)
    
    #print channelArray
    #if len(channelArray):
    #print(data['CB'][['channel','offset','scale','raw','value']])
    #print(data['tagger'][['channel','value']].view(np.float).reshape(-1,2))
    #taggerchan.fill_array(data['tagger'][['channel']])
    #taggerchantime.fill_array(data['tagger'][['value','channel']].view(np.float).reshape(-1,2))
    ####cbchanenergy.fill_array(data['CB'][['value','channel']].view(np.float).reshape(-1,2))
    #cbchantime.fill_array(data['CBTime'][['value','channel']].view(np.float).reshape(-1,2))
    #tapschanenergy.fill_array(data['TAPS'][['value','channel']].view(np.float).reshape(-1,2))
    #tapschantime.fill_array(data['TAPSTime'][['value','channel']].view(np.float).reshape(-1,2))
    #print(len(data['CBTime']),len(data['CB']))
    #print(len(data['TAPSTime']),len(data['TAPS']))
    #print(' ')
    #print
    #    taggerchantime.fill(channelArrayaqdet.TaggerChannels(aq.adcArray))
    if(aq.eventNo%1000==0):
        print('number',aq.eventNo)

def findEpics():
    if(len(aq.epicsBuffers)):
        print(aq.epicsBuffers)
        
def printEpics():
    aq.dumpEpicsBuffer()
    
def getEpics():
    print('chan 0')
    print(aq.getEpicsPV(0))
    print('chan 1')
    print(aq.getEpicsPV('TAGG:MagneticField'))
    print('chan 2')
    print(aq.getEpicsPV(2))
    print('chan 3')
    print(aq.getEpicsPV(3))
    print('chan 4')
    print(aq.getEpicsPV(4))

def dumpTimepix():
    # Create timepix time array
    nHitsA   = aq.getEpicsPV('PPOL:TIMEPIXA:NHITS')
    encodedA = aq.getEpicsPV('PPOL:TIMEPIXA:ENCODED')
    TimepixAData = aqdet.TimepixDecode(nHitsA,encodedA)
    nHitsB   = aq.getEpicsPV('PPOL:TIMEPIXB:NHITS')
    encodedB = aq.getEpicsPV('PPOL:TIMEPIXB:ENCODED')
    TimepixBData = aqdet.TimepixDecode(nHitsB,encodedB)
    nsTimeA = 25*TimepixAData[['ToA']].astype(float) - 25/16*TimepixAData[['FToA']].astype(float)
    nsTimeB = 25*TimepixBData[['ToA']].astype(float) - 25/16*TimepixBData[['FToA']].astype(float)

    #print TimepixAData[['y','x']].tolist()

    for i, data in enumerate(TimepixAData):
        timeDataqdetD = np.full((len(nsTimeB),2),nsTimeA[i])        
        timeDataqdetD[:,0] = nsTimeB-nsTimeA[i]   
        timeData = nsTimeB-nsTimeA[i]
        #print timeDataqdetD
        tpxtimeDiff.fill_array(timeDataqdetD[:,0])
        #tpxtimeDiff.fill_array(nsTimeB-nsTimeA[i])
        tpxtimeDiff2.fill_array(timeDataqdetD)
        tpxXY1.Fill(data['y'],data['x'])
        
    for i, data in enumerate(TimepixBData):
        tpxXY2.Fill(data['y'],data['x'])
        

taggedADC        = [[]]*2
taggedClockLong  = [[]]*2
taggedClockShort = [[]]*2
previousStart    = 0
taggedClockStart = 0
addToArray       = True
start301         = 0
prev301          = 0
arrayIndex       = 0
epicsIndex       = 0

def taggerTimepix():
    global taggedADC
    global taggedClockLong
    global taggedClockShort
    global previousStart
    global taggedClockStart
    global start301
    global prev301
    global arrayIndex
    global epicsIndex

    adc       = aq.adcArray
    longClock = adc[np.where(adc[:,0]==301)][0][1]
    
    if((longClock-prev301)==373*4):
        prev301 = longClock
        arrayIndex += 1
        taggedADC[arrayIndex%2]        = []
        taggedClockLong[arrayIndex%2]  = []
        taggedClockShort[arrayIndex%2] = []
        #print longClock
        #print epicsIndex,arrayIndex,'TRIGGER'

    #print taggedADC
    taggedADC[arrayIndex%2] += [aqdet.TaggerChannels(adc)]
    taggedClockLong[arrayIndex%2]  += [adc[np.where(adc[:,0]==301)[0][0],1]]
    taggedClockShort[arrayIndex%2] += [adc[np.where(adc[:,0]==300)[0][0],1]]

    if(len(aq.epicsBuffers) and aq.eventNo!=0):
            
        # Create timepix time array
        nHitsA       = aq.getEpicsPV('PPOL:TIMEPIXA:NHITS')
        encodedA     = aq.getEpicsPV('PPOL:TIMEPIXA:ENCODED')
        
        if(previousStart != encodedA[0]): 
            previousStart = encodedA[0]
            if(prev301==0):
                prev301 = longClock
                return

            # Shift timing ADCs to start from 0
            taggedClockStart = taggedClockLong[epicsIndex%2][0]*65536*2.5
            print(taggedClockStart)
        
            TimepixAData = aqdet.TimepixDecode(nHitsA,encodedA)
            nsTimeA      = 25*TimepixAData[['ToA']].astype(float) - 25/16*TimepixAData[['FToA']].astype(float)
            taggedClock  = [(taggedClockShort[epicsIndex%2][i]+taggedClockLong[epicsIndex%2][i]*65536)*2.5-taggedClockStart for i in range(len(taggedClockShort[epicsIndex%2]))]
            #Shortened list of timepix hits
            #removeCluster = nsTimeA
            removeCluster = np.extract(([0]+np.diff(nsTimeA))>10,nsTimeA)
            

            #print taggedClock
            #Fill histograms
            for i, time in enumerate(taggedClock):
                for channel in taggedADC[epicsIndex%2][i]:
                    if(channel[0]==81):
                        timeDiff2D = np.full((len(removeCluster),2),time)
                        timeDiff2D[:,0] = removeCluster-time-channel[1]
                        #timeDiff = np.extract(abs(removeCluster-time-channel[1])<1E8,removeCluster-time-channel[1])
                        taggtpxDiff.fill_array(timeDiff2D[:,0])
                        taggtpxDiff2.fill_array(timeDiff2D)
                        #for j, diff in enumerate(timeDiff):
                        #    taggtpxDiff2.Fill(diff,time)
            
            # Reset and fill for next epics buffer
            taggedADC[epicsIndex%2]        = []
            taggedClockLong[epicsIndex%2]  = []
            taggedClockShort[epicsIndex%2] = []
            #epicsIndex += 1
            #print prev301,longClock
            #print epicsIndex,arrayIndex,'EPICS'
            epicsIndex = arrayIndex



def fillADCHist():
    global adchist
    global adchist2d
    adchist.fill_array(aq.adcArray['adc'])
    adchist2d.fill_array(np.vstack((aq.adcArray['adc'],aq.adcArray['val'])).T)    
    if(aq.eventNo%10000==0):
        print('number',aq.eventNo)

def fillTP3():        
    if(aq.epicsEvent==1):
        nHitsA       = aq.getEpicsPV('PPOL:TIMEPIXA:NHITS')
        encodedA     = aq.getEpicsPV('PPOL:TIMEPIXA:ENCODED')
        TimepixAData = Timepix.Decode(nHitsA,encodedA)
        print TimepixAData['y']
        tpxXY1.fill_array(np.vstack((TimepixAData['y'],TimepixAData['x'])).T)
        nsTimeA = 25*TimepixAData[['ToA']].astype(float) - 25/16*TimepixAData[['FToA']].astype(float)
        print nsTimeA/1000000000
        tpxTime.fill_array(nsTimeA/1000000000)

        
def ADCTrends():
    adc1 = 927
    adc2 = 1026
    adc = aq.adcArray
    index1 = np.where(adc[:,0]==adc1)[0]
    index2 = np.where(adc[:,0]==adc2)[0]
    for indexA in index1:
        valueA = aq.adcArray[indexA,1]
        for indexB in index2:        
            adccompare.Fill(valueA,aq.adcArray[indexB,1])

def ADCEventTrend():
    adc1 = 301
    adc2 = 300
    adc = aq.adcArray
    index1 = np.where(adc[:,0]==adc1)[0]
    index2 = np.where(adc[:,0]==adc2)[0]
    
    for indexA in index1:        
        adcevent.Fill(aq.eventNo,aq.adcArray[indexA,1])
    for indexA in index2:
        adcevent2.Fill(aq.eventNo,aq.adcArray[indexA,1])
        

if __name__ == "__main__": main()  # call main comes at the end: a quirk of python
