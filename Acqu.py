#!/usr/bin/python
import numpy as np
import sys
import Mk2Format as mk2
import Mk1Format as mk1


# Reading AcquDAQ data 
 
# A whole acqu.dat file gets handled in a numpy memap as an array of unsigned ints.
# Most of an acqu file is adc data of the form index, datum which is viewed as unsigned shorts.

# numpy handles the buffering between the diskfile and the array.
# For non adc data it handled by "casting" to np.datatypes (like c structs)

# ACQU data buffer header types and data delimiters
EHeadBuff     = 0x10101010  # header buffer (experimental parameters)
EEndBuff      = 0x30303030  # end-of-file buffer
EKillBuff     = 0x40404040  # shut-down buffer
ENullADC      = -1          # undefined ADC value
ENullHit      = 0xFFFFFFFF  # undefined hit index (end of hit buffer)
ENullStore    = 0x80008000  # for multi-hit ADC handling
ENullFloat    = -99999      # optional null indicator

#eventHead = np.dtype( [ ('evNo',   '<i4'),
#                        ('evEnd','<u4')])

scalerHead = np.dtype ( [ ('nscaler', '<u4')])
scalerBuff = np.dtype ( [ ('index', '<u4'),
                          ('datum', '<u4')])
                         
fp         = 0        #buffer for a whole data file
buffEnd    = 0        #end index of buffer in byte array

ADC        = 0        #ADC array with all ADCs by index
ScalerCurr = 0        #Current scaler buffer with all scalers by index
ScalerAcc  = 0        #Accumulated scaler buffer with all scalers by index

adcArray      = []
epicsBuffers  = []
epicsInfo     = []
epicsPVs      = []
scalerBuffers = []
eventNo       = 0
streamEventNo = 0

    
ScalerOffset  = 0
NScaler       = 0
ScalerLen     = 0

IsEpicsEvent  = 0
EpicsOffset   = 0
EpicsLen      = 0
           
##################################################################
# Open file, interpret and skip header
##################################################################

def openFile(fileName):
    #globals we need to modify in this function
    global fp                
    fp = np.memmap(fileName, dtype='uint32', mode='r') #make a memory map of the whole file as unsigned ints
    determineDataFormat()
    processHeader()

        
##################################################################
# Close the file
##################################################################

def closeFile():
    global fp
    del fp

    
##################################################################
# Check if the file is Mk1 or Mk2 data format
##################################################################

def determineDataFormat():
    global df
    global fp
    if(np.all(fp[0:2]==EHeadBuff)):
        df = mk2
    elif(np.all(fp[0:1]==EHeadBuff)):
        df = mk1
    else:
        sys.exit('Unknown Data Format')
        
    print(df.name)
    
    
##################################################################
# Get the information about the file and skip past the header
##################################################################

def processHeader():
    #globals we need to modify in this function
    global fileInfo
    global fileTrailer
    global ADCDetails
    global maxADC
    global ScalerDetails 
    global ModuleDetails  
    global fp
    
    byteOffset = df.ByteOffset
    
    # Get File Header
    fileInfo  = np.frombuffer(fp, dtype=df.recHead, count=1, offset=byteOffset) #get the information from the header
    byteOffset += fileInfo.itemsize
    maxADC = fileInfo['fNADC'][0]
    #print(df.recHead)
    #print(fileInfo)
    
    # Get ADC Addresses
    if(df.adcHeadExists):
        ADCDetails = np.frombuffer(fp, dtype=df.adcHead, count=fileInfo['fNADC'][0], offset=byteOffset)
        byteOffset += ADCDetails.itemsize*fileInfo['fNADC'][0]

    # Get Scaler Addresses
    if(df.scalerHeadExists):
        ScalerDetails = np.frombuffer(fp, dtype=df.scalerHead, count=fileInfo['fNScaler'][0], offset=byteOffset)
        byteOffset += ScalerDetails.itemsize*fileInfo['fNScaler'][0]
        #print(ScalerDetails)
        
    # Get Module Details
    if(df.moduleHeadExists):
        ModuleDetails = np.frombuffer(fp, dtype=df.moduleHead, count=fileInfo['fNModule'][0], offset=byteOffset)
        byteOffset += ModuleDetails.itemsize*fileInfo['fNModule'][0]
        #print(ModuleDetails)
        
    if(df.scalerHeadExists and df.moduleHeadExists):
        df.MakeScalerArray(ModuleDetails,ScalerDetails)
        #print(df.scalerArray)

    # Get File Tail
    fileTrailer = np.frombuffer(fp, dtype=df.recTrailer, count=1, offset = int(len(fp)/2-df.recTrailer.itemsize))

    #printHeaderDetails()

    # Remove trailing data, and header buffer then reshape the data into a buffer array
    bufferLocs = np.where(fp==df.EDataBuff)[0]
    #print(bufferLocs)
    #print(fileInfo['fRecLen'][0])
    #print(df.recSize/32)

    #Fuck you messing up my code
    fp = fp[:np.where(fp==EEndBuff)[0][0]].reshape(-1,bufferLocs[0])[1:]
    #fp = fp[:np.where(fp==EEndBuff)[0][0]].reshape(-1,bufferLocs[1]-bufferLocs[0])[1:]
    #fp = fp[:np.where(fp==EEndBuff)[0][0]].reshape(-1,int(fileInfo['fRecLen'][0]*df.recSize/4))[1:]
    #fp = fp[:np.where(fp==EEndBuff)[0][0]].reshape(-1,int(fileInfo['fRecLen'][0]*df.recSize/32))[1:]

##################################################################
# Print the information from the header
##################################################################
    
def printHeaderDetails():
    print(df.recHead)
    print(fileInfo)
    if(df.adcHeadExists):
        print(ADCDetails)
    if(df.scalerHeadExists):
        print(ScalerDetails)
    if(df.moduleHeadExists):
        print(ModuleDetails)
    print(fileTrailer)

    
##################################################################
# Run a function on each event
##################################################################

def runFunction(function,minEvents=0,maxEvents=0):
    
    global eventNo
    eventNo = df.evtCountStart
    # Loop over data buffers
    for i, dataBuffer in enumerate(fp[df.bufferStart:]):
        if(dataBuffer[0]!=df.EDataBuff):
            print(i, 'Bad data format or file end')
            return
        
        # Set event limits
        eventBuffers = np.append([1],np.where(dataBuffer==df.EEndEvent)[0]+1)

        # Loop over events in buffer
        for j, (start,stop) in enumerate(zip(eventBuffers[0:-1],eventBuffers[1:])):
            event = dataBuffer[start:stop-1]            
            if(len(event)==0): break
            #Check if the events aren't synced at the start of the file or go out of sync during
            if(eventNo==-1): eventNo = event[0]
            if(event[0]!=eventNo): print('Events out of sync', event[0], eventNo)
            eventNo += 1
            if(processEvent(event)):
                if (eventNo>=minEvents): function()
            if (eventNo>=maxEvents and maxEvents): return

           
##################################################################
# Run a function on only epics buffers
##################################################################

def runEPICSFunction(function,minEvents=0,maxEvents=0):
    global epicsBuffers
    global epicsInfo
    
    indeces = np.where(fp==df.EEPICSBuffer)[0]
    
    # Iterate over epicsBuffers
    for i, epicsBufferStart in enumerate(indeces):
        if( i<minEvents or (i>maxEvents and maxEvents) ): continue
        epicsBuffer  = fp[epicsBufferStart+1:]
        epicsInfo    = np.frombuffer(epicsBuffer, dtype=df.epicsHead, count=1)    #get the information from the header
        #epicsEnd     = epicsInfo[0]['len']/4+2 # Hard coded length while we sort out the short/int issue
        print(epicsInfo)
        epicsEnd     = 120264
        epicsBuffers = [epicsBuffer[:epicsEnd]]
        print(epicsBuffers)
        
        function()

        
##################################################################
# Run a function on only scaler buffers
##################################################################
def runScalerFunction(function):
    global scalerBuffers

    scalerLocations = np.where(fp==EScalerBuffer)[0]
        
    if(len(scalerLocations)):
        scalerLocations = scalerLocations.reshape((2,-1),order='F')
        for indeces in scalerLocations:
            scalerBuffers += [eventArray[indeces[0]:indeces[1]]]
            function()

            
##################################################################
# Get the data for a the next event
##################################################################
def processEvent(eventData):
    
    global epicsBuffers
    global scalerBuffers
    global adcArray
    global streamEventNo
    global epicsInfo
    global eventInfo
    global epicsEvent
    global scalerEvent

    eventInfo  = np.frombuffer(eventData,dtype=df.eventHead,count=1)
    byteOffset = eventInfo.itemsize-1 #-1 added for Glasgow TimePix test
        
    eventArray = eventData[int(byteOffset/4):] 
    
    epicsEvent=0
    # Get Epics Data in event
    if(df.EPICSExist):
        epicsBuffers, epicsIndices, epicsInfo = df.FillEPICSArray(eventArray)
        eventArray = np.delete(eventArray,epicsIndices)
        if(len(epicsBuffers)):
            epicsEvent=1

    scalerEvent=0
    #Separate scaler data out
    if(df.ScalersExist):
        scalerBuffers, scalerIndices = df.FillScalerArray(eventArray)
        #if(len(scalerBuffers)):
        eventArray = np.delete(eventArray,scalerIndices)
        if(len(scalerBuffers)):
            scalerEvent=1


    # Check for errors
    errorIndices = df.CheckErrors(eventArray)
    eventArray   = np.delete(eventArray,errorIndices)
    if len(errorIndices):
        #print(eventArray.view(np.uint16).reshape(-1,2)
        return 0
    
    #adcArray   = eventArray.view(np.uint16).reshape(-1,2)
    adcArray   = eventArray.view(dtype=[('adc', np.int16), ('val', np.uint16)])
    #print(adcArray)
    
    return 1

'''
##################################################################
# Print the information about an epics buffer
##################################################################
def dumpEpicsBuffer(buffNo=0):
    
    eInfo = np.frombuffer(epicsBuffers[buffNo], dtype=epicsHead, count=1)[0]    #get the information from the header
    print(eInfo)
    time= eInfo['time']
    print(datetime.datetime.fromtimestamp(time))
          
    index = eInfo.itemsize+2
    for pv in range(eInfo['nchan']):
        pvInfo  = np.frombuffer(epicsBuffers[buffNo], dtype=epicsChan,count=1,offset=index)[0]
        pvName  = pvInfo['pvname']
        eType   = pvInfo['type']
        nElem   = pvInfo['nelem']
        nBytes  = pvInfo['bytes']
        print(eType)
        pvStart = index+pvInfo.itemsize
        pvData  = np.frombuffer(epicsBuffers[buffNo], dtype=epicsTypes[eType], count=nElem,offset=pvStart)
        print(pvInfo)
        print(pvName)
        print(pvData)
        index += nBytes

'''
##################################################################
# Return epics buffer channel as an array either by number or name
##################################################################
def getEpicsPV(pv,buffNo=0):    
    eInfo = np.frombuffer(epicsBuffers[buffNo], dtype=df.epicsHead, count=1)[0]    #get the information from the header
    if(isinstance(pv, int) and eInfo['nchan']<pv):
        print('channel number',pv,'does not exist')
        return 0
        
    index = eInfo.itemsize+2
    for n in range(eInfo['nchan']):
        pvInfo  = np.frombuffer(epicsBuffers[buffNo], dtype=df.epicsChan,count=1,offset=index)[0]
        pvName  = pvInfo['pvname']
        nBytes  = pvInfo['bytes']
        
        if((pv == pvName) or (isinstance(pv, int) and n==pv)):  
            eType   = pvInfo['type']       
            nElem   = pvInfo['nelem']
            pvStart = index+pvInfo.itemsize
            pvData  = np.frombuffer(epicsBuffers[buffNo], dtype=df.epicsTypes[eType], count=nElem,offset=pvStart)
            #print(pvData)
            if(nElem==1):
                return pvData[0]
            else:
                return pvData
        index += nBytes

'''
##################################################################
# Prints all of the epics PVs in the file
##################################################################
def listEpicsPVs():
    
    global epicsPVs
    
    indeces = np.where(fp==EEPICSBuffer)[0]
    
    # Iterate over epicsBuffers
    for i, epicsBufferStart in enumerate(indeces):
        epicsBuffer  = fp[epicsBufferStart+1:]
        eInfo        = np.frombuffer(epicsBuffer, dtype=epicsHead, count=1)    #get the information from the header

        index = eInfo.itemsize+2
        for n in range(eInfo['nchan']):
            pvInfo  = np.frombuffer(epicsBuffer, dtype=epicsChan,count=1,offset=index)[0]
            pvName  = pvInfo['pvname']
            nBytes  = pvInfo['bytes']        
            index += nBytes
            if not pvName in epicsPVs: epicsPVs.append(pvName)

    print(epicsPVs)
    return epicsPVs
'''
        
'''
def getEvent():

    global eventCount
    global buffIndex
    global buffLen
    global buffEnd
    global ADC
    global ScalerCurr
    global ScalerAcc

    global IsScalerEvent
    global ScalerOffset
    global NScaler
    global ScalerLen
    
    global IsEpicsEvent
    global EpicsOffset
    global EpicsLen

    global evEnd
    
    IsScalerEvent = 0           #init counters and flags
    ScalerOffset  = 0
    NScaler       = 0
    ScalerLen     = 0
    
    IsEpicsEvent  = 0
    EpicsOffset   = 0
    EpicsLen      = 0


    ADC.fill(ENullADC)          #set all ADCs to NULL
    ScalerCurr.fill(0)          #set all ADCs to NULL
    
    if(eventCount < 0):         #if this is -ve it swa the last event in the file
        return eventCount
        
    eventInfo = np.frombuffer(fp,dtype=eventHead,count=1,offset=buffIndex*2)
    evEnd = buffIndex+eventInfo[0]['evLen']/2
    #print("eventInfo", eventInfo)
    #print(eventCount)
    buffIndex += 6
    #    print("eventInfo", eventInfo)

    while (buffIndex < evEnd):
        #print("gen", buffIndex,hex(buffIndex),hex(fp[buffIndex]),hex(fp[buffIndex+1]),hex(fp[buffIndex+2]),hex(fp[buffIndex+3]))
        #print("gen", buffIndex,buffIndex,fp[buffIndex],fp[buffIndex+1],fp[buffIndex+2],fp[buffIndex+3])
        
        if(fp[buffIndex]==EEndEvent):
             if(fp[buffIndex+1]==EEndEvent):
                 buffIndex+=2
                 break

        elif(fp[buffIndex]==EEPICSBuffer):
            if(fp[buffIndex+1]==EEPICSBuffer):
                buffIndex+=2                                                                #skip past the marker
                EpicsOffset=buffIndex*2                                                     #save the index of the EPICS buffer
                epicsInfo=np.frombuffer(fp, dtype=epicsHead, count=1,offset=buffIndex*2)    #get the information from the header
                print(epicsInfo)
                print("eventInfo", eventInfo)
                print(eventCount)
                EpicsLen=epicsInfo[0]['len']
                IsEpicsEvent=1
                buffIndex+=EpicsLen/2
            
        elif(fp[buffIndex]==EScalerBuffer):
            if(fp[buffIndex+1]==EScalerBuffer):
                buffIndex+=2                                                                    #skip past the marker
                scalerInfo   = np.frombuffer(fp, dtype='uint32', count=1,offset=buffIndex*2)    #get the information from the header
                buffIndex+=2                                                                    #skip past the marker
                print("scalerInfo", hex(buffIndex*2),scalerInfo)
                print("eventInfo", eventInfo)
                print(eventCount)
                while((fp[buffIndex]!=EScalerBuffer) and (fp[buffIndex+1]!=EScalerBuffer)):
                    scalerData = np.frombuffer(fp, dtype=scalerBuff, count=1,offset=buffIndex*2)
                    ScalerCurr[scalerData[0]['index']]=scalerData[0]['datum']
                    #print(buffIndex*2,hex(scalerData[0]['index']), hex(scalerData[0]['datum']))
                    buffIndex+=4
                    IsScalerEvent=1
                buffIndex+=2

        elif(fp[buffIndex]==EReadError):
            if(fp[buffIndex+1]==EReadError):
                print("eek")
                        
        else:
            ADC[fp[buffIndex]]=fp[buffIndex+1]
            #print("ADC", fp[buffIndex],fp[buffIndex+1])
            buffIndex += 2
        
    eventCount+=1
    #print("evEnded", buffIndex,buffEnd, eventCount)
    #print("bufferEnd",buffIndex,fp[buffIndex],fp[buffIndex+1],fp[buffIndex+2],fp[buffIndex+3])
    buffIndex+=4
    #print("bufferEnd",buffIndex,buffIndex,fp[buffIndex],fp[buffIndex+1],fp[buffIndex+2],fp[buffIndex+3])
    if(buffIndex>=buffEnd):
        print("eventInfo", eventInfo)
        #print("bufferEnd",fp.size,buffIndex,buffEnd,fp[buffIndex],fp[buffIndex+1],fp[buffIndex+2],fp[buffIndex+3])
        
        nextBuffer()
        #buffIndex+=4
    return eventCount



        '''


