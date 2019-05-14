#!/usr/bin/python
import numpy as np

name = 'Mk2 Data'

ByteOffset = 8

EDataBuff     = 0x70707070  # data buffer
EEndEvent     = 0xFFFFFFFF  # end of event marker
EBufferEnd    = 0xFFFFFFFF  # end of buffer marker x2
EScalerBuffer = 0xFEFEFEFE  # start/end of scaler read out
EReadError    = 0xEFEFEFEF  # start of error block (hardware error)
EEPICSBuffer  = 0xFDFDFDFD  # start of EPICS read out
EPhysBuff     = 0x50505050  # reserved
EHeadPhysBuff = 0x60606060  # reserved

adcHeadExists    = 0
scalerHeadExists = 0
moduleHeadExists = 1
EPICSExist       = 1
ScalersExist     = 1

# The python versions of structs to deal with acqu buffer headers and data
recHead    = np.dtype( [ ('fTime',         '|S32' ),     # run start time (ascii)     
                         ('fDescription',  '|S256'),     # description of experiment         
                         ('fRunNote',      '|S256'),     # particular run note               
                         ('fOutFile',      '|S128'),     # output file       
                         ('fRun',          '<u4'  ),     # run number                        
                         ('fNModule',      '<u4'  ),     # total no. modules                 
                         ('fNADCModule',   '<u4'  ),     # no. ADC modules                   
                         ('fNScalerModule','<u4'  ),     # no. scaler modules                
                         ('fNADC',         '<u4'  ),     # no. ADC's read out                
                         ('fNScaler',      '<u4'  ),     # no. scalers readout               
                         ('fRecLen',       '<u4'  ) ] )  # maximum buffer length = record len

recSize    = 8
evtCountStart = 0
bufferStart = 0

recTrailer = np.dtype( [ ('fNBuffers',        '<u4'   ),     # Number of buffers read
                         ('fTime',            '|S20'  ) ] )  # End Time

moduleHead = np.dtype( [ ('fID',              '<i4'  ),     # Acqu ID
                         ('fIndex',           '<i4'  ),     # Index
                         ('fModuleType',      '<i4'  ),     # Module Type
                         ('fMinChannel',      '<i4'  ),     # Minimum channel
                         ('fNChannel',        '<i4'  ),     # Number of channels
                         ('fNScalerChannels', '<i4'  ),     # Number of scaler channels
                         ('fNBits',           '<i4'  ) ] )  # significant bits from output word

eventHead = np.dtype( [ ('evNo',   '<u4'),
                        ('evLen',  '<u4'),
                        ('adcInd', '<u2'),
                        ('adcCnt', '<u2')])

#                 BYTE    STRING  SHORT    LONG     FLOAT      DOUBLE
epicsTypes    = ['int8', '<S40', 'int16', 'int64', 'float32', 'float64']

epicsHead = np.dtype ( [ ('epics', '|S32'),
                         ('time',  '<u4' ),
                         ('index', '<u2' ),
                         ('period','<u2' ),
                         ('id',    '<u2' ),
                         ('nchan', '<u2' ),
                         ('len',   '<u2' ) ])

epicsChan = np.dtype ( [ ('pvname', '|S32'),        #Process variable name
                         ('bytes',   '<u2'),        #No of bytes for this channel
                         ('nelem',   '<u2'),        #No of elements in array
                         ('type',    '<u2') ])      #No of bytes for this channel
                         #('dummy',   '<u2') ])      #Type of element
                         

readError     = np.dtype ( [ ('fHeader', '<u4'),        #error block header
                             ('ModID', '<u4'),          #hardware identifier
                             ('ModIndex', '<u4'),       #list index of module 
                             ('ErrCode', '<u4'),        #error code returned
                             ('fTrailer', '<u4') ])     #end of error block marker

def FillScalerArray(dataArray):
    
    scalerIndices = []
    scalerHeaders = []
    scalerLocations = np.where(dataArray==EScalerBuffer)[0]
    if(len(scalerLocations)):        
        scalerLocations = scalerLocations.reshape((-1,2),order='C')
        for indeces in scalerLocations:
            scalerHeaders += [indeces[0],indeces[0]+1]
            scalerIndices += range(indeces[0]+2,indeces[1])
        scalerArray = np.take(dataArray,scalerIndices).reshape((-1,2))
        return scalerArray, scalerIndices+scalerHeaders
    return [], []

def FillEPICSArray(dataArray):
        
    epicsIndices = []
    epicsBuffers = []
    epicsInfo    = []
    for epicsBufferStart in np.where(dataArray==EEPICSBuffer)[0]:
        epicsBuffer  = dataArray[epicsBufferStart+1:]
        epicsInfo    = np.frombuffer(epicsBuffer, dtype=epicsHead, count=1)    #get the information from the header
        #epicsEnd     = epicsInfo[0]['len']/4+2 # Hard coded length while we sort out the short/int issue
        epicsEnd     = 120264
        epicsBuffer  = epicsBuffer[:epicsEnd]
        epicsBuffers += [epicsBuffer]
        epicsIndices += range(epicsBufferStart,epicsBufferStart+epicsEnd)
    return epicsBuffers, epicsIndices, epicsInfo

def CheckErrors(dataArray):
    errorIndices = []
    for errorMark in np.where(dataArray==EReadError)[0]:
        #print errorMark
        #print np.frombuffer(dataArray[errorMark:], dtype=readError, count=1)
        errorIndices += range(errorMark,errorMark+5)
    return errorIndices
