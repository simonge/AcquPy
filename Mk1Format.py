#!/usr/bin/python
import numpy as np

name = 'Mk1 Data'

ByteOffset = 4

EDataBuff     = 0x20202020  # standard data buffer
EEndEvent     = 0xFFFFFFFF  # end of event marker
EBufferEnd    = 0xFFFFFFFFFFFFFFFF  # end of buffer marker x2
EScalerBuffer = 0xFEFEFEFE  # start/end of scaler read out
EReadError    = 0xEFEFEFEF  # start of error block (hardware error)
EEPICSBuffer  = 0xFDFDFDFD  # start of EPICS read out
EPhysBuff     = 0x50505050  # reserved
EHeadPhysBuff = 0x60606060  # reserved

adcHeadExists    = 1
scalerHeadExists = 1
moduleHeadExists = 1
EPICSExist       = 0
ScalersExist     = 1

# The python versions of structs to deal with acqu buffer headers and data
recHead    = np.dtype( [ ('fTime',            '|S26' ),     # run start time (ascii)     
                         ('fDescription',     '|S133'),     # description of experiment         
                         ('fRunNote',         '|S133'),     # particular run note               
                         ('fOutFile',         '|S40' ),     # output file       
                         ('fRun',             '<u2'  ),     # run number                        
                         ('fNSlaveVME',       '<u2'  ),     # no. slave VMEs
                         ('fNModule',         '<u2'  ),     # total no. modules                 
                         ('fNVME',            '<u2'  ),     # no. VME modules                 
                         ('fNCAMAC',          '<u2'  ),     # no. CAMAC modules                 
                         ('fNFASTBUS',        '<u2'  ),     # no. FASTBUS modules                 
                         ('fNADC',            '<u2'  ),     # total no. ADCs                   
                         ('fNScaler',         '<u2'  ),     # total no. scalers                
                         ('fNCAMAC-ADC',      '<u2'  ),     # total no. CAMAC ADCs                
                         ('fNCAMAC-Scalers',  '<u2'  ),     # total no. CAMAC scalers          
                         ('fNFASTBUS-ADC',    '<u2'  ),     # total no. CAMAC ADCs                
                         ('fNFASTBUS-Scalers','<u2'  ),     # total no. CAMAC scalers
                         ('fRecLen',          '<u2'  ) ] )  # maximum buffer length = record len

recSize    = 1
evtCountStart = -1
bufferStart = 1

recTrailer = np.dtype( [ ('fNBuffers',        '<u2'  ),     # Number of buffers read
                         ('fTime',            '|S20'  ) ] )  # End Time
                    
adcHead    = np.dtype( [ ('fIndex',           '<i2'  ),     # ADC Module Index
                         ('fSubAddress',      '<u2'  ) ] )  # ADC Module SubAddress

scalerHead = np.dtype( [ ('fIndex',           '<u2'  ),     # Scaler Module Index
                         ('fSubAddress',      '<u2'  ) ] )  # Scaler Module SubAddress

moduleHead = np.dtype( [ ('fName',            '|S20' ),     # Module name
                         ('fVMECrateNo',      '<i2'  ),     # VME Crate Number
                         ('fBusType',         '<i2'  ),     # Bus Type
                         ('fModuleType',      '<i2'  ),     # Module Type
                         ('fBranchAddress',   '<i2'  ),     # Branch Address
                         ('fCrateAddress',    '<i2'  ),     # Crate Address
                         ('fStationAddress',  '<i2'  ),     # Station Address
                         ('fMinSubAddress',   '<i2'  ),     # Minimum sub address
                         ('fMaxSubAddress',   '<i2'  ),     # Maximum sub address
                         ('fMaxNBits',        '<i2'  ) ] )  # Max number of bits

eventHead  = np.dtype( [ ('evNo',             '<u4'  ) ] )  # Event number

readError     = np.dtype ( [ ('fHeader', '<u4'),        #error block header
                             ('ModID', '<u2'),          #hardware identifier
                             ('ModID2', '<u2'),          #hardware identifier
                             ('ModIndex', '<u2'),       #list index of module 
                             ('ModIndex2', '<u2'),       #list index of module 
                             ('ErrCode', '<u2'),        #error code returned
                             ('ErrCode2', '<u2') ] )     #error code returned

moduleSeparator = '/cbd0'

def MakeScalerArray(moduleList,scalerList):
    global scalerPositions
    global NScaler
    global NScalerBlock
    NScalerBlock = []
    cbdList = np.where(moduleList['fName']==moduleSeparator)
    NScaler = len(scalerList)
    scalerPositions = np.arange(len(scalerList))
    scalerPositions = np.split(scalerPositions,scalerList['fIndex'].searchsorted(cbdList[0][1:]))
    for block in scalerPositions:
        NScalerBlock += [len(block)]

def FillScalerArray(dataArray):
    
    #np.set_printoptions(threshold=np.nan)
    scalerHeaders = []
    scalerIndices = []
    scalerLocations = np.where(dataArray==EScalerBuffer)[0]
    if(len(scalerLocations)):
        if(len(scalerLocations)!=len(NScalerBlock)):
            print('Bad scaler block')
            return [], []
        for i, index in enumerate(scalerLocations):
            scalerHeaders += [index,index+1]
            scalerIndices += range(index+2,index+2+NScalerBlock[i])
        scalerArray = np.column_stack((np.arange(NScaler),np.take(dataArray,scalerIndices)))
        dataArray
        return scalerArray, scalerIndices+scalerHeaders
    return [], []
    
def CheckErrors(dataArray):
    errorIndices = []
    for errorMark in np.where(dataArray==EReadError)[0]:
        #print errorMark
        #print np.frombuffer(dataArray[errorMark:], dtype=readError, count=1)
        errorIndices += range(errorMark,errorMark+4)
    return errorIndices
