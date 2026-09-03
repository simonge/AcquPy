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
    
    scalerLocations = np.where(dataArray==EScalerBuffer)[0]
    if not len(scalerLocations):
        return [], []
    if len(scalerLocations) != len(NScalerBlock):
        print('Bad scaler block')
        return [], []

    block_sizes = np.asarray(NScalerBlock, dtype=np.intp)
    # Fully vectorised ragged range: replaces Python for-loop list builds
    bases   = np.repeat(scalerLocations + 2, block_sizes)
    cum     = np.concatenate([[0], np.cumsum(block_sizes[:-1])])
    offsets = np.arange(NScaler, dtype=np.intp) - np.repeat(cum, block_sizes)
    scalerIndices = bases + offsets

    scalerHeaders = np.ravel(np.column_stack([scalerLocations, scalerLocations + 1]))
    scalerArray = np.column_stack((np.arange(NScaler), dataArray[scalerIndices]))
    return scalerArray, np.concatenate([scalerIndices, scalerHeaders])

def CheckErrors(dataArray):
    errorMarks = np.where(dataArray==EReadError)[0]
    if not len(errorMarks):
        return []
    # Vectorised: broadcast each error mark across offsets 0..3
    return (errorMarks[:, np.newaxis] + np.arange(4, dtype=np.intp)).ravel()
