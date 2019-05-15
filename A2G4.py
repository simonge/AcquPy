import numpy as np
import ROOT
from glob import glob
from root_numpy import tree2array, root2array, list_branches

TAPSList     = ['ictaps','ectapsl','tctaps','pctaps']
CBList       = ['icryst','ecryst','tcryst', 'pcryst']
PIDList      = ['iveto','eveto','tveto', 'pveto']
VETOList     = ['ivtaps','evtaps','tvtaps','pvtaps']
DetectorDict = {'TAPS':TAPSList,'CB':CBList,'PID':PIDList,'VETO':VETOList}
InputList    = np.array(['beam','vertex','dircos', 'plab', 'elab', 'idpart'])

def openFile(fileName):
    global detectorArray
    global inputArray
    detectorArray = root2array(filenames=fileName,treename='h12',branches=TAPSList+CBList)
    inputArray    = root2array(filenames=fileName,treename='h12',branches=InputList)
    
def openFiles(fileNames):
    global array
    infile =  ROOT.TFile(fileName)
    tree   = infile.Get("h12")
    array  = root2array(filenames=fileNames,treename='h12').view(np.recarray)
#def openChain(fileList):
#    global array
#    chain = ROOT.TChain("h12")
#    files = glob(fileList)
#    print files
#    for f in files:
#        chain.AddFile(f)
#    #branches = list_branches(fileList)
#    array  = tree2array(chain).view(np.recarray)


def runFunction(function,minEvents=0,maxEvents=0):
    global eventNo
    global signals
    global inpart
    
    for eventNo, entry, inputEntry in zip(range(len(detectorArray)),detectorArray,inputArray):
        if(eventNo<minEvents): continue
        if(eventNo>maxEvents and maxEvents!=0): break
        
        # Input to the simulation
        inpart  = inputEntry
        # Create numpy arrays from entry lists
        signals = {key: np.core.records.fromarrays(entry[value],names='channel,Energy,Time,partID') for key, value in DetectorDict.iteritems()} 
        # Change GeV values to MeV
        for key, value in signals.iteritems(): value['Energy'] = value['Energy']*1000
        
        function()
