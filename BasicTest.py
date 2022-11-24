#!/usr/bin/python3
import numpy as np
import Acqu as aq
import AcquDetector as aqdet
import argparse 
import time
import ROOT

#canvas      = Canvas()
adchist     = ROOT.TH1F('adcs','adcs',10000,0,10000)
adchist2d   = ROOT.TH2F('adcvalues','adcvalues',500,0,500,68000,0,68000)
taggerchan  = ROOT.TH1F('tagchans','tagchans',366,0,366)

taggerchantime  = ROOT.TH2F("taggerchantime","taggerchantime",2000,-1000,1000,366,0,366)
cbchanenergy    = ROOT.TH2F("cbchanenergy","cbchanenergy",100,0,1200,720,0,720)
cbchantime      = ROOT.TH2F("cbchantime","cbchantime",2000,-1000,1000,720,0,720)
tapschanenergy  = ROOT.TH2F("tapschanenergy","tapschanenergy",100,0,1200,384,0,384)
tapschantime    = ROOT.TH2F("tapschantime","tapschantime",2000,-1000,1000,384,0,384)

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
        print(aq.fileInfo)

        #Load JSON files for detectors here
        aqdet.LoadDetectors(['DetectorSettings/taggerNew.json'])
        
        middle = time.time()
        print('Open file: ',middle-start)
        # Run function
        #aq.runFunction(fillADCHist,0,100000)
        aq.runFunction(fillTagger,0,100000)     
        print(aq.eventNo)
        
        end = time.time()
        print('Process:   ',end-middle)

    # Write out histograms
    adchist.Write()
    adchist2d.Write()
    taggerchan.Write()
    taggerchantime.Write()

       
def fillTagger():
    #np.set_printoptions(threshold=np.nan)
    print("HI0")
    aqdet.Calibrate(aq.adcArray)
    print("HI")
    
    number   = np.size(data['tagger'][['channel']])
    channels = data['tagger'][['channel']]
    times    = data['tagger'][['channel']]
    weights  = np.ones(number)
    
    taggerchan.FillN(number,channels,weights)
    taggerchantime.FillN(number,channels,times,weights)
    if(aq.eventNo%1000==0):
        print('number',aq.eventNo)


def fillADCHist():
    global adchist
    global adchist2d
    number  = np.size(aq.adcArray['adc'])
    weights = np.ones(number)
    adcs    = aq.adcArray['adc'].astype(float)
    values  = aq.adcArray['val'].astype(float)
    adchist.FillN(number,adcs,weights)
    adchist2d.FillN(number,adcs,values,weights)    
    #adchist2d.FillN(np.vstack((aq.adcArray['adc'],aq.adcArray['val'])).T)    
    if(aq.eventNo%10000==0):
        print('number',aq.eventNo)

if __name__ == "__main__": main()  # call main comes at the end: a quirk of python
