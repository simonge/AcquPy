import numpy as np
import pandas as pd
import igraph as ig
import plotly.graph_objs as go
from pandas.io.json import json_normalize
import json

pd.set_option('display.max_rows', 10)

##################################################################
# Open detector file and load detectors
##################################################################
def LoadDetectorsFile(FileArray):
    with open(FileArray) as f:
        LoadDetectors(f.read().splitlines())

##################################################################
# Load detectors from files in array
##################################################################
def LoadDetectors(jsonFileArray):
    global detmask
    global detignore
    global detequ
    global detref
    global detectors
    global Channels
    global detgraphs
    global detparams
    detmask   = {}
    detignore = {}
    detequ    = {}
    detref    = {}
    detparams = {}
    detectors = []
    Channels  = {}
    detgraphs = {}
    for detNo, detectorFile in enumerate(jsonFileArray):
        print(detectorFile)
        infile    =  open(detectorFile)
        data      =  json.load(infile)
        detName   =  data['detector'].encode('ascii')
        detectors += [detName]
        detparams[detName] = [dat.encode('ascii') for dat in data['parameters']]
        detequ[detName]    = [DefineFunction(equ) for equ in data['calEq']]
        detref[detName]    = data['references']

        # Create dataframe of channels
        jsonArray          = json_normalize(data,'channels')
        detmask[detName]   = np.array(jsonArray.adc.values.tolist())
        Channels[detName]  = len(jsonArray)
        detignore[detName] = np.array(range(Channels[detName]-len(data['ignore'])))
        if 'startChan' in data.keys():
            detignore[detName] = np.insert(detignore[detName],range(data['startChan']),-1)
        for index in data['ignore']:
            detignore[detName] = np.insert(detignore[detName],index,-1)
        detgraphs[detName] = MakeGraph(jsonArray.to_records())
        if 'ignore' in data.keys():
            detgraphs[detName].delete_vertices(data['ignore'])
            detmask[detName] = np.delete(detmask[detName],data['ignore'],0)

    
##################################################################
# Define calibration function
##################################################################
def DefineFunction(funcString):
    funcString = 'def func(parameter,x,ref,i): x[parameter] = ' + funcString
    exec(funcString)
    return func

##################################################################
# Constructs graph object for each detector
##################################################################
def MakeGraph(array):
    
    matrix = np.identity(len(array),np.bool)
    for channel in array:
        matrix[channel['channel']][channel['neighbours']] = True
    graph = ig.Graph.Adjacency(matrix.tolist(),mode=1)
    for name in array.dtype.names:
        graph.vs[name] = np.array(array[name].tolist())
        
    return graph

##################################################################
# Separate the detector values and apply calibration equation
##################################################################
def Calibrate(adcArray,detlist=[]):
    global Graphs
    global Arrays

    # If no detector list provided calibrate all
    if not len(detlist):
        detlist = detectors

    adcArray.sort()
    
    Graphs = {}
    Arrays = {}
    
    for detector in detlist:

        subgraph = {}

        for i, equation, param in zip(range(len(detequ[detector])),detequ[detector],detparams[detector]):
            
            sort = np.argsort(detmask[detector][:,i])
            filt = np.searchsorted(detmask[detector][:,i],adcArray['adc'],sorter=sort)

            
            filt[filt==len(detmask[detector])] = 0
            filt[detmask[detector][sort][filt,i] != adcArray['adc']] = -1
                        
            rawValues               = CollapseArray(adcArray[(filt+1).astype(np.bool)])
            subgraph[param]         = ig.VertexSeq(detgraphs[detector],np.unique(sort[filt[filt!=-1]]))
            subgraph[param]['raw']  = rawValues
            
            #Get reference values
            refs = []
            if(len(detref[detector][i])):
                filt = np.isin(adcArray['adc'],detref[detector][i])
                refs = adcArray['val'][filt]
                ####BAD HACK####
                if(len(refs)!= len(detref[detector][i])):
                    refs = np.zeros(len(detref[detector][i]))

            equation(param,subgraph[param],refs,i)

        Arrays[detector] = subgraph
        
##################################################################
# Put values with shared adc in the same array
##################################################################
def CollapseArray(array):
    sumIndeces  = np.nonzero(np.diff(array['adc']))[0]+1
    valueCol    = np.split(array['val'],sumIndeces)
    return valueCol

##################################################################
# Get defined graph by the union of vertex arrays eg. time and energy
##################################################################
def GetGraph(detector,paramlist=[],union=1):
    
    # If no parameter list given use all
    if not len(paramlist):
        paramlist = detparams[detector]
    
    array = []
    for param in paramlist:
        array += Arrays[detector][param].indices
    array = np.array(array)
    if union:
        unique, counts = np.unique(array, return_counts=True)
        array = unique[counts==len(paramlist)]
    return detgraphs[detector].subgraph(array)

##################################################################
# Calibrate MC data, putting values in graphs
##################################################################
def MCCalibrate(mcArray,particles=[],EnergyThresholds=3):
    global MCGraph
    global MCArray

    # Select hits from the particle list
    if(len(particles)):        
        mcArray = {key: value[np.isin(value['partID'],particles)] for key, value in mcArray.iteritems() }
        
    MCGraph = {}
    MCArray = {}
    
    for detector in detectors:
        
        if(not len(mcArray[detector])):
            MCGraph[detector] = detgraphs[detector].subgraph([])
            MCArray[detector] = MCGraph[detector].vs
            continue      

        # Sum any values in the same crystal and sort by channel
        #print mcArray[detector]
        data = sumCols(mcArray[detector])
        #print data
        MCGraph[detector] = detgraphs[detector]
        
        energies = np.zeros(MCGraph[detector].vcount())
        times    = np.zeros(MCGraph[detector].vcount())
        energies[detignore[detector][data['channel']]] = data['Energy']
        times[detignore[detector][data['channel']]]    = data['Time']
        
        MCGraph[detector].vs['Energy']  = energies
        MCGraph[detector].vs['Time']    = times
        if(EnergyThresholds!=0):
            MCGraph[detector] = MCGraph[detector].subgraph(MCGraph[detector].vs.select(Energy_ge=EnergyThresholds))     
        
        MCArray[detector] = MCGraph[detector].vs
        

##################################################################
# Sums values in MC data which share the same crystal
##################################################################
def sumCols(array):    
    sortOrder   = np.argsort(array.channel)
    sumIndeces  = np.nonzero(np.diff(array.channel[sortOrder]))[0]+1
    sumIndeces  = np.insert(sumIndeces,0,0)
    channelCol  = array.channel[sortOrder][sumIndeces]
    summedCol   = np.add.reduceat(array.Energy[sortOrder], sumIndeces)
    minimumCol  = np.minimum.reduceat(array.Time[sortOrder], sumIndeces)
    SortedArray = np.core.records.fromarrays([channelCol,summedCol,minimumCol],names='channel,Energy,Time')    
    return SortedArray
    
graphLabels = [('Time','ns'),('Energy','MeV')]

##################################################################
# Create a 3D plot of a graph
##################################################################
def GraphPlot(graph,gtype=1):
       
    xN = [x[0] for x in graph.vs['position']]
    yN = [x[1] for x in graph.vs['position']]
    zN = [x[2] for x in graph.vs['position']]
    tN = []
    #if label in graph.vs.attribute_names():
    #    tN = [str(text)+' ns' for text in graph.vs[label]]
    #tN = [str(chan) for chan in graph.vs['channel']]

    for vertex in graph.vs:
        text = ''
        for label in graphLabels:
            if label[0] in graph.vs.attribute_names():
                text += str(vertex[label[0]]) + " " + label[1] + " "
        tN += [text]
        #tN = [str(Ttext)+' ns, '+str(Etext)+' MeV' for Ttext, Etext in zip(graph.vs['Time'],graph.vs['Energy'])]
        #tN = [str(Ttext)+' ns' for Ttext in graph.vs['Time']]

    Xe=[]
    Ye=[]
    Ze=[]
    for e in graph.es:
        Xe+=[xN[e.source],xN[e.target], None]# x-coordinates of edge ends
        Ye+=[yN[e.source],yN[e.target], None]# x-coordinates of edge ends
        Ze+=[zN[e.source],zN[e.target], None]# x-coordinates of edge ends

    if(gtype==1):
        edgewidth = 2
        markersize = 4
        markerstyle=dict(symbol='circle',
                         size=markersize,
                         #color = graph.vs['Time'],
                         #colorscale='Viridis',
                         #showscale=True
                         line = dict(
                             #color = graph.vs['Energy'],
                             #colorscale='Viridis',
                             width=1                             
                             )
                         )
        hoverType='text'
    else:
        edgewidth = 0.5
        markersize = 1
        markerstyle = dict(symbol='circle',
                           size=markersize,
                           line=dict(color='rgb(50,50,50)', width=0.5)
                           )
        hoverType='none'

        
    edges = go.Scatter3d(
        x=Xe,
        y=Ye,
        z=Ze,
        mode='lines',
        line=dict(color='rgb(125,125,125)', width=edgewidth),
        hoverinfo='none'
    )



    nodes = go.Scatter3d(
        x=xN,y=yN,z=zN,
        mode='markers',
        marker=markerstyle,
        text=tN,
        hoverinfo=hoverType
    )

    return [edges, nodes]

vertexZ = 0

##################################################################
# Create a 3D plot of the MC inputs
##################################################################
def InputPlot(inputs,type=0,particles=[]):
    global vertexZ

    if(not len(particles)):
        particles = range(inputs)        
    
    xN = [inputs['vertex'][0]]
    yN = [inputs['vertex'][1]]
    vertexZ = inputs['vertex'][2]
    zN = [vertexZ]
    
    #zN = [14]
    
    Xe=[]
    Ye=[]
    Ze=[]

    theta = []
    phi   = []
    
    distance = 200
    
    Te=['vertex '+ str(inputs['beam'][4]*1000) + ' MeV']
    
    for e in inputs['dircos'][particles]:
        xN+=[xN[0]+e[0]*distance]
        yN+=[yN[0]+e[1]*distance]
        zN+=[zN[0]+e[2]*distance]
        Xe+=[xN[0],xN[0]+e[0]*distance, None]# x-coordinates of edge ends
        Ye+=[yN[0],yN[0]+e[1]*distance, None]# x-coordinates of edge ends
        Ze+=[zN[0],zN[0]+e[2]*distance, None]# x-coordinates of edge ends
        theta += [np.rad2deg(np.arccos(e[2]))]
        phi   += [np.rad2deg(np.arctan2(e[0],e[1]))]
    
    Te += ['Particle: '+str(i)+' ID: '+str(t[0]) + ' Theta: '+str(t[1]) + ' Phi: '+str(t[2])  + ' Energy: '+str(t[3]) for i, t in enumerate(zip(inputs['idpart'][particles],theta,phi,1000*inputs['elab'][particles]))]
    #Te=['vertex '+ str(inputs['beam'][4]*1000) + ' MeV']+['Particle: '+str(i)+' ID: '+str(t[0]) for i, t in enumerate(inputs['idpart'])]

    nodes = go.Scatter3d(
        x=xN,y=yN,z=zN,
        mode='markers',
        marker=dict(symbol='cross',
            size=10,
            colorscale='Viridis',
            line=dict(color='rgb(50,50,50)', width=0.5)
        ),
        text=Te,
        hoverinfo='text'
    )
    
    edges = go.Scatter3d(
        x=Xe,
        y=Ye,
        z=Ze,
        mode='lines',
        line=dict(color='rgb(125,125,125)', width=3),
        hoverinfo='none',
    )
    return nodes, edges

##################################################################
# Create a 3D plot of the MC inputs
##################################################################
def OutputPlot(outputs,type=0):

    tracks   = outputs['position']
    energies = outputs['energy']
    
    xN = [0]
    yN = [0]
    #zN = [inputs['vertex'][2]]
    zN = [vertexZ]
    
    Xe=[]
    Ye=[]
    Ze=[]

    theta = []
    phi   = []
        
    for e in tracks:
        xN+=[e[0]]
        yN+=[e[1]]
        zN+=[e[2]]
        Xe+=[xN[0],e[0], None]# x-coordinates of edge ends
        Ye+=[yN[0],e[1], None]# x-coordinates of edge ends
        Ze+=[zN[0],e[2], None]# x-coordinates of edge ends
        theta += [np.rad2deg(np.arccos(e[2]))]
        phi   += [np.rad2deg(np.arctan2(e[0],e[1]))]
    
    Te=['0']+['Track: '+str(i) + ' Theta: '+str(t[0]) + ' Phi: '+str(t[1])  + ' Energy: '+str(t[2]) for i, t in enumerate(zip(theta,phi,energies))]
    #Te=['vertex '+ str(inputs['beam'][4]*1000) + ' MeV']+['Particle: '+str(i)+' ID: '+str(t[0]) for i, t in enumerate(inputs['idpart'])]

    nodes = go.Scatter3d(
        x=xN,y=yN,z=zN,
        mode='markers',
        marker=dict(symbol='circle-open',
            size=10,
            colorscale='Viridis',
            line=dict(color='rgb(50,50,50)', width=0.5)
        ),
        text=Te,
        hoverinfo='text'
    )
    
    edges = go.Scatter3d(
        x=Xe,
        y=Ye,
        z=Ze,
        mode='lines',
        line=dict(color='rgb(125,125,125)', width=3),
        hoverinfo='none',
    )
    return nodes, edges


tagTimeOffset = 0
#tagTimeOffset = 4050
tagTimeScale  = 0.1
paddleADC     = 1055

#Change the tagger ADC values to channels and times
#def TaggerChannels(adcArray):
#    tagFilter = tagger.GetValues(adcArray)
#    return tagFilter


#TRIGGER ADC Time
#LongADC  = 301
#ShortADC = 300
