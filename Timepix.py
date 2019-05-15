import numpy as np

#TIMEPIX
tpxType=[('x', np.uint8),('y', np.uint8),('ToA', np.uint32),('FToA', np.uint8),('ToT', np.uint16)]
def Decode(nHits,tpxArray):
    return np.frombuffer(tpxArray, dtype=tpxType, count = int(nHits))

def CalibrateTime(array):
    times = 25*array[['ToA']].astype(float) - 25/16*array[['FToA']].astype(float)
    return times
