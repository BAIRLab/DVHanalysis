import pandas as pd
import numpy as np
from collections import OrderedDict

def find_idx_nearest_val(array, value):
    idx = int(np.searchsorted(-array, -value, side="right"))
    if idx >= len(array):
        idx_nearest = idx-1
    elif idx == 0:
        idx_nearest = idx
    else:
        if abs(value - array[idx-1]) < abs(value - array[idx]):
            idx_nearest = idx-1
        else:
            idx_nearest = idx
    return idx_nearest

def randROI(maxdose, fulldose, dosearray, vol, name):

    arr = np.linspace(0, 56, dosearray.size).round(1)

    full = find_idx_nearest_val(dosearray, fulldose)
    maxd = find_idx_nearest_val(dosearray, maxdose)

    arr[0:full] = 100
    arr[full:maxd] = np.geomspace(100, 0.01, num=maxd-full)
    arr[maxd:] = 0.0

    roi = pd.DataFrame(np.transpose(arr), columns=[name])
    roi.loc['Volume'] = vol
    
    return roi    

def vol_DVH(dvh, absolute=True):
    vol_dvh = dvh.drop(columns=dvh.columns[0])
    if absolute:
        vol_dvh = vol_dvh.iloc[:-6].multiply(vol_dvh.loc['Volume']).divide(100)
    else:
        vol_dvh = vol_dvh.iloc[:-6]
        
    vol_dvh = vol_dvh.astype('float64')
    
    #error checking:
    nans = vol_dvh.isnull().values.any()
    num_nans = vol_dvh.isnull().sum().sum()
    if num_nans > 0:
        print('Generated absolute volume dataframe contains '
              '{} NaN values'.format(num_nans))
    
    return vol_dvh

def add_identifiers(dose_DF, dvh):
    dose_DF['Pt'] = dvh.loc['Pt'][0]
    dose_DF['Modality'] = dvh.loc['Modality'][0]
    dose_DF['Side'] = dvh.loc['Side'][0]
    dose_DF['Loc'] = dvh.loc['Loc'][0]
    dose_DF['Size'] = dvh.loc['Size'][0]
    
    return dose_DF

def roi_check(dvh, roi):
    if roi is not None:
        if type(roi) != list:
            if type(roi) == tuple:
                roi = list(roi)
            else:
                roi = [roi]
        rois = list(dvh.columns[1:])
        if any(x in rois for x in roi):
            drop_cols = [i for i in rois if i not in roi]
            dvh = dvh.drop(columns=drop_cols)
    return dvh

def dose_at_volume(dvh, volume, roi=None, absolute=True):
    #First check if ROI(s) were passed and eliminate unwanted columns
    dvh = roi_check(dvh, roi)

    #Then calculate volume DVH
    vol_dvh = vol_DVH(dvh, absolute)    
    
    #Create Empty DataFrame variable
    dose_DF = pd.DataFrame()
    
    #First the easy cases (volume is in table for all ROIs or vol < 0):
    
    if all([volume in vol_dvh.values[:, x] for x in (range(len(vol_dvh.columns)))]):
        print('In all')
        #Checks if volume passed to fxn is already present in all ROIs
        idxs = []
        for col in vol_dvh.columns:
            idxs.append(vol_dvh.loc[vol_dvh[col] == volume].index[0])
        for idx,col in zip(idxs, vol_dvh.columns):
            vol_idx = vol_dvh.iloc[[idx],[vol_dvh.columns.get_loc(col)]].index[0]
            dose = dvh.iloc[vol_idx]['Dose']
            temp = pd.DataFrame(dose, columns=[col], 
                                index=['Dose to {} cc'.format(volume)])
            dose_DF = pd.concat([dose_DF, temp], axis=1)        
        return add_identifiers(dose_DF, dvh)
    if volume <= 0:
        print('Nonsense')
        #Nonsensical. Return 0 filled dataframe
        dose_DF = pd.DataFrame(0, columns=list(vol_dvh.columns),
                               index=['Dose to {} cc'.format(volume)])
        return add_identifiers(dose_DF, dvh)
    
    #We will need to interpolate for our other cases
    for col in vol_dvh.columns:
        vol_idx = find_idx_nearest_val(vol_dvh[col], volume)
        dose = 'Error'
        noskip = True
        
        if volume >= dvh[col]['Volume'] and noskip:
            #input volume is greater than or equal to organ volume. 
            #Return max dose to whole organ.
            vol_idx = np.where(dvh.iloc[:-6][col] < 100)[0][0] - 1
            dose = dvh['Dose'][vol_idx]
            noskip = False
            
        if (vol_idx == 1 or vol_idx == 0) and vol_dvh[col][1] == 0 and noskip:
            #Organ gets no dose
            dose = 0.0
            noskip = False
            
        if noskip:
            #volume is less than organ volume and organ is recieving dose
            vol2 = None
            if volume > vol_dvh[col][vol_idx] and vol_idx < len(vol_dvh[col]) - 1:
                vol2 = vol_idx+1
            elif volume < vol_dvh[col][vol_idx] and vol_idx != 0:
                vol2 = vol_idx-1
            else:
                vol2 = vol_idx 
            xp = [vol_idx, vol2]
            fp = [dvh['Dose'][vol_idx], dvh['Dose'][vol2]] 
            dose = np.interp(volume, xp, fp)
        
        if dose == 'Error':
            print('Volume: ', volume)
            print('vol_idx: ',vol_idx)
            print('vol_2: ',vol2)
        
        if np.isnan(dose):
            print('You have generated a NaN for your dose!')
        
        roi_DF = pd.DataFrame(dose, columns=[col], 
                              index=['D at {} cc or whole organ'.format(volume)])
        dose_DF = pd.concat([dose_DF, roi_DF], axis=1)
        dose_DF = polish_dvh(dose_DF, dvh)
        
        #error checking:
        if nan_check(dose_DF):
            print('dose_at_volume() generated dataframe contains '
                  '{} NaN values'.format(num_nans(dose_DF)))
            
    return dose_DF

def volume_at_dose(dvh, dose, roi=None, absolute=True):
    #First check if ROI(s) were passed and eliminate unwanted columns
    dvh = roi_check(dvh, roi)
    
    #Then calculate volume DVH
    vol_dvh = vol_DVH(dvh, absolute)
  
    #Create Empty DataFrame variable
    dose_DF = pd.DataFrame()
    
    #First the easy cases (dose is <0, >max, or exactly in table):
    if dose in dvh['Dose'].values:
        idx = dvh.loc[dvh['Dose'] == dose].index[0]
        dose_DF = vol_dvh.iloc[idx, :].to_frame().T
        dose_DF = dose_DF.rename(index={idx: 'V getting at least {} cGy'.format(dose)})
        dose_DF = polish_dvh(dose_DF, dvh)
        
        #error checking:
        if nan_check(dose_DF):
            print('volume_at_dose() generated dataframe contains '
                  '{} NaN values'.format(num_nans(dose_DF)))
        
        return dose_DF.round(1)
    
    if dose > dvh['Dose'].max() or dose < 0:
        volume = 0.0
        dose_DF = vol_dvh.iloc[0, :].to_frame().T
        dose_DF = dose_DF * 0
        dose_DF = dose_DF.rename(index={0: 'V getting at least {} cGy'.format(dose)})
        dose_DF = polish_dvh(dose_DF, dvh)
        
        #error checking:
        if nan_check(dose_DF):
            print('volume_at_dose() generated dataframe contains '
                  '{} NaN values'.format(num_nans(dose_DF)))
        
        return dose_DF.round(1)
        
    #Need to interpolate otherwise
    dose_idx = find_idx_nearest_val(dvh['Dose'], dose)
    xp = None
    fp = None
    greater = None
    volume = None
    
    if dose > dvh['Dose'][dose_idx] and dose_idx < len(vol_dvh.index) - 1:
        xp = [dvh['Dose'][dose_idx], dvh['Dose'][dose_idx+1]]
        greater = True
    elif dose < dvh['Dose'][dose_idx] and dose_idx > 0:
        xp = [dvh['Dose'][dose_idx-1], dvh['Dose'][dose_idx]]
        greater = False
        
    if roi==None:
        for col in vol_dvh.columns:            
            if greater:
                fp = [vol_dvh[col][dose_idx], vol_dvh[col][dose_idx+1]]
                volume = np.interp(dose, xp, fp)            
            elif greater == False:
                fp = [vol_dvh[col][dose_idx], vol_dvh[col][dose_idx+1]]
                volume = np.interp(dose, xp, fp)
            else:
                volume = vol_dvh[col][dose_idx]
            
            if dose > dvh['Dose'].max() or dose < 0:
                volume = 0.0
             
            roi_DF = pd.DataFrame(volume, columns=[col], 
                                  index=['V getting at least {} cGy'.format(dose)])
                
            dose_DF = pd.concat([dose_DF, roi_DF], axis=1)
            
    dose_DF = polish_dvh(dose_DF, dvh)
    
    #error checking:
    if nan_check(dose_DF):
        print('volume_at_dose() generated dataframe contains '
              '{} NaN values'.format(num_nans(dose_DF)))
    
    return dose_DF
                 
def max_dose(dvh, roi=None):
    dvh = roi_check(dvh, roi)
    
    dose_DF = pd.DataFrame()
    dvh1 = dvh.iloc[1:-6, 1:]
    cols = range(len(dvh.columns)-1)
    
    max_idx = [np.searchsorted(dvh1.values[:,x], 0) for x in cols]
    
    
                 

def analyze(dvh_obj, method, value, roi=None, absolute=True):
    dose_DF = pd.DataFrame()
    if type(dvh_obj) != list:
        dvh_obj = [[dvh_obj]]
    for x in dvh_obj:
        for y in x:
            if method == 'Dose at Vol':
                dose_DF = pd.concat([dose_DF, 
                                     dose_at_volume(y, value, roi, absolute)], axis=0)
            if method == 'Vol at Dose':
                dose_DF = pd.concat([dose_DF, 
                                     volume_at_dose(y, value, roi, absolute)], axis=0)
            if method == 'Max Dose':
                dose_DF = pd.concat([dose_DF, max_dose(y, roi)], axis=0)
    dose_DF = multi_index(dose_DF) 
    
    #error checking:
    if nan_check(dose_DF):
        print('analyze() generated dataframe contains '
              '{} NaN values'.format(num_nans(dose_DF)))
    
    return dose_DF

def multi_index(df):
    def mindex_format(iter_list):
        iterables = OrderedDict()
        pt = None
        mod = None
        side = None
        loc = None
        size = None
        for x in iter_list:
            x.sort()
            if type(x[0]) == float:
                #Patient num.
                for i, y in enumerate(x):
                    x[i] = 'Pt {}'.format(str(int(y)))
                pt = x
                iterables['Pt'] = x
            else:
                if x[0] in ['Proton', 'VMAT']:
                    mod = x
                    iterables['Modality'] = x
                if x[0] in ['R', 'L']:
                    side = x
                    iterables['Side'] = x
                #Find Loc:
                if x[0] in ['Inf', 'Pelv', 'PM', 'Sup']:
                    for i, y in enumerate(x):
                        if y == 'PM':
                            x[i] = 'Post Mid'
                    loc = x
                    iterables['Loc'] = x
                #Find Size:
                if x[0] in ['1', '2', '4']:
                    for i, y in enumerate(x):
                        x[i] = y+' cm'
                    size = x
                    iterables['Size'] = x
        
        keyorder = []
        for x in (pt, mod, side, loc, size):
            if x != None:
                keyorder.append(*(k for k, value in iterables.items() if value == x))
        od = OrderedDict((k, iterables[k]) for k in keyorder)
        
        return od   

    #error checking:
    if nan_check(df):
        print('multi_index() recieved dataframe containing '
              '{} NaN values'.format(num_nans(df)))
    
    #iterables = [['Pt1', 'Pt2', 'Pt3'], ['Proton', 'VMAT'], ['L', 'R'],
    #             ['Inf', 'Pelv', 'Post Mid', 'Sup'], ['1 cm', '2 cm', '4 cm']]
    ident = [list(df.iloc[:,-x]) for x in range(1,6)]
    iterables = list(map(list, (set(x) for x in ident)))
    iterables = mindex_format(iterables)

    names = list(iterables.keys())
    mindex = pd.MultiIndex.from_product(iterables.values(), names=names)
    
    df.sort_values(names, inplace=True)
    df.index = mindex
    drop_cols = names 
    df.drop(columns=drop_cols, inplace=True)
    
    #error checking:
    if nan_check(df):
        print('multi_index() generated dataframe '
              'contains {} NaN values'.format(num_nans(df)))

    return df

def polish_dvh(dose_DF, dvh):
    #Add Identifiers and Sort Columns
    dose_DF = add_identifiers(dose_DF, dvh)   
    dose_DF = sort_cols(dose_DF)
    
    return dose_DF
    

def sort_cols(dvh):
    column_std = ['Pt', 'Modality', 'Side', 'Loc', 'Size']
    roi_cols = [x for x in dvh.columns if x not in column_std]
    column_sort = roi_cols + column_std
    dose_DF = dvh[column_sort]
    
    return dose_DF


def nan_check(df):
    return df.isnull().values.any()

def num_nans(df):
    return df.isnull().sum().sum()
    