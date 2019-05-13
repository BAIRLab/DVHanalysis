import glob
import re
import os

import numpy as np
import pandas as pd

#Set data folder
data = 'Data/RCC/'
pt1 = 'Zielinsky/'
pt2 = 'Strnad/'
pt3 = 'Zeop/'

stdROIs = ['Cord', 'Heart', 'Stomach', 'Duodenum',
           'Small Bowel', 'Large Bowel', 'Liver',
           'Skin Rind']

def generate_DataFrames():
    #Function to build list of csv files from a folder
    def csv_list(patient_folder):
        csv_file_list = glob.glob(patient_folder+'*.csv', recursive=True)
        return csv_file_list
    
    def plan_details(csvfilename):
        # Get a filename:
        name = os.path.split(csvfilename)[-1].rstrip('cmDVH.csv')
        
        modality = re.match('(.*)(?=(L|R))', name).group()
        side = re.search('L|R', name).group()
        loc = re.search('(?<=(L|R))(.*)(?=Kid)', name).group()
        size = re.search('\d', name).group()
        
        return modality, side, loc, size
    
    def relevant_column(col, csv):
        if col == 'Unnamed: 0':
            return True
        if 'interior' in col:
            return False
        if any(x in col for x in stdROIs):
            return True
        
        #Get Relevant Parts (Modality, Side, Loc, Size)
        modality, side, loc, size = plan_details(csv)
        relevant_list = [modality, side, loc, size]
                            
        #Find Contralateral Kidney
        ckid = re.search('(L(?=eft)|R(?=ight))(?=(.*)Kidney)', col)
        if ckid:
            if side == ckid.group():
                return False
            return True
        #Find Ipsilateral Kidney-CTV
        if re.search(side+'.Kidney.'+loc+size, col):
            return True
    
        #Find CTV/PTV
        if loc == 'PM':
            loc = 'Post Mid'
        if modality == 'Proton':
            if re.search(side+'.'+loc+'.Kidney.'+size+'.cm.(?!PTV)', col):
                return True
            return False
        if modality == 'VMAT':
            if re.search(side+'.'+loc+'.Kidney.'+size+'.cm.(?=PTV)', col):
                return True
            return False
        return False
    
    def prettify_dvh(dvh, csv, pt):
        #Strip volume measurements from column title and append to bottom of dataframe
        pat = re.compile('(?<=Volume..)(.*)(?=\))')
        volumes = [re.search(pat, x).group() for x in dvh.columns if re.search(pat, x)]
        volumes.insert(0,0.0)
        dvh.loc['Volume'] = np.array(volumes, dtype=float)
        
        #Label Dose Column
        dvh.rename(columns={dvh.columns[0]: "Dose"}, inplace=True)
        
        #Fill NaNs with 0
        dvh.fillna(0.0, inplace=True)
        #Rename Columns to Standard Names
        pat2 = re.compile('(.*)(?= \()')
        names = [re.search(pat2, x).group() for x in dvh.columns if re.search(pat2, x)]
        names.insert(0, dvh.columns[0])
        
        modality, side, loc, size = plan_details(csv)
        dvh.loc['Pt'] = pt+1
        dvh.loc['Modality'] = modality
        dvh.loc['Side'] = side
        dvh.loc['Loc'] = loc
        dvh.loc['Size'] = size
        
        if modality == 'Proton':
            locate = loc
            if loc == 'PM':
                locate = 'Post Mid'
            ctv_pat = re.compile(side+'.'+locate+'.Kidney.'+size+'.cm')
            ctv = [re.search(ctv_pat, x).group() for x in names if re.search(ctv_pat, x)]
            idx = names.index(ctv[0])
            names[idx] = 'CTV'  
        if modality == 'VMAT':
            ptv_pat = re.compile('(.*)PTV')
            ptv = [re.search(ptv_pat, x).group() for x in names if re.search(ptv_pat, x)]
            idx = names.index(ptv[0])
            names[idx] = 'PTV'
        
        ikid_pat = re.compile(side+'.Kidney.'+loc+size+'cm')
        ikid = [re.search(ikid_pat, x).group() for x in names if re.search(ikid_pat, x)]
        idx = names.index(ikid[0])
        names[idx] = 'Ipsilateral Kidney'
        
        ckid_pat = re.compile('(Left|Right).Kidney')
        ckid = [re.search(ckid_pat, x).group() for x in names if re.search(ckid_pat, x)]
        idx = names.index(ckid[0])
        names[idx] = 'Contralateral Kidney'
        
        name_map = dict(zip(dvh.columns, names))
        dvh = dvh.rename(columns=name_map)
        
        TV = None
        if modality == 'Proton':
            TV = 'CTV'
        elif modality == 'VMAT':
            TV = 'PTV'
        column_sort = ['Dose', TV, 'Ipsilateral Kidney', 'Contralateral Kidney',
                       'Stomach', 'Duodenum', 'Small Bowel', 'Large Bowel',
                       'Liver', 'Heart', 'Cord', 'Skin Rind']

        dvh = dvh[column_sort]
        
        return dvh
    
    #build a nested patient list with csv file locations:
    #  [[pt1], [pt2], [pt3]]
    #  pt = [csv1, csv2, ... csv n]
    pt1_csv, pt2_csv, pt3_csv = csv_list(data+pt1), csv_list(data+pt2), csv_list(data+pt3)
    pt_csv_list = [pt1_csv, pt2_csv, pt3_csv]
    #return pt_csv_list
    #create a nested patient list with csv files read in as dataframes:
    pt_dvh_df_list = []
    for i, pt in enumerate(pt_csv_list):
        temp_list = []
        for csv in pt:            
            dvh = pd.read_csv(csv, header=1,
                              usecols=(lambda x: relevant_column(x, csv)),
                              dtype=np.float64)
            temp_list.append(prettify_dvh(dvh, csv, i))
            #temp_list.append(dvh, csv)
            
        pt_dvh_df_list.append(temp_list)
        
    #Optional Error-checking:
    num_csv = sum(len(x) for x in pt_csv_list)
    num_dvh = sum(len(x) for x in pt_dvh_df_list)
    print('{} csv files found.'.format(num_csv))
    print('Your final list contains {} dvhs'.format(num_dvh))
    
    mod_pat = re.compile('(.*)(?=(L|R))')
    flat_list = [os.path.split(x)[-1].rstrip('cmDVH.csv') for y in pt_csv_list for x in y]
    plans = [re.search(mod_pat, x).group() for x in flat_list]
    proton = plans.count('Proton')
    vmat = plans.count('VMAT')
    print('{} Proton and {} VMAT csv files were found.'.format(proton, vmat))
    
    flat_list = [x for y in pt_dvh_df_list for x in y]
    ctv = sum(1 for x in flat_list if 'CTV' in x.columns)
    ptv = sum(1 for x in flat_list if 'PTV' in x.columns)
    print('{} DVHs w/CTVs and {} DVHs w/PTVs are in your final list.'.format(ctv,ptv))

    
    return pt_dvh_df_list
    