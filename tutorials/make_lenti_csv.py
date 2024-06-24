import h5py
import numpy as np
import pandas as pd
import tqdm
import sys
sys.path.append('../../Occasio_Dev/src/')
import FUNCTIONS_4_DALdna


if __name__=='__main__':
    ##h5f='../../Occasio_Dev/inputs/newLentiMPRAK562_processed_for_dal.h5'
    h5f='../../Occasio_Dev/inputs/newLentiMPRAK562_labels-seed0_random0_25000.h5'
    data=h5py.File(h5f,'r')
    X_train=data['X_train']
    y_train=data['Y_train']
    database={'Sequence':[],'y':[]}
    for i_x,x in tqdm.tqdm(enumerate(X_train),total=len(X_train)):
        dna=FUNCTIONS_4_DALdna.ohe_to_seq(x, four_zeros_ok=True) #.squeeze(0))
        database['Sequence'].append(dna)
        database['y'].append(y_train[i_x])
    df=pd.DataFrame(data=database)
    csvf='./forcortex_newLentiMPRAK562_labels-seed0_random0_25000.csv'
    df.to_csv(csvf,sep=',')

    #https://github.com/alescrnjar/Minerva_Tests/blob/main/src/Step0_Database_scraper.py
    #database={'ruling_type':[],'url':[],'articles':[],'facts':[]} # inizializza un nuovo dizionario vuoto ma con i giusti campi.
    #df=pd.DataFrame(data=database) 
    #df_merged.to_csv(outname,sep=';')