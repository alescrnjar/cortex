import pandas as pd

if __name__=='__main__':
    csvf='/home/alessandro/Downloads/rfp_known_structures/rfp_known_structures.csv'
    df=pd.read_csv(csvf,sep=',')
    print(f"{df.columns=}")
    #print(f"{df['tokenized_seq'][0]=}")
    print(f"{df['foldx_seq'][0]=}")
    tokenized_seqs = []
    for seq in df['foldx_seq']:
        tokenized_seqs.append(" ".join(seq))
    print(f"{tokenized_seqs[0]=}")
    print(f"{df['foldx_total_energy'][0]=}")
    print(f"{df['SASA'][0]=}")
