
# /home/alessandro/Documents/LocalColabFold_Files/localcolabfold/colabfold-conda/lib/python3.10/site-packages/cortex/data/dataset/_rfp_dataset.py
import pandas as pd

from cortex.data.dataset._data_frame_dataset import DataFrameDataset

#_DOWNLOAD_URL = (
#    "https://raw.githubusercontent.com/samuelstanton/lambo/main/lambo/assets/fpbase/rfp_known_structures.tar.gz"
#)
_DOWNLOAD_URL = (
    "dummy"
)

def tokenize_acgt_df(data: pd.DataFrame) -> pd.DataFrame:
    raw_seqs = data["foldx_seq"]
    tokenized_seqs = []
    for seq in raw_seqs:
        tokenized_seqs.append(" ".join(seq))
    data["tokenized_seq"] = tokenized_seqs
    return data


class ACGTDataset(DataFrameDataset):
    _name = "acgt"
    _target = "forcortex_newLentiMPRAK562_labels-seed0_random0_25000.csv" # "rfp_known_structures.csv"
    columns = [
        #"tokenized_seq",
        #"foldx_total_energy",
        #"SASA",
        "Sequence",
        "y",
    ]
    """
    df['foldx_seq'][0]='LSKHGLTKDMTMKYRMEGCVDGHKFVITGHGNGSPFEGKQTINLCVVEGGPLPFSEDILSAVFNRVFTDYPQGMVDFFKNSCPAGYTWQRSLLFEDGAVCTASADITVSVEENCFYHESKFHGVNFPADGPVMKKMTINWEPCCEKIIPVPRQGILKGDVAMYLLLKDGGRYRCQFDTVYKAKTDSKKMPEWHFIQHKLTREDRSDAKNQKWQLAEHSVASRSALA'
    tokenized_seqs[0]='L S K H G L T K D M T M K Y R M E G C V D G H K F V I T G H G N G S P F E G K Q T I N L C V V E G G P L P F S E D I L S A V F N R V F T D Y P Q G M V D F F K N S C P A G Y T W Q R S L L F E D G A V C T A S A D I T V S V E E N C F Y H E S K F H G V N F P A D G P V M K K M T I N W E P C C E K I I P V P R Q G I L K G D V A M Y L L L K D G G R Y R C Q F D T V Y K A K T D S K K M P E W H F I Q H K L T R E D R S D A K N Q K W Q L A E H S V A S R S A L A'
    """

    def __init__(self, root: str, download: bool = False, download_source: str = _DOWNLOAD_URL, **kwargs):
        super().__init__(root=root, download=download, download_source=download_source, **kwargs)
        self._data = tokenize_acgt_df(self._data)
