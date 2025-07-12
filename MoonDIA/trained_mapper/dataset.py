import torch, numpy as np, json
from torch.utils.data import Dataset
from pathlib import Path
from semantic2dac_model import EOS, PAD

def flatten_dac(mat: np.ndarray) -> np.ndarray:
    """
    Convert (T, 9) raw DAC matrix → 1-D sequence with band offset.

        class_id = band * 1024 + code
    """
    offset = (np.arange(9, dtype=np.int64) * 1024)[None, :]   # (1, 9)
    return (mat.astype(np.int64) + offset).reshape(-1)


def load_audio_assessment_results(assessment_file="audio_assessment_results.json", min_similarity=1.0):
    """Load satisfactory pairs from audio assessment results."""
    with open(assessment_file, 'r') as f:
        results = json.load(f)
    
    # Extract indices of satisfactory pairs with perfect similarity
    satisfactory_indices = []
    for pair in results.get("satisfactory_pairs", []):
        if pair.get("both_satisfactory", False) and pair.get("similarity", 0) >= min_similarity:
            satisfactory_indices.append(pair["index"])
    
    print(f"Loaded {len(satisfactory_indices)} satisfactory pairs with similarity >= {min_similarity} from audio assessment")
    return sorted(satisfactory_indices)

class PairDataset(Dataset):
    def __init__(self, root, assessment_file="audio_assessment_results.json", min_similarity=1.0):
        self.root = Path(root)
        
        # Load satisfactory indices from audio assessment
        satisfactory_indices = load_audio_assessment_results(assessment_file, min_similarity)
        
        # Get only the satisfactory pairs
        self.ids = []
        for idx in satisfactory_indices:
            base_name = f"{idx:05d}"  # Format as 5-digit zero-padded
            mc_file = self.root / f"{base_name}.mc.npy"
            dac_file = self.root / f"{base_name}.dac.npy"
            
            if mc_file.exists() and dac_file.exists():
                self.ids.append(base_name)
            else:
                print(f"Warning: Missing files for index {idx} ({base_name})")
        
        self.ids = sorted(self.ids)
        print(f"Found {len(self.ids)} valid pairs from audio assessment")

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        uid = self.ids[i]
        sem = np.load(self.root/f"{uid}.mc.npy").astype(np.int64)
        dac = flatten_dac(np.load(self.root/f"{uid}.dac.npy"))
        return torch.tensor(sem), torch.tensor(dac)

def collate(batch):
    sems, dacs = zip(*batch)
    S = max(len(s) for s in sems)
    T = max(len(d) for d in dacs) + 2          # +<eos> + leading token
    sem_pad = torch.full((len(batch), S), PAD, dtype=torch.long)
    dac_pad = torch.full((len(batch), T), PAD, dtype=torch.long)
    for i,(s,d) in enumerate(zip(sems,dacs)):
        sem_pad[i,:len(s)] = s
        dac_pad[i,0]       = EOS               # start token
        dac_pad[i,1:1+len(d)] = d
        dac_pad[i,1+len(d)]  = EOS             # <eos>
    return sem_pad, dac_pad 