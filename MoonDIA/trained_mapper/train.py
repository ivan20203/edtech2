import torch, torch.nn.functional as F, random, numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from semantic2dac_model import Semantic2DAC, PAD
from dataset import PairDataset, collate

# Aggressive training hyper-params
BS, EPOCHS, LR = 12, 30, 1e-3          # Much higher LR: 6e-4 → 1e-3
WARM_STEPS = 100                       # Shorter warm-up: 200 → 100
PATIENCE = 8
device = "cuda" if torch.cuda.is_available() else "cpu"

ds = PairDataset("data/train")                 # <-- path to your .npy pairs
tr, va = random_split(ds, [int(0.85*len(ds)), len(ds)-int(0.85*len(ds))])
tr_dl = DataLoader(tr, BS, shuffle=True,  collate_fn=collate)
va_dl = DataLoader(va, BS, shuffle=False, collate_fn=collate)

model = Semantic2DAC(d_model=384, n_heads=12, dropout=0.0).to(device)  # No dropout

# Start fresh - don't load old checkpoint
print("Starting training from scratch with aggressive settings")

# Snappier Adam with higher LR
opt = torch.optim.AdamW(model.parameters(), LR, betas=(0.9, 0.95))

# Much longer cosine cycle to prevent premature LR decay
steps_per_epoch = len(tr_dl)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt,
    T_max=10000,                       # ~200 epochs instead of 30
    eta_min=LR * 0.1                   # 1e-4 minimum
)

# Mixed precision training
scaler = torch.cuda.amp.GradScaler()

best = 9e9; patience=0
global_step = 0

for epoch in range(EPOCHS):
    model.train(); tot=0; steps=0
    for sem,dac in tqdm(tr_dl, desc=f"epoch {epoch}"):
        sem,dac = sem.to(device), dac.to(device)
        
        with torch.cuda.amp.autocast():
            logits  = model(sem, dac)
            # No label smoothing at all
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                dac[:,1:].reshape(-1),
                ignore_index=PAD,
            )
        
        opt.zero_grad()
        scaler.scale(loss).backward()
        
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        scaler.step(opt); scaler.update()
        global_step += 1
        
        # Linear warm-up
        if global_step < WARM_STEPS:
            lr_scale = global_step / WARM_STEPS
            for pg in opt.param_groups:
                pg["lr"] = LR * lr_scale
        else:
            scheduler.step()
            
        tot+=loss.item(); steps+=1
    print(f"train CE {tot/steps:.3f}")
    print(f"current LR: {opt.param_groups[0]['lr']:.2e}")

    # validation
    model.eval(); vtot=0; vsteps=0
    with torch.no_grad():
        for sem,dac in va_dl:
            sem,dac = sem.to(device), dac.to(device)
            logits  = model(sem,dac)
            vtot += F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                dac[:,1:].reshape(-1),
                ignore_index=PAD,
            ).item(); vsteps+=1
    val = vtot/vsteps
    print(f"val CE {val:.3f}")

    if val < best: best=val; patience=0; torch.save(model.state_dict(),"s2d.pt")
    else:
        patience+=1
        if patience==PATIENCE:
            print("Early stop."); break

print("Best val CE:", best) 