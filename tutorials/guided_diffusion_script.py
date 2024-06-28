

# initialize config and wandb
from omegaconf import OmegaConf
import hydra
from cortex.logging import wandb_setup

import acgt_dataset
import torch
import lightning as L

import numpy as np

if __name__=='__main__':

    outdir='.'

    #####

    with hydra.initialize(config_path="./hydra"):
        cfg = hydra.compose(config_name="4_guided_diffusion")
        OmegaConf.set_struct(cfg, False)

    wandb_setup(cfg)
    
    dataset=acgt_dataset.ACGTDataset(root=outdir,train=True)

    med_idx = len(dataset) // 2
    init_df = dataset._data.sort_values("y").iloc[med_idx : med_idx + 1]
    init_df = init_df.sample(n=cfg.optim.max_num_solutions, replace=True)

    #print(dataset)
    #print(len(dataset))
    #print(dir(dataset))
    #print(dataset._data.keys())
    #print(dataset._data['y'])
    
    L.seed_everything(seed=cfg.random_seed, workers=True) # set random seed

    model = hydra.utils.instantiate(cfg.tree) # instantiate model
    model.build_tree(cfg, skip_task_setup=False)

    trainer = hydra.utils.instantiate(cfg.trainer)    # instantiate trainer, set logger

    trainer.fit(
        model,
        train_dataloaders=model.get_dataloader(split="train"),
        val_dataloaders=model.get_dataloader(split="val"),
    )

    # construct guidance objective
    initial_solution = init_df["tokenized_seq"].values
    acq_fn_runtime_kwargs = hydra.utils.call(
        cfg.guidance_objective.runtime_kwargs, model=model, candidate_points=initial_solution
    )
    acq_fn = hydra.utils.instantiate(cfg.guidance_objective.static_kwargs, **acq_fn_runtime_kwargs)

    tokenizer_transform = model.root_nodes["protein_seq"].eval_transform
    tokenizer = tokenizer_transform[0].tokenizer

    tok_idxs = tokenizer_transform(initial_solution)
    is_mutable = tokenizer.get_corruptible_mask(tok_idxs)
    is_mutable

    optimizer = hydra.utils.instantiate(
        cfg.optim,
        params=tok_idxs,
        is_mutable=is_mutable,
        model=model,
        objective=acq_fn,
        constraint_fn=None,
    )
    for _ in range(cfg.num_steps):
        optimizer.step()
    new_designs = optimizer.get_best_solutions()

    with torch.inference_mode():
        tree_output = model.call_from_str_array(new_designs["protein_seq"].values, corrupt_frac=0.0)
        final_obj_vals = acq_fn.get_objective_vals(tree_output)
    
    print(f"{new_designs.shape=}") # new_designs.shape=(16, 4)
    print(f"{new_designs=}")
    np.save('new_designs.npy',new_designs)
    print(f"{final_obj_vals=}")
    print("SCRIPT END.")