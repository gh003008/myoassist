#!/usr/bin/env bash
# Training with v7 HDF5-converted reference motion
# Proper initialization with reference first pose

(C:\ProgramData\Anaconda3\Scripts\activate)
conda activate myoassist

python rl_train/run_train.py \
  --env MyoAssistLegImitation-v1_0 \
  --model_xml models/26muscle_3D/myoLeg26_BASELINE.xml \
  --reference_npz rl_train/reference_data/S004_trial01_08mps_3D_HDF5_v7.npz \
  --total_timesteps 50000000 \
  --n_steps 2048 \
  --batch_size 64 \
  --n_epochs 10 \
  --learning_rate 0.0003 \
  --gamma 0.99 \
  --gae_lambda 0.95 \
  --clip_range 0.2 \
  --ent_coef 0.0 \
  --vf_coef 0.5 \
  --max_grad_norm 0.5 \
  --device cuda \
  --num_envs 12 \
  --checkpoint_save_interval 1000000 \
  --eval_render_interval 5000000 \
  --wandb_project myoassist-imitation \
  --wandb_name S004_3D_HDF5_v7_proper_init \
  --use_wandb
