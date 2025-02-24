./isaaclab.sh -p source/standalone/workflows/skrl_ctrl/train_sac.py --headless --enable_cameras > scripts/logs/train_sac.log &
./isaaclab.sh -p source/standalone/workflows/skrl_ctrl/train.py --headless --enable_cameras --env_version legtrain-finetune > scripts/logs/train_ctrlsac.log&
