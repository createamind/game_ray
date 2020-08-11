
for((i=1;i<=99;i++));  
do

	echo 'echo `date`'
	echo 'CUDA_VISIBLE_DEVICES="0" python ptest.py  --test_days 120 --num_workers 120 --checkpoint="/home/zdx/ray_results/PPO-dataV-r19-num_workers=130-model=800,800,600,500,500,500,500,500-lstm=True-batch_size=40000-obs_dim38-as15-action_repeat=2500-auto_follow16-max_ep_len300-burn_in3000-fs1-jump1-ts0-ss1.5-ps0-ap0-dl0-clip10000000000-gamma0.8-lr4e-05-entropy0.003/PPO_TradingEnv_0_2020-08-07_11-51-03wy8du9r1/checkpoint_'$i'0/checkpoint-'$i'0"   > logs5.'$i'0'
	#echo  'CUDA_VISIBLE_DEVICES="0" python ptest.py  --test_days 120 --num_workers 120 --checkpoint="/home/zdx/ray_results/PPO-dataV-r19-num_workers=130-model=800,800,600,500,500,500,500,500-lstm=True-batch_size=40000-obs_dim38-as15-action_repeat=3000-auto_follow16-max_ep_len300-burn_in3000-fs1-jump1-ts0-ss1.5-ps0-ap0-dl0-clip10000000000-gamma0.8-lr4e-05-entropy0.008/PPO_TradingEnv_0_2020-08-07_18-50-017ysj3r0f/checkpoint_'$i'0/checkpoint-'$i'0"    > logs8.'$i'0'
	echo 'echo `date`'
done
