
for((i=1;i<=80;i++));  
do

	echo 'echo `date`'
	echo 'pkill -f ray;sleep 0.1'
	#echo 'CUDA_VISIBLE_DEVICES="0" python ptest.py  --test_days 120 --num_workers 120 --checkpoint="/home/jz/tmp/PPO_TradingEnv_0_2020-08-12_16-23-48i3lk_4k2/checkpoint_'$i'0/checkpoint-'$i'0"   > logs8.'$i'0'
	echo  'CUDA_VISIBLE_DEVICES="0" python ptest.py  --test_days 120 --num_workers 120 --checkpoint="/home/jz/tmp/PPO_TradingEnv_0_2020-08-12_16-23-06q9ortuzz/checkpoint_'$i'0/checkpoint-'$i'0"    > logs4.'$i'0'
	echo 'echo `date`'
done
