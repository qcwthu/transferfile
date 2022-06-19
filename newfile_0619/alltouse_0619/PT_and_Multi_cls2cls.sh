cd CrossFit
echo "start anli of PT for cls2cls"
bash tune_hps_singletask_ddp_prompt_cls2cls.sh
echo "finish anli of PT for cls2cls"
cd ..



cd CrossFit_multi
echo "start all tasks of multi for cls2cls"
bash singletask_from_multi_cls2cls.sh 
echo "finish all tasks of multi for cls2cls"
cd ..
