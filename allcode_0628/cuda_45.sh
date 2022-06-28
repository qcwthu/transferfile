
#cd CrossFit_multi/
#echo "multi t5base downstream"
#bash singletask_from_multi_cls2cls_t5base.sh
#rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data/*/*.json
#cd ..



cd CrossFit_multi/
#echo "multi 32shot upstream"
#bash cli_multitask_ddp_prompt_cls2cls_32shot.sh
#rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data_32/*/*.json

#echo "multi 64shot upstream"
#bash cli_multitask_ddp_prompt_cls2cls_64shot.sh
#rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data_64/*/*.json

#echo "multi 128shot upstream"
#bash cli_multitask_ddp_prompt_cls2cls_128shot.sh
#rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data_128/*/*.json
cd ..


cd CrossFit_multi/
#echo "multi 16shot downstream for 32shot upstream"
#bash singletask_from_multi_cls2cls_up32shot.sh
#rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data/*/*.json

#echo "multi 16shot downstream for 64shot upstream"
#bash singletask_from_multi_cls2cls_up64shot.sh
#rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data/*/*.json

echo "t5base para multi upstream"
bash cli_multitask_ddp_prompt_nopara2para_t5base.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data/*/*.json

echo "t5base para multi downstream"
bash singletask_from_multi_nopara2para_t5base.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data/*/*.json

echo "multi 16shot downstream for 128shot upstream"
bash singletask_from_multi_cls2cls_up128shot.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data/*/*.json
cd ..

#cd CrossFit_multi/
#echo "multi 128shot downstream for 16shot upstream"
#bash singletask_from_multi_cls2cls_down128shot.sh
#rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data_128/*/*.json
cd ..


#########################################################################################from cuda_new_45.sh
cd CrossFit_multi/
#echo "multi 50prompt upstream"
#bash cli_multitask_ddp_prompt_cls2cls_50prompt.sh
#rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data/*/*.json

#echo "multi 50prompt downstream"
#bash singletask_from_multi_cls2cls_50prompt.sh
#rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data/*/*.json

#echo "multi 200prompt upstream"
#bash cli_multitask_ddp_prompt_cls2cls_200prompt.sh
#rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data/*/*.json

#echo "multi 200prompt downstream"
#bash singletask_from_multi_cls2cls_200prompt.sh
#rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data/*/*.json
cd ..
