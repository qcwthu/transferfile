cd CrossFit_reptile/
echo "reptile upstream"
bash cli_reptile_ddp_prompt_nopara2para.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/data/*/*.json
cd ..

cd CrossFit_reptile/
echo "reptile downstream"
bash singletask_from_reptile_nopara2para.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/data/*/*.json
cd ..

cd CrossFit/
echo "pt downstream"
bash tune_hps_singletask_ddp_prompt_nopara2para.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/data/*/*.json
cd ..


cd CrossFit_reptile/
echo "reptile t5base upstream"
bash cli_reptile_ddp_prompt_cls2cls_t5base.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/data/*/*.json
cd ..

cd CrossFit_reptile/
echo "reptile t5base downstream"
bash singletask_from_reptile_cls2cls_t5base.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/data/*/*.json
cd ..

cd CrossFit/
echo "pt t5base downstream"
bash tune_hps_singletask_ddp_prompt_cls2cls_t5base.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/data/*/*.json
cd ..



#cd CrossFit/
#echo "pt 32shot downstream"
#bash tune_hps_singletask_ddp_prompt_cls2cls_down32shot.sh
#rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/data_32/*/*.json
#
#echo "pt 64shot downstream"
#bash tune_hps_singletask_ddp_prompt_cls2cls_down64shot.sh
#rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/data_64/*/*.json
#
#echo "pt 128shot downstream"
#bash tune_hps_singletask_ddp_prompt_cls2cls_down128shot.sh
#rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/data_128/*/*.json
#cd ..


##########################################################################################from cuda_new_67.sh
#cd CrossFit/
#echo "pt 50prompt downstream"
#bash tune_hps_singletask_ddp_prompt_cls2cls_50prompt.sh
#rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/data/*/*.json
#
#echo "pt 200prompt downstream"
#bash tune_hps_singletask_ddp_prompt_cls2cls_200prompt.sh
#rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/data/*/*.json
#cd ..