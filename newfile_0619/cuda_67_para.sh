cd CrossFit_reptile/
echo "t5base para reptile upstream"
bash cli_reptile_ddp_prompt_nopara2para_t5base.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/data/*/*.json
cd ..

cd CrossFit_reptile/
echo "t5base para reptile downstream"
bash singletask_from_reptile_nopara2para_t5base.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/data/*/*.json
cd ..

cd CrossFitFT/
echo "t5base para ft downstream"
bash tune_singletask_nopara2para_t5base.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/data/*/*.json
cd ..

cd CrossFit_multi/
echo "t5base para multi upstream"
bash cli_multitask_ddp_prompt_nopara2para_t5base.sh
#rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data/*/*.json
cd ..

cd CrossFit_multi/
echo "t5base para multi downstream"
bash singletask_from_multi_nopara2para_t5base.sh
#rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data/*/*.json
cd ..