####reptile multi

cd CrossFit_reptile/
echo "bart-large para reptile upstream"
bash cli_reptile_ddp_prompt_nopara2para_bart.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/data/*/*.json
cd ..

cd CrossFit_reptile/
echo "bart-large para reptile downstream"
bash singletask_from_reptile_nopara2para_bart.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/data/*/*.json
cd ..

cd CrossFit_multi/
echo "bart-large para multi upstream"
bash cli_multitask_ddp_prompt_nopara2para_bart.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data/*/*.json
cd ..

cd CrossFit_multi/
echo "bart-large para multi downstream"
bash singletask_from_multi_nopara2para_bart.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data/*/*.json
cd ..