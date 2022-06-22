####maml pt

cd CrossFit_maml/
echo "bart-large para maml upstream"
bash cli_maml_ddp_prompt_nopara2para_bart.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data/*/*.json
cd ..

cd CrossFit_maml/
echo "bart-large para maml downstream"
bash singletask_from_meta_nopara2para_bart.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data/*/*.json
cd ..

cd CrossFit/
echo "bart-large para pt downstream"
bash tune_hps_singletask_ddp_prompt_nopara2para_bart.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/data/*/*.json
cd ..




