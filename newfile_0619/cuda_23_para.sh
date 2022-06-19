####maml fomaml pt

cd CrossFit_fomaml/
echo "t5base para fomaml upstream"
bash cli_maml_ddp_prompt_fo_nopara2para_t5base.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/data/*/*.json
cd ..

cd CrossFit_fomaml/
echo "t5base para fomaml downstream"
bash singletask_from_fomaml_nopara2para_t5base.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/data/*/*.json
cd ..

cd CrossFit/
echo "t5base para pt downstream"
bash tune_hps_singletask_ddp_prompt_nopara2para_t5base.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/data/*/*.json
cd ..

cd CrossFit_maml/
echo "t5base para maml upstream"
bash cli_maml_ddp_prompt_nopara2para_t5base.sh
#rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data/*/*.json
cd ..

cd CrossFit_maml/
echo "t5base para maml downstream"
bash singletask_from_meta_nopara2para_t5base.sh
#rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data/*/*.json
cd ..



