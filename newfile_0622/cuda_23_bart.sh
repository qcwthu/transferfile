####fomaml ft

cd CrossFit_fomaml/
echo "bart-large para fomaml upstream"
bash cli_maml_ddp_prompt_fo_nopara2para_bart.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/data/*/*.json
cd ..

cd CrossFit_fomaml/
echo "bart-large para fomaml downstream"
bash singletask_from_fomaml_nopara2para_bart.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/data/*/*.json
cd ..

cd CrossFitFT/
echo "bart-large para ft downstream"
bash tune_singletask_nopara2para_bart.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/data/*/*.json
cd ..




