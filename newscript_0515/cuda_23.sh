cd CrossFit_fomaml/
echo "fomaml upstream"
bash cli_maml_ddp_prompt_fo_nopara2para.sh
cd ..

cd CrossFit_fomaml/
echo "fomaml downstream"
bash singletask_from_fomaml_nopara2para.sh
cd ..

rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/data/*/*.json

cd CrossFitFT/
echo "ft downstream"
bash tune_singletask_nopara2para.sh
cd ..

rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/data/*/*.json

cd CrossFitFT/
echo "ft 32shot downstream"
bash tune_singletask_cls2cls_down32shot.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/data/*/*.json

echo "ft 64shot downstream"
bash tune_singletask_cls2cls_down64shot.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/data/*/*.json

echo "ft 128shot downstream"
bash tune_singletask_cls2cls_down128shot.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/data/*/*.json
cd ..
