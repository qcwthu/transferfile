#cd CrossFit_fomaml/
#echo "fomaml upstream"
#bash cli_maml_ddp_prompt_fo_nopara2para.sh
#rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/data/*/*.json
#cd ..

#cd CrossFit_fomaml/
#echo "fomaml downstream"
#bash singletask_from_fomaml_nopara2para.sh
#rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/data/*/*.json
#cd ..

#cd CrossFitFT/
#echo "ft downstream"
#bash tune_singletask_nopara2para.sh
#rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/data/*/*.json
#cd ..


#cd CrossFit_fomaml/
#echo "fomaml t5base upstream"
#bash cli_maml_ddp_prompt_fo_cls2cls_t5base.sh
#rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/data/*/*.json
#cd ..

cd CrossFit_fomaml/
echo "fomaml t5base downstream"
bash singletask_from_fomaml_cls2cls_t5base.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/data/*/*.json
cd ..

cd CrossFitFT/
echo "ft t5base downstream"
bash tune_singletask_cls2cls_t5base.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/data/*/*.json
cd ..



cd CrossFitFT/
echo "ft 32shot downstream"
bash tune_singletask_cls2cls_down32shot.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/data_32/*/*.json

echo "ft 64shot downstream"
bash tune_singletask_cls2cls_down64shot.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/data_64/*/*.json

echo "ft 128shot downstream"
bash tune_singletask_cls2cls_down128shot.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/data_128/*/*.json
cd ..
