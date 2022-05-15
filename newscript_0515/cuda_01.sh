cd CrossFit_maml/
echo "maml upstream"
bash cli_maml_ddp_prompt_nopara2para.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data/*/*.json

echo "maml downstream"
bash singletask_from_meta_nopara2para.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data/*/*.json

echo "maml 32shot upstream"
bash cli_maml_ddp_prompt_cls2cls_32shot.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_32/*/*.json

echo "maml 64shot upstream"
bash cli_maml_ddp_prompt_cls2cls_64shot.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_64/*/*.json

echo "maml 128shot upstream"
bash cli_maml_ddp_prompt_cls2cls_128shot.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_128/*/*.json
cd ..


cd CrossFit_maml/
echo "maml 16shot downstream for 32shot upstream"
bash singletask_from_meta_cls2cls_up32shot.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data/*/*.json

echo "maml 16shot downstream for 64shot upstream"
bash singletask_from_meta_cls2cls_up64shot.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data/*/*.json

echo "maml 16shot downstream for 128shot upstream"
bash singletask_from_meta_cls2cls_up128shot.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data/*/*.json
cd ..

cd CrossFit_maml/
echo "maml 32shot downstream for 16shot upstream"
bash singletask_from_meta_cls2cls_down32shot.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_32/*/*.json

echo "maml 64shot downstream for 16shot upstream"
bash singletask_from_meta_cls2cls_down64shot.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_64/*/*.json

echo "maml 128shot downstream for 16shot upstream"
bash singletask_from_meta_cls2cls_down128shot.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_128/*/*.json
cd ..

cd CrossFit_maml/
echo "maml t5base upstream"
bash cli_maml_ddp_prompt_cls2cls_t5base.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data/*/*.json

echo "maml t5base downstream"
bash singletask_from_meta_cls2cls_t5base.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data/*/*.json
cd ..

