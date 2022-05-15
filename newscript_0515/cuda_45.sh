cd CrossFit_multi/
echo "multi upstream"
bash cli_multitask_ddp_prompt_nopara2para.sh

echo "multi downstream"
bash singletask_from_multi_nopara2para.sh

rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data/*/*.json

echo "multi 32shot upstream"
bash cli_multitask_ddp_prompt_cls2cls_32shot.sh

echo "multi 64shot upstream"
bash cli_multitask_ddp_prompt_cls2cls_64shot.sh

echo "multi 128shot upstream"
bash cli_multitask_ddp_prompt_cls2cls_128shot.sh
cd ..

rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data/*/*.json

cd CrossFit_multi/
echo "multi 16shot downstream for 32shot upstream"
bash singletask_from_multi_cls2cls_up32shot.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data/*/*.json

echo "multi 16shot downstream for 64shot upstream"
bash singletask_from_multi_cls2cls_up64shot.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data/*/*.json

echo "multi 16shot downstream for 128shot upstream"
bash singletask_from_multi_cls2cls_up128shot.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data/*/*.json
cd ..

cd CrossFit_multi/
echo "multi 32shot downstream for 16shot upstream"
bash singletask_from_multi_cls2cls_down32shot.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data/*/*.json

echo "multi 64shot downstream for 16shot upstream"
bash singletask_from_multi_cls2cls_down64shot.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data/*/*.json

echo "multi 128shot downstream for 16shot upstream"
bash singletask_from_multi_cls2cls_down128shot.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data/*/*.json
cd ..

