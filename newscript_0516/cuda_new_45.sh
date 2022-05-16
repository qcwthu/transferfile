cd CrossFit_multi/
echo "multi 50prompt upstream"
bash cli_multitask_ddp_prompt_cls2cls_50prompt.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data/*/*.json

echo "multi 50prompt downstream"
bash singletask_from_multi_cls2cls_50prompt.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data/*/*.json

echo "multi 200prompt upstream"
bash cli_multitask_ddp_prompt_cls2cls_200prompt.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data/*/*.json

echo "multi 200prompt downstream"
bash singletask_from_multi_cls2cls_200prompt.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data/*/*.json
cd ..