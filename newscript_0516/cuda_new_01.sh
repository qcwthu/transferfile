cd CrossFit_maml/
echo "maml 50prompt upstream"
bash cli_maml_ddp_prompt_cls2cls_50prompt.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data/*/*.json

echo "maml 50prompt downstream"
bash singletask_from_meta_cls2cls_50prompt.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data/*/*.json

echo "maml 200prompt upstream"
bash cli_maml_ddp_prompt_cls2cls_200prompt.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data/*/*.json

echo "maml 200prompt downstream"
bash singletask_from_meta_cls2cls_200prompt.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data/*/*.json
cd ..

echo "-----------------------------------------------------------------------------------"

cd CrossFit_maml/
echo "maml 12uptasks upstream"
bash cli_maml_ddp_prompt_cls2cls_12uptasks.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data/*/*.json

echo "maml 12uptasks downstream"
bash singletask_from_meta_cls2cls_12uptasks.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data/*/*.json

echo "maml 24uptasks upstream"
bash cli_maml_ddp_prompt_cls2cls_24uptasks.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data/*/*.json

echo "maml 24uptasks downstream"
bash singletask_from_meta_cls2cls_24uptasks.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data/*/*.json
cd ..
