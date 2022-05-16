cd CrossFit/
echo "pt 50prompt downstream"
bash tune_hps_singletask_ddp_prompt_cls2cls_50prompt.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/data/*/*.json

echo "pt 200prompt downstream"
bash tune_hps_singletask_ddp_prompt_cls2cls_200prompt.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/data/*/*.json
cd ..