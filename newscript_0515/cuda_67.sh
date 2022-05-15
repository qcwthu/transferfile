cd CrossFit_reptile/
echo "reptile upstream"
bash cli_reptile_ddp_prompt_nopara2para.sh
cd ..

cd CrossFit_reptile/
echo "reptile downstream"
bash singletask_from_reptile_nopara2para.sh
cd ..

rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/data/*/*.json

cd CrossFit/
echo "pt downstream"
bash tune_hps_singletask_ddp_prompt_nopara2para.sh
cd ..

rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/data/*/*.json

cd CrossFit/
echo "pt 32shot downstream"
bash tune_hps_singletask_ddp_prompt_cls2cls_down32shot.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/data/*/*.json

echo "pt 64shot downstream"
bash tune_hps_singletask_ddp_prompt_cls2cls_down64shot.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/data/*/*.json

echo "pt 128shot downstream"
bash tune_hps_singletask_ddp_prompt_cls2cls_down128shot.sh
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/data/*/*.json
cd ..