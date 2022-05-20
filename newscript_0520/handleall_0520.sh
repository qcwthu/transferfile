########downstream Few-Shot to More-Shot, we don't need re-train upstream
###pt ft maml multi
cp ../newscript_0515/32-shot/tune_hps_singletask_ddp_prompt_cls2cls_down32shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/
cp ../newscript_0515/32-shot/tune_singletask_cls2cls_down32shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/
cp ../newscript_0515/32-shot/singletask_from_meta_cls2cls_down32shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/
cp ../newscript_0515/32-shot/singletask_from_multi_cls2cls_down32shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/

cp ../newscript_0515/64-shot/tune_hps_singletask_ddp_prompt_cls2cls_down64shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/
cp ../newscript_0515/64-shot/tune_singletask_cls2cls_down64shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/
cp ../newscript_0515/64-shot/singletask_from_meta_cls2cls_down64shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/
cp ../newscript_0515/64-shot/singletask_from_multi_cls2cls_down64shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/

cp ../newscript_0515/128-shot/tune_hps_singletask_ddp_prompt_cls2cls_down128shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/
cp ../newscript_0515/128-shot/tune_singletask_cls2cls_down128shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/
cp ../newscript_0515/128-shot/singletask_from_meta_cls2cls_down128shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/
cp ../newscript_0515/128-shot/singletask_from_multi_cls2cls_down128shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/


