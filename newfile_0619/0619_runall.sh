mkdir alltouse_0619
cd alltouse_0619

###cp alllog
cp /export/share/sjoty/continual-learning/MetaPromptTuning/*.log ./
cp /export/share/sjoty/continual-learning/MetaPromptTuning/*.sh ./

###cp all folders


###32 64 128 upstream, maml multi
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/*up64shot* ./ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/*up128shot* ./ -r


###32 64 128 downstream, pt ft maml multi
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/*down128shot* ./ -r

###number of prompt token: 50,200  pt maml multi
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/*200prompt* ./ -r

###number of meta-training tasks: maml cls2cls
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/*24uptasks* ./ -r


cd ..

cp t5base_nopara2para/cli_maml_ddp_prompt_nopara2para_t5base.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/
cp t5base_nopara2para/cli_maml_ddp_prompt_fo_nopara2para_t5base.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/
cp t5base_nopara2para/cli_multitask_ddp_prompt_nopara2para_t5base.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/
cp t5base_nopara2para/cli_reptile_ddp_prompt_nopara2para_t5base.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/

cp t5base_nopara2para/tune_hps_singletask_ddp_prompt_nopara2para_t5base.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/
cp t5base_nopara2para/tune_singletask_nopara2para_t5base.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/
cp t5base_nopara2para/singletask_from_meta_nopara2para_t5base.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/
cp t5base_nopara2para/singletask_from_fomaml_nopara2para_t5base.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/
cp t5base_nopara2para/singletask_from_multi_nopara2para_t5base.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/
cp t5base_nopara2para/singletask_from_reptile_nopara2para_t5base.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/

cp t5base_nopara2para/allfolders/T5-base-nopara2para /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/ -r
cp t5base_nopara2para/allfolders/T5-base-ft-nopara2para /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/ -r
cp t5base_nopara2para/allfolders/T5-base-maml-nopara2para-3e-5-2-5000-5e-1 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/ -r
cp t5base_nopara2para/allfolders/T5-base-fomaml-nopara2para-3e-5-2-5000-5e-1 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/ -r
cp t5base_nopara2para/allfolders/T5-base-multitask-nopara2para-5e-1-4-20 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/ -r
cp t5base_nopara2para/allfolders/T5-base-reptile-nopara2para-3e-5-2-5000-5e-1-10 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/ -r


cp cuda_23_para.sh cuda_67_para.sh /export/share/sjoty/continual-learning/MetaPromptTuning/