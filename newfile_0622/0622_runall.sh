mkdir alltouse_0622
cd alltouse_0622

###cp alllog
cp /export/share/sjoty/continual-learning/MetaPromptTuning/*.log ./
cp /export/share/sjoty/continual-learning/MetaPromptTuning/*.sh ./

###cp all folders


###32 64 128 upstream, maml multi
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/*up128shot* ./ -r

###number of prompt token: 50,200  pt maml multi
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/*200prompt* ./ -r

###base nopara2para
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/models/T5-base-nopara2para ./ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/models/T5-base-ft-nopara2para ./ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-base-maml-nopara2para-3e-5-2-5000-5e-1 ./ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/models/T5-base-fomaml-nopara2para-3e-5-2-5000-5e-1 ./ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-base-multitask-nopara2para-5e-1-4-20 ./ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/models/T5-base-reptile-nopara2para-3e-5-2-5000-5e-1-10 ./ -r

cd ..


#######handle bart-large


cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/cli_maml_ddp_prompt_nonli2nli.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/cli_maml_ddp_prompt_nopara2para_bart.py
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/cli_maml_ddp_prompt_fo_nonli2nli.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/cli_maml_ddp_prompt_fo_nopara2para_bart.py
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/cli_multitask_ddp_prompt_nonli2nli.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/cli_multitask_ddp_prompt_nopara2para_bart.py
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/cli_reptile_ddp_prompt_nonli2nli.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/cli_reptile_ddp_prompt_nopara2para_bart.py

cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/tune_hps_singletask_ddp_prompt_nonli2nli.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/tune_hps_singletask_ddp_prompt_nopara2para_bart.py
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/tune_singletask_nonli2nli.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/tune_singletask_nopara2para_bart.py
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/singletask_from_meta_nonli2nli.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/singletask_from_meta_nopara2para_bart.py
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/singletask_from_fomaml_nonli2nli.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/singletask_from_fomaml_nopara2para_bart.py
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/singletask_from_multi_nonli2nli.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/singletask_from_multi_nopara2para_bart.py
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/singletask_from_reptile_nonli2nli.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/singletask_from_reptile_nopara2para_bart.py

cp BartLarge-files/allfolders/Bart-large-nopara2para /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/models/ -r
cp BartLarge-files/allfolders/Bart-large-ft-nopara2para /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/models/ -r
cp BartLarge-files/allfolders/Bart-large-maml-nopara2para-3e-5-2-5000-5e-1 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/ -r
cp BartLarge-files/allfolders/Bart-large-fomaml-nopara2para-3e-5-2-5000-5e-1 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/models/ -r
cp BartLarge-files/allfolders/Bart-large-multitask-nopara2para-5e-1-4-20 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/ -r
cp BartLarge-files/allfolders/Bart-large-reptile-nopara2para-3e-5-2-5000-5e-1-10 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/models/ -r

cp BartLarge-files/cli_maml_ddp_prompt_nopara2para_bart.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/
cp BartLarge-files/cli_maml_ddp_prompt_fo_nopara2para_bart.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/
cp BartLarge-files/cli_multitask_ddp_prompt_nopara2para_bart.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/
cp BartLarge-files/cli_reptile_ddp_prompt_nopara2para_bart.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/

cp BartLarge-files/tune_hps_singletask_ddp_prompt_nopara2para_bart.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/
cp BartLarge-files/tune_singletask_nopara2para_bart.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/
cp BartLarge-files/singletask_from_meta_nopara2para_bart.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/
cp BartLarge-files/singletask_from_fomaml_nopara2para_bart.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/
cp BartLarge-files/singletask_from_multi_nopara2para_bart.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/
cp BartLarge-files/singletask_from_reptile_nopara2para_bart.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/

####备份要修改的
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/T5Prompt.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/T5Prompt_0622bak.py
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/T5Model.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/T5Model_0622bak.py
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/T5Prompt.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/T5Prompt_0622bak.py
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/T5Prompt.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/T5Prompt_0622bak.py
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/T5Prompt.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/T5Prompt_0622bak.py
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/T5Prompt.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/T5Prompt_0622bak.py


cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/run_singletask_ddp_prompt.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/run_singletask_ddp_prompt_0622bak.py
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/run_singletask_ddp.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/run_singletask_ddp_0622bak.py
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/run_singletask_ddp_prompt.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/run_singletask_ddp_prompt_0622bak.py
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/run_singletask_ddp_prompt.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/run_singletask_ddp_prompt_0622bak.py
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/run_singletask_ddp_prompt.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/run_singletask_ddp_prompt_0622bak.py
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/run_singletask_ddp_prompt.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/run_singletask_ddp_prompt_0622bak.py


####需要手动copy的几个，先vim看一下下面几个文件，看一下里面用的哪个file，然后备份一下
###vim cli_maml_ddp_prompt_nopara2para_bart.py
###vim cli_maml_ddp_prompt_fo_nopara2para_bart.py
###vim cli_multitask_ddp_prompt_nopara2para_bart.py
###vim cli_reptile_ddp_prompt_nopara2para_bart.py



####cuda sh
cp cuda_01_bart.sh cuda_23_bart.sh cuda_67_bart.sh /export/share/sjoty/continual-learning/MetaPromptTuning/
