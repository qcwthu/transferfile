cp newsettings/*.json /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/dataloader/custom_tasks_splits/ -r
cp newsettings/*.json /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/dataloader/custom_tasks_splits/ -r
cp newsettings/*.json /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/dataloader/custom_tasks_splits/ -r
cp newsettings/*.json /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/dataloader/custom_tasks_splits/ -r
cp newsettings/*.json /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/dataloader/custom_tasks_splits/ -r
cp newsettings/*.json /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/dataloader/custom_tasks_splits/ -r

cp allscriptforQA/cli_maml_ddp_prompt_qa2qa.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/ -r
cp allscriptforQA/cli_maml_ddp_prompt_noqa2qa.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/ -r

cp allscriptforQA/cli_maml_ddp_prompt_fo_qa2qa.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/ -r
cp allscriptforQA/cli_maml_ddp_prompt_fo_noqa2qa.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/ -r

cp allscriptforQA/cli_multitask_ddp_prompt_qa2qa.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/ -r
cp allscriptforQA/cli_multitask_ddp_prompt_noqa2qa.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/ -r

cp allscriptforQA/cli_reptile_ddp_prompt_qa2qa.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/ -r
cp allscriptforQA/cli_reptile_ddp_prompt_noqa2qa.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/ -r

cp allscriptfornli/cli_maml_ddp_prompt_nonli2nli.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/ -r

cp allscriptfornli/cli_maml_ddp_prompt_fo_nonli2nli.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/ -r

cp allscriptfornli/cli_multitask_ddp_prompt_nonli2nli.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/ -r

cp allscriptfornli/cli_reptile_ddp_prompt_nonli2nli.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/ -r