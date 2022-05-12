cp task_file_nocls/* /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/tasks/ -r
cp task_file_nocls/* /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/tasks/ -r
cp task_file_nocls/* /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/tasks/ -r
cp task_file_nocls/* /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/tasks/ -r
cp task_file_nocls/* /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/tasks/ -r
cp task_file_nocls/* /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/tasks/ -r

cp newsettings/newtaskfile_fornli/* /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/tasks/ -r
cp newsettings/newtaskfile_fornli/* /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/tasks/ -r
cp newsettings/newtaskfile_fornli/* /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/tasks/ -r
cp newsettings/newtaskfile_fornli/* /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/tasks/ -r
cp newsettings/newtaskfile_fornli/* /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/tasks/ -r
cp newsettings/newtaskfile_fornli/* /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/tasks/ -r

cp newsettings/newtaskfile_forqa/* /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/tasks/ -r
cp newsettings/newtaskfile_forqa/* /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/tasks/ -r
cp newsettings/newtaskfile_forqa/* /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/tasks/ -r
cp newsettings/newtaskfile_forqa/* /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/tasks/ -r
cp newsettings/newtaskfile_forqa/* /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/tasks/ -r
cp newsettings/newtaskfile_forqa/* /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/tasks/ -r

cp newsettings/*.json /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/dataloader/custom_tasks_splits/ -r
cp newsettings/*.json /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/dataloader/custom_tasks_splits/ -r
cp newsettings/*.json /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/dataloader/custom_tasks_splits/ -r
cp newsettings/*.json /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/dataloader/custom_tasks_splits/ -r
cp newsettings/*.json /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/dataloader/custom_tasks_splits/ -r
cp newsettings/*.json /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/dataloader/custom_tasks_splits/ -r

cp noclsdata/* /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/data/ -r
cp noclsdata/* /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/data/ -r
cp noclsdata/* /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data/ -r
cp noclsdata/* /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/data/ -r
cp noclsdata/* /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data/ -r
cp noclsdata/* /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/data/ -r

cp allscriptforQA/cli_maml_ddp_prompt_qa2qa.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/cli_maml_ddp_prompt_tune1.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/cli_maml_ddp_prompt_qa2qa.py -r
cp allscriptforQA/cli_maml_ddp_prompt_noqa2qa.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/cli_maml_ddp_prompt_tune_nocls2cls.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/cli_maml_ddp_prompt_noqa2qa.py -r

cp allscriptforQA/cli_maml_ddp_prompt_fo_qa2qa.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/cli_maml_ddp_prompt_fo.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/cli_maml_ddp_prompt_fo_qa2qa.py -r
cp allscriptforQA/cli_maml_ddp_prompt_fo_noqa2qa.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/cli_maml_ddp_prompt_fo_nocls2cls.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/cli_maml_ddp_prompt_fo_noqa2qa.py -r

cp allscriptforQA/cli_multitask_ddp_prompt_qa2qa.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/cli_multitask_ddp_prompt_tune1.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/cli_multitask_ddp_prompt_qa2qa.py -r
cp allscriptforQA/cli_multitask_ddp_prompt_noqa2qa.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/cli_multitask_ddp_prompt_nocls2cls.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/cli_multitask_ddp_prompt_noqa2qa.py -r

cp allscriptforQA/cli_reptile_ddp_prompt_qa2qa.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/cli_reptile_ddp_prompt_tune1.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/cli_reptile_ddp_prompt_qa2qa.py -r
cp allscriptforQA/cli_reptile_ddp_prompt_noqa2qa.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/cli_reptile_ddp_prompt_nocls2cls.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/cli_reptile_ddp_prompt_noqa2qa.py -r

cp allscriptforQA/tune_hps_singletask_ddp_prompt_qa2qa.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/tune_hps_singletask_ddp_prompt_cls2cls.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/tune_hps_singletask_ddp_prompt_qa2qa.py -r

cp allscriptforQA/tune_singletask_qa2qa.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/tune_singletask_cls2cls.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/tune_singletask_qa2qa.py -r

cp allscriptforQA/singletask_from_meta_qa2qa.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/singletask_from_meta_cls2cls.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/singletask_from_meta_qa2qa.py -r
cp allscriptforQA/singletask_from_meta_noqa2qa.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/singletask_from_meta_nocls2cls.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/singletask_from_meta_noqa2qa.py -r

cp allscriptforQA/singletask_from_fomaml_qa2qa.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/singletask_from_fomaml_cls2cls.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/singletask_from_fomaml_qa2qa.py -r
cp allscriptforQA/singletask_from_fomaml_noqa2qa.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/singletask_from_fomaml_nocls2cls.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/singletask_from_fomaml_noqa2qa.py -r

cp allscriptforQA/singletask_from_multi_qa2qa.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/singletask_from_multi_cls2cls.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/singletask_from_multi_qa2qa.py -r
cp allscriptforQA/singletask_from_multi_noqa2qa.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/singletask_from_multi_nocls2cls.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/singletask_from_multi_noqa2qa.py -r

cp allscriptforQA/singletask_from_reptile_qa2qa.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/singletask_from_reptile_cls2cls.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/singletask_from_reptile_qa2qa.py -r
cp allscriptforQA/singletask_from_reptile_noqa2qa.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/singletask_from_reptile_nocls2cls.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/singletask_from_reptile_noqa2qa.py -r

cp allnewfolders/T5-large-qa2qa /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/models/ -r
cp allnewfolders/T5-large-nonli2nli /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/models/ -r

cp allnewfolders/T5-large-ft-qa2qa /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/models/ -r
cp allnewfolders/T5-large-ft-nonli2nli /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/models/ -r

cp allnewfolders/T5-large-maml-qa2qa-3e-5-2-5000-5e-1 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/ -r
cp allnewfolders/T5-large-maml-noqa2qa-3e-5-2-5000-5e-1 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/ -r
cp allnewfolders/T5-large-maml-nonli2nli-3e-5-2-5000-5e-1 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/ -r

cp allnewfolders/T5-large-fomaml-qa2qa-3e-5-2-5000-5e-1 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/models/ -r
cp allnewfolders/T5-large-fomaml-noqa2qa-3e-5-2-5000-5e-1 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/models/ -r
cp allnewfolders/T5-large-fomaml-nonli2nli-3e-5-2-5000-5e-1 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/models/ -r

cp allnewfolders/T5-large-multitask-qa2qa-5e-1-4-20 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/ -r
cp allnewfolders/T5-large-multitask-noqa2qa-5e-1-4-20 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/ -r
cp allnewfolders/T5-large-multitask-nonli2nli-5e-1-4-20 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/ -r

cp allnewfolders/T5-large-reptile-qa2qa-3e-5-2-5000-5e-1-10 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/models/ -r
cp allnewfolders/T5-large-reptile-noqa2qa-3e-5-2-5000-5e-1-10 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/models/ -r
cp allnewfolders/T5-large-reptile-nonli2nli-3e-5-2-5000-5e-1-10 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/models/ -r

cp allscriptfornli/cli_maml_ddp_prompt_nonli2nli.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/cli_maml_ddp_prompt_tune_nocls2cls.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/cli_maml_ddp_prompt_nonli2nli.py -r

cp allscriptfornli/cli_maml_ddp_prompt_fo_nonli2nli.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/cli_maml_ddp_prompt_fo_nocls2cls.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/cli_maml_ddp_prompt_fo_nonli2nli.py -r

cp allscriptfornli/cli_multitask_ddp_prompt_nonli2nli.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/cli_multitask_ddp_prompt_nocls2cls.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/cli_multitask_ddp_prompt_nonli2nli.py -r

cp allscriptfornli/cli_reptile_ddp_prompt_nonli2nli.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/cli_reptile_ddp_prompt_nocls2cls.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/cli_reptile_ddp_prompt_nonli2nli.py -r

cp allscriptfornli/tune_hps_singletask_ddp_prompt_nonli2nli.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/tune_hps_singletask_ddp_prompt_nocls2cls.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/tune_hps_singletask_ddp_prompt_nonli2nli.py -r

cp allscriptfornli/tune_singletask_nonli2nli.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/tune_singletask_nocls2cls.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/tune_singletask_nonli2nli.py -r

cp allscriptfornli/singletask_from_meta_nonli2nli.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/singletask_from_meta_nocls2cls.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/singletask_from_meta_nonli2nli.py -r

cp allscriptfornli/singletask_from_fomaml_nonli2nli.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/singletask_from_fomaml_nocls2cls.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/singletask_from_fomaml_nonli2nli.py -r

cp allscriptfornli/singletask_from_multi_nonli2nli.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/singletask_from_multi_nocls2cls.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/singletask_from_multi_nonli2nli.py -r

cp allscriptfornli/singletask_from_reptile_nonli2nli.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/singletask_from_reptile_nocls2cls.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/singletask_from_reptile_nonli2nli.py -r
