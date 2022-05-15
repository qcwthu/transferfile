########first, we delete the test file of upstream tasks for 32, 64 and 128
allname=("superglue-rte" "tweet_eval-sentiment" "discovery" "glue-rte" "superglue-wsc"
"scicite" "glue-mrpc" "tweet_eval-stance_hillary" "tweet_eval-offensive" "emotion"
"hatexplain" "glue-cola" "sick" "paws" "ethos-sexual_orientation"
"glue-qqp" "tweet_eval-emotion" "sms_spam" "health_fact" "glue-mnli"
"imdb" "ethos-disability" "glue-wnli" "scitail" "trec-finegrained"
"yahoo_answers_topics" "liar" "glue-sst2" "tweet_eval-stance_abortion" "circa"
"tweet_eval-stance_climate" "glue-qnli" "tweet_eval-emoji" "ethos-directed_vs_generalized" "ade_corpus_v2-classification"
"ag_news" "hate_speech_offensive" "superglue-wic" "google_wellformed_query" "tweet_eval-irony"
"ethos-gender" "onestop_english" "trec" "rotten_tomatoes" "kilt_fever")
for name in ${allname[@]}
do
  echo $name
  rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_32/$name/*test*
  rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_64/$name/*test*
  rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_128/$name/*test*
done
########first, we copy new data to different folders
#cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_32 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit -r
#cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_64 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit -r
#cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_128 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit -r
#
#cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_32 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT -r
#cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_64 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT -r
#cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_128 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT -r
#
#cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_32 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml -r
#cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_64 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml -r
#cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_128 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml -r
#
#cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_32 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi -r
#cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_64 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi -r
#cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_128 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi -r
#
#cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_32 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile -r
#cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_64 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile -r
#cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_128 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile -r

########we delete json files
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/data/*/*.json
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/data/*/*.json
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data/*/*.json
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/data/*/*.json
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data/*/*.json
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/data/*/*.json

#########handle nopara2para
###upstream
cp nopara2para/*.json /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/dataloader/custom_tasks_splits/
cp nopara2para/*.json /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/dataloader/custom_tasks_splits/
cp nopara2para/*.json /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/dataloader/custom_tasks_splits/
cp nopara2para/*.json /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/dataloader/custom_tasks_splits/
cp nopara2para/*.json /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/dataloader/custom_tasks_splits/
cp nopara2para/*.json /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/dataloader/custom_tasks_splits/

cp nopara2para/cli_maml_ddp_prompt_nopara2para.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/
cp nopara2para/cli_maml_ddp_prompt_fo_nopara2para.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/
cp nopara2para/cli_multitask_ddp_prompt_nopara2para.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/
cp nopara2para/cli_reptile_ddp_prompt_nopara2para.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/

cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/cli_maml_ddp_prompt_nonli2nli.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/cli_maml_ddp_prompt_nopara2para.py
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/cli_maml_ddp_prompt_fo_nonli2nli.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/cli_maml_ddp_prompt_fo_nopara2para.py
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/cli_multitask_ddp_prompt_nonli2nli.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/cli_multitask_ddp_prompt_nopara2para.py
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/cli_reptile_ddp_prompt_nonli2nli.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/cli_reptile_ddp_prompt_nopara2para.py

###downstream
cp nopara2para/tune_hps_singletask_ddp_prompt_nopara2para.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/
cp nopara2para/tune_singletask_nopara2para.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/
cp nopara2para/singletask_from_meta_nopara2para.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/
cp nopara2para/singletask_from_fomaml_nopara2para.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/
cp nopara2para/singletask_from_multi_nopara2para.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/
cp nopara2para/singletask_from_reptile_nopara2para.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/

cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/tune_hps_singletask_ddp_prompt_nonli2nli.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/tune_hps_singletask_ddp_prompt_nopara2para.py
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/tune_singletask_nonli2nli.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/tune_singletask_nopara2para.py
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/singletask_from_meta_nonli2nli.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/singletask_from_meta_nopara2para.py
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/singletask_from_fomaml_nonli2nli.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/singletask_from_fomaml_nopara2para.py
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/singletask_from_multi_nonli2nli.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/singletask_from_multi_nopara2para.py
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/singletask_from_reptile_nonli2nli.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/singletask_from_reptile_nopara2para.py

###copy folder for downstream
cp nopara2para/allfolders/T5-large-nopara2para /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/ -r
cp nopara2para/allfolders/T5-large-ft-nopara2para /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/ -r
cp nopara2para/allfolders/T5-large-maml-nopara2para-3e-5-2-5000-5e-1 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/ -r
cp nopara2para/allfolders/T5-large-fomaml-nopara2para-3e-5-2-5000-5e-1 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/ -r
cp nopara2para/allfolders/T5-large-multitask-nopara2para-5e-1-4-20 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/ -r
cp nopara2para/allfolders/T5-large-reptile-nopara2para-3e-5-2-5000-5e-1-10 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/ -r

####scripts to run
cp cuda_01.sh cuda_23.sh cuda_45.sh cuda_67.sh /export/share/sjoty/continual-learning/MetaPromptTuning/


########Does it help if we have more labelled data for upstream tasks
###upstream
cp 32-shot/cli_maml_ddp_prompt_cls2cls_32shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/
cp 64-shot/cli_maml_ddp_prompt_cls2cls_64shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/
cp 128-shot/cli_maml_ddp_prompt_cls2cls_128shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/

cp 32-shot/cli_multitask_ddp_prompt_cls2cls_32shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/
cp 64-shot/cli_multitask_ddp_prompt_cls2cls_64shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/
cp 128-shot/cli_multitask_ddp_prompt_cls2cls_128shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/

###downstream
cp 32-shot/singletask_from_meta_cls2cls_up32shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/
cp 64-shot/singletask_from_meta_cls2cls_up64shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/
cp 128-shot/singletask_from_meta_cls2cls_up128shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/

cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot -r
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot/*/*pt
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot/*/*csv
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot/*/*txt
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up64shot -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up128shot -r

cp 32-shot/singletask_from_multi_cls2cls_up32shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/
cp 64-shot/singletask_from_multi_cls2cls_up64shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/
cp 128-shot/singletask_from_multi_cls2cls_up128shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/

cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-large-multitask-cls2cls-5e-1-4-20-up32shot -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-large-multitask-cls2cls-5e-1-4-20-up64shot -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-large-multitask-cls2cls-5e-1-4-20-up128shot -r



########downstream Few-Shot to More-Shot, we don't need re-train upstream
###pt ft maml multi
cp 32-shot/tune_hps_singletask_ddp_prompt_cls2cls_down32shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/
cp 32-shot/tune_singletask_cls2cls_down32shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/
cp 32-shot/singletask_from_meta_cls2cls_down32shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/
cp 32-shot/singletask_from_multi_cls2cls_down32shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/

cp 64-shot/tune_hps_singletask_ddp_prompt_cls2cls_down64shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/
cp 64-shot/tune_singletask_cls2cls_down64shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/
cp 64-shot/singletask_from_meta_cls2cls_down64shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/
cp 64-shot/singletask_from_multi_cls2cls_down64shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/

cp 128-shot/tune_hps_singletask_ddp_prompt_cls2cls_down128shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/
cp 128-shot/tune_singletask_cls2cls_down128shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/
cp 128-shot/singletask_from_meta_cls2cls_down128shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/
cp 128-shot/singletask_from_multi_cls2cls_down128shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/

cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/models/T5-large-cls2cls-down32shot -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/models/T5-large-ft-cls2cls-down32shot -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-down32shot -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-large-multitask-cls2cls-5e-1-4-20-down32shot -r

cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/models/T5-large-cls2cls-down64shot -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/models/T5-large-ft-cls2cls-down64shot -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-down64shot -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-large-multitask-cls2cls-5e-1-4-20-down64shot -r

cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/models/T5-large-cls2cls-down128shot -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/models/T5-large-ft-cls2cls-down128shot -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-down128shot -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-large-multitask-cls2cls-5e-1-4-20-down128shot -r




########different backbone model
cp t5base/tune_singletask_cls2cls_t5base.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/
cp t5base/tune_hps_singletask_ddp_prompt_cls2cls_t5base.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/
cp t5base/cli_maml_ddp_prompt_cls2cls_t5base.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/
cp t5base/singletask_from_meta_cls2cls_t5base.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/
cp t5base/cli_maml_ddp_prompt_fo_cls2cls_t5base.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/
cp t5base/singletask_from_fomaml_cls2cls_t5base.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/
cp t5base/cli_multitask_ddp_prompt_cls2cls_t5base.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/
cp t5base/singletask_from_multi_cls2cls_t5base.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/
cp t5base/cli_reptile_ddp_prompt_cls2cls_t5base.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/
cp t5base/singletask_from_reptile_cls2cls_t5base.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/

cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/models/T5-base-ft-cls2cls -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/models/T5-base-cls2cls -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-base-maml-cls2cls-3e-5-2-5000-5e-1 -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/models/T5-base-fomaml-cls2cls-3e-5-2-5000-5e-1 -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-base-multitask-cls2cls-5e-1-4-20 -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/models/T5-base-reptile-cls2cls-3e-5-2-5000-5e-1-10 -r




