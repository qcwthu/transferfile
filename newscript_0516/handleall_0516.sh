########first, we copy failed data to CrossFit_maml
cp ../data_0515/data_32/* /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_32/ -r
cp ../data_0515/data_64/* /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_64/ -r
cp ../data_0515/data_128/* /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_128/ -r

########we delete the test file of upstream tasks for 32, 64 and 128
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
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_32 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_64 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_128 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit -r

cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_32 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_64 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_128 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT -r

cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_32 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_64 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_128 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml -r

cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_32 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_64 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_128 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi -r

cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_32 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_64 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_128 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile -r




########different number of prompt tokens
cp promptnum/50/tune_hps_singletask_ddp_prompt_cls2cls_50prompt.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit
cp promptnum/50/cli_maml_ddp_prompt_cls2cls_50prompt.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml
cp promptnum/50/singletask_from_meta_cls2cls_50prompt.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml
cp promptnum/50/cli_multitask_ddp_prompt_cls2cls_50prompt.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi
cp promptnum/50/singletask_from_multi_cls2cls_50prompt.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi

cp promptnum/200/tune_hps_singletask_ddp_prompt_cls2cls_200prompt.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit
cp promptnum/200/cli_maml_ddp_prompt_cls2cls_200prompt.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml
cp promptnum/200/singletask_from_meta_cls2cls_200prompt.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml
cp promptnum/200/cli_multitask_ddp_prompt_cls2cls_200prompt.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi
cp promptnum/200/singletask_from_multi_cls2cls_200prompt.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi

cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/models/T5-large-cls2cls-50prompt -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-50prompt -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-large-multitask-cls2cls-5e-1-4-20-50prompt -r

cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/models/T5-large-cls2cls-200prompt -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-200prompt -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-large-multitask-cls2cls-5e-1-4-20-200prompt -r

cp cuda_new_01.sh cuda_new_45.sh cuda_new_67.sh /export/share/sjoty/continual-learning/MetaPromptTuning/

########change number of meta-training tasks
cp changeupnum/train_classification_test_classification_12uptasks.json /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/dataloader/custom_tasks_splits/
cp changeupnum/train_classification_test_classification_24uptasks.json /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/dataloader/custom_tasks_splits/
cp changeupnum/cli_maml_ddp_prompt_cls2cls_12uptasks.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/
cp changeupnum/cli_maml_ddp_prompt_cls2cls_24uptasks.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/
cp changeupnum/singletask_from_meta_cls2cls_12uptasks.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/
cp changeupnum/singletask_from_meta_cls2cls_24uptasks.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/

cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-12uptasks -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-up32shot /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2cls-3e-5-2-5000-5e-1-24uptasks -r


