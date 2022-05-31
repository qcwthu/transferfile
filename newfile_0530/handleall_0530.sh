#######handleall_0530
###dont forget to modify load data
###dont forget to delete json
###dont forget to get the result of para

#delete json
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/data_32/*.json
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/data_64/*.json
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/data_128/*.json

rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/data_32/*.json
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/data_64/*.json
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/data_128/*.json

rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_32/*.json
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_64/*.json
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_128/*.json

rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/data_32/*.json
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/data_64/*.json
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/data_128/*.json

rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data_32/*.json
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data_64/*.json
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data_128/*.json

rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/data_32/*.json
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/data_64/*.json
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/data_128/*.json

####get para res
cd /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/example_scripts
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0531
echo "handle nopara"

python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/models/T5-large-nopara2para --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0531/allres_pt_para.csv
python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/models/T5-large-ft-nopara2para --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0531/allres_ft_para.csv

python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-nopara2para-3e-5-2-5000-5e-1 --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0531/allres_maml_nopara2para.csv

python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/models/T5-large-fomaml-nopara2para-3e-5-2-5000-5e-1 --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0531/allres_fomaml_nopara2para.csv

python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-large-multitask-nopara2para-5e-1-4-20 --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0531/allres_multi_nopara2para.csv

python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/models/T5-large-reptile-nopara2para-3e-5-2-5000-5e-1-10 --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0531/allres_reptile_nopara2para.csv
cd -
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0531 ../ -r

####copy data
cp task_128/fewshot_gym_dataset.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/task_128/
cp task_128/fewshot_gym_dataset.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/task_128/
cp task_128/fewshot_gym_dataset.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/task_128/
cp task_128/fewshot_gym_dataset.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/task_128/
cp task_128/fewshot_gym_dataset.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/task_128/
cp task_128/fewshot_gym_dataset.py /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/task_128/

cp data_128/superglue-cb /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/data_128/ -r
cp data_128/superglue-cb /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/data_128/ -r
cp data_128/superglue-cb /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data_128/ -r
cp data_128/superglue-cb /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/data_128/ -r
cp data_128/superglue-cb /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data_128/ -r
cp data_128/superglue-cb /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/data_128/ -r


#######copy all files
cp /export/share/sjoty/continual-learning/MetaPromptTuning/cuda_01.sh /export/share/sjoty/continual-learning/MetaPromptTuning/cuda_01_0531bak.sh
cp /export/share/sjoty/continual-learning/MetaPromptTuning/cuda_45.sh /export/share/sjoty/continual-learning/MetaPromptTuning/cuda_45_0531bak.sh
cp /export/share/sjoty/continual-learning/MetaPromptTuning/cuda_67.sh /export/share/sjoty/continual-learning/MetaPromptTuning/cuda_67_0531bak.sh

cp cuda_01.sh /export/share/sjoty/continual-learning/MetaPromptTuning/
cp cuda_45.sh /export/share/sjoty/continual-learning/MetaPromptTuning/
cp cuda_67.sh /export/share/sjoty/continual-learning/MetaPromptTuning/

cp singletask_from_meta_cls2cls_down128shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/
cp singletask_from_meta_cls2cls_down32shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/
cp singletask_from_meta_cls2cls_down64shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/
cp singletask_from_multi_cls2cls_down128shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/
cp singletask_from_multi_cls2cls_t5base.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/
cp singletask_from_reptile_cls2cls_t5base.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/
cp tune_hps_singletask_ddp_prompt_cls2cls_down128shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/
cp tune_hps_singletask_ddp_prompt_cls2cls_down32shot.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/
cp tune_hps_singletask_ddp_prompt_cls2cls_t5base.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/