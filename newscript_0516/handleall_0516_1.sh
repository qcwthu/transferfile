########run scripts to get all results
cd /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/example_scripts
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0518

echo "handle para"
python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/models/T5-large-nopara2para --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0518/allres_pt_para.csv
python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/models/T5-large-ft-nopara2para --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0518/allres_ft_para.csv

python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-nopara2para-3e-5-2-5000-5e-1 --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0518/allres_maml_nopara2para.csv

python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/models/T5-large-fomaml-nopara2para-3e-5-2-5000-5e-1 --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0518/allres_fomaml_nopara2para.csv

python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-large-multitask-nopara2para-5e-1-4-20 --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0518/allres_multi_nopara2para.csv

python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/models/T5-large-reptile-nopara2para-3e-5-2-5000-5e-1-10 --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0518/allres_reptile_nopara2para.csv
cd -
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0518 ../ -r

##eval "$(ssh-agent -s)"
##ssh-add /export/share/sjoty/.ssh/id_rsa

