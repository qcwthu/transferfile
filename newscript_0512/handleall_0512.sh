########first, we delete json files
#rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/data/*/*.json
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/data/*/*.json
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/data/*/*.json
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/data/*/*.json
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/data/*/*.json
rm -f /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/data/*/*.json

########run scripts to get all results
cd /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/example_scripts
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0512
echo "handle nocls"
python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/models/T5-large-cls2nocls --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0512/allres_pt_nocls.csv
python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/models/T5-large-ft-cls2nocls --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0512/allres_ft_nocls.csv

python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-both2nocls-3e-5-2-5000-5e-1 --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0512/allres_maml_both2nocls.csv
python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2nocls-3e-5-2-5000-5e-1 --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0512/allres_maml_cls2nocls.csv
python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-nocls2nocls-3e-5-2-5000-5e-1 --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0512/allres_maml_nocls2nocls.csv

python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/models/T5-large-fomaml-both2nocls-3e-5-2-5000-5e-1 --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0512/allres_fomaml_both2nocls.csv
python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/models/T5-large-fomaml-cls2nocls-3e-5-2-5000-5e-1 --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0512/allres_fomaml_cls2nocls.csv
python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/models/T5-large-fomaml-nocls2nocls-3e-5-2-5000-5e-1 --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0512/allres_fomaml_nocls2nocls.csv

python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-large-multitask-both2nocls-5e-1-4-20 --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0512/allres_multi_both2nocls.csv
python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-large-multitask-cls2nocls-5e-1-4-20 --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0512/allres_multi_cls2nocls.csv
python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-large-multitask-nocls2nocls-5e-1-4-20 --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0512/allres_multi_nocls2nocls.csv

python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/models/T5-large-reptile-both2nocls-3e-5-2-5000-5e-1-10 --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0512/allres_reptile_both2nocls.csv
python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/models/T5-large-reptile-cls2nocls-3e-5-2-5000-5e-1-10 --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0512/allres_reptile_cls2nocls.csv
python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/models/T5-large-reptile-nocls2nocls-3e-5-2-5000-5e-1-10 --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0512/allres_reptile_nocls2nocls.csv

echo "handle qa"
python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/models/T5-large-qa2qa --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0512/allres_pt_qa.csv
python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/models/T5-large-ft-qa2qa --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0512/allres_ft_qa.csv

python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-qa2qa-3e-5-2-5000-5e-1 --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0512/allres_maml_qa2qa.csv
python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-noqa2qa-3e-5-2-5000-5e-1 --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0512/allres_maml_noqa2qa.csv

python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/models/T5-large-fomaml-qa2qa-3e-5-2-5000-5e-1 --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0512/allres_fomaml_qa2qa.csv
python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/models/T5-large-fomaml-noqa2qa-3e-5-2-5000-5e-1 --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0512/allres_fomaml_noqa2qa.csv

python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-large-multitask-qa2qa-5e-1-4-20 --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0512/allres_multi_qa2qa.csv
python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-large-multitask-noqa2qa-5e-1-4-20 --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0512/allres_multi_noqa2qa.csv

python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/models/T5-large-reptile-qa2qa-3e-5-2-5000-5e-1-10 --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0512/allres_reptile_qa2qa.csv
python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/models/T5-large-reptile-noqa2qa-3e-5-2-5000-5e-1-10 --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0512/allres_reptile_noqa2qa.csv

echo "handle nli"
python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/models/T5-large-nonli2nli --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0512/allres_pt_nli.csv
python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/models/T5-large-ft-nonli2nli --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0512/allres_ft_nli.csv

python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-nonli2nli-3e-5-2-5000-5e-1 --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0512/allres_maml_nonli2nli.csv

python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/models/T5-large-fomaml-nonli2nli-3e-5-2-5000-5e-1 --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0512/allres_fomaml_nonli2nli.csv

python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-large-multitask-nonli2nli-5e-1-4-20 --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0512/allres_multi_nonli2nli.csv

python collect_results.py --logs_dir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/models/T5-large-reptile-nonli2nli-3e-5-2-5000-5e-1-10 --output_file /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0512/allres_reptile_nonli2nli.csv
cd -
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/allres_0512 ../ -r

##eval "$(ssh-agent -s)"
##ssh-add /export/share/sjoty/.ssh/id_rsa

