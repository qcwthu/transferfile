cp new_files_2nocls/tune_hps_singletask_ddp_prompt_cls2nocls.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/ -r
cp new_files_2nocls/tune_singletask_cls2nocls.sh /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/ -r
cp new_files_2nocls/*meta* /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/ -r
cp new_files_2nocls/*fomaml* /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/ -r
cp new_files_2nocls/*multi* /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/ -r
cp new_files_2nocls/*reptile* /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/ -r

mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/models/T5-large-cls2nocls/singletask-quoref
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/models/T5-large-cls2nocls/singletask-wiki_split
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/models/T5-large-cls2nocls/singletask-ai2_arc
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/models/T5-large-cls2nocls/singletask-break-QDMR
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/models/T5-large-cls2nocls/singletask-crawl_domain
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/models/T5-large-cls2nocls/singletask-samsum

mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/models/T5-large-ft-cls2nocls/singletask-quoref
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/models/T5-large-ft-cls2nocls/singletask-wiki_split
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/models/T5-large-ft-cls2nocls/singletask-ai2_arc
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/models/T5-large-ft-cls2nocls/singletask-break-QDMR
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/models/T5-large-ft-cls2nocls/singletask-crawl_domain
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/models/T5-large-ft-cls2nocls/singletask-samsum



mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-both2nocls-3e-5-2-5000-5e-1/singletask-quoref
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-both2nocls-3e-5-2-5000-5e-1/singletask-wiki_split
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-both2nocls-3e-5-2-5000-5e-1/singletask-ai2_arc
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-both2nocls-3e-5-2-5000-5e-1/singletask-break-QDMR
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-both2nocls-3e-5-2-5000-5e-1/singletask-crawl_domain
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-both2nocls-3e-5-2-5000-5e-1/singletask-samsum

mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2nocls-3e-5-2-5000-5e-1/singletask-quoref
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2nocls-3e-5-2-5000-5e-1/singletask-wiki_split
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2nocls-3e-5-2-5000-5e-1/singletask-ai2_arc
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2nocls-3e-5-2-5000-5e-1/singletask-break-QDMR
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2nocls-3e-5-2-5000-5e-1/singletask-crawl_domain
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-cls2nocls-3e-5-2-5000-5e-1/singletask-samsum

mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-nocls2nocls-3e-5-2-5000-5e-1/singletask-quoref
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-nocls2nocls-3e-5-2-5000-5e-1/singletask-wiki_split
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-nocls2nocls-3e-5-2-5000-5e-1/singletask-ai2_arc
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-nocls2nocls-3e-5-2-5000-5e-1/singletask-break-QDMR
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-nocls2nocls-3e-5-2-5000-5e-1/singletask-crawl_domain
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/T5-large-maml-nocls2nocls-3e-5-2-5000-5e-1/singletask-samsum




mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/models/T5-large-fomaml-both2nocls-3e-5-2-5000-5e-1/singletask-quoref
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/models/T5-large-fomaml-both2nocls-3e-5-2-5000-5e-1/singletask-wiki_split
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/models/T5-large-fomaml-both2nocls-3e-5-2-5000-5e-1/singletask-ai2_arc
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/models/T5-large-fomaml-both2nocls-3e-5-2-5000-5e-1/singletask-break-QDMR
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/models/T5-large-fomaml-both2nocls-3e-5-2-5000-5e-1/singletask-crawl_domain
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/models/T5-large-fomaml-both2nocls-3e-5-2-5000-5e-1/singletask-samsum

mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/models/T5-large-fomaml-cls2nocls-3e-5-2-5000-5e-1/singletask-quoref
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/models/T5-large-fomaml-cls2nocls-3e-5-2-5000-5e-1/singletask-wiki_split
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/models/T5-large-fomaml-cls2nocls-3e-5-2-5000-5e-1/singletask-ai2_arc
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/models/T5-large-fomaml-cls2nocls-3e-5-2-5000-5e-1/singletask-break-QDMR
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/models/T5-large-fomaml-cls2nocls-3e-5-2-5000-5e-1/singletask-crawl_domain
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/models/T5-large-fomaml-cls2nocls-3e-5-2-5000-5e-1/singletask-samsum

mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/models/T5-large-fomaml-nocls2nocls-3e-5-2-5000-5e-1/singletask-quoref
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/models/T5-large-fomaml-nocls2nocls-3e-5-2-5000-5e-1/singletask-wiki_split
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/models/T5-large-fomaml-nocls2nocls-3e-5-2-5000-5e-1/singletask-ai2_arc
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/models/T5-large-fomaml-nocls2nocls-3e-5-2-5000-5e-1/singletask-break-QDMR
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/models/T5-large-fomaml-nocls2nocls-3e-5-2-5000-5e-1/singletask-crawl_domain
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/models/T5-large-fomaml-nocls2nocls-3e-5-2-5000-5e-1/singletask-samsum




mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-large-multitask-both2nocls-5e-1-4-20/singletask-quoref
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-large-multitask-both2nocls-5e-1-4-20/singletask-wiki_split
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-large-multitask-both2nocls-5e-1-4-20/singletask-ai2_arc
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-large-multitask-both2nocls-5e-1-4-20/singletask-break-QDMR
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-large-multitask-both2nocls-5e-1-4-20/singletask-crawl_domain
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-large-multitask-both2nocls-5e-1-4-20/singletask-samsum

mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-large-multitask-cls2nocls-5e-1-4-20/singletask-quoref
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-large-multitask-cls2nocls-5e-1-4-20/singletask-wiki_split
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-large-multitask-cls2nocls-5e-1-4-20/singletask-ai2_arc
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-large-multitask-cls2nocls-5e-1-4-20/singletask-break-QDMR
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-large-multitask-cls2nocls-5e-1-4-20/singletask-crawl_domain
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-large-multitask-cls2nocls-5e-1-4-20/singletask-samsum

mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-large-multitask-nocls2nocls-5e-1-4-20/singletask-quoref
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-large-multitask-nocls2nocls-5e-1-4-20/singletask-wiki_split
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-large-multitask-nocls2nocls-5e-1-4-20/singletask-ai2_arc
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-large-multitask-nocls2nocls-5e-1-4-20/singletask-break-QDMR
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-large-multitask-nocls2nocls-5e-1-4-20/singletask-crawl_domain
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/T5-large-multitask-nocls2nocls-5e-1-4-20/singletask-samsum




mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/models/T5-large-reptile-both2nocls-3e-5-2-5000-5e-1-10/singletask-quoref
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/models/T5-large-reptile-both2nocls-3e-5-2-5000-5e-1-10/singletask-wiki_split
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/models/T5-large-reptile-both2nocls-3e-5-2-5000-5e-1-10/singletask-ai2_arc
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/models/T5-large-reptile-both2nocls-3e-5-2-5000-5e-1-10/singletask-break-QDMR
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/models/T5-large-reptile-both2nocls-3e-5-2-5000-5e-1-10/singletask-crawl_domain
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/models/T5-large-reptile-both2nocls-3e-5-2-5000-5e-1-10/singletask-samsum

mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/models/T5-large-reptile-cls2nocls-3e-5-2-5000-5e-1-10/singletask-quoref
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/models/T5-large-reptile-cls2nocls-3e-5-2-5000-5e-1-10/singletask-wiki_split
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/models/T5-large-reptile-cls2nocls-3e-5-2-5000-5e-1-10/singletask-ai2_arc
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/models/T5-large-reptile-cls2nocls-3e-5-2-5000-5e-1-10/singletask-break-QDMR
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/models/T5-large-reptile-cls2nocls-3e-5-2-5000-5e-1-10/singletask-crawl_domain
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/models/T5-large-reptile-cls2nocls-3e-5-2-5000-5e-1-10/singletask-samsum

mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/models/T5-large-reptile-nocls2nocls-3e-5-2-5000-5e-1-10/singletask-quoref
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/models/T5-large-reptile-nocls2nocls-3e-5-2-5000-5e-1-10/singletask-wiki_split
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/models/T5-large-reptile-nocls2nocls-3e-5-2-5000-5e-1-10/singletask-ai2_arc
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/models/T5-large-reptile-nocls2nocls-3e-5-2-5000-5e-1-10/singletask-break-QDMR
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/models/T5-large-reptile-nocls2nocls-3e-5-2-5000-5e-1-10/singletask-crawl_domain
mkdir /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/models/T5-large-reptile-nocls2nocls-3e-5-2-5000-5e-1-10/singletask-samsum