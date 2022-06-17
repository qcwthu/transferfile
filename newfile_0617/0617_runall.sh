mkdir alltouse_0617
cd alltouse_0617

###cp collect_results.py
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/example_scripts/collect_results.py ./

###cp alllog
cp /export/share/sjoty/continual-learning/MetaPromptTuning/*.log ./
cp /export/share/sjoty/continual-learning/MetaPromptTuning/*.sh ./

###cp all folders
###t5-base
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/models/*T5-base* ./ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/models/*T5-base* ./ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/*T5-base* ./ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/models/*T5-base* ./ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/*T5-base* ./ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/models/*T5-base* ./ -r


###32 64 128 upstream, maml multi
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/*up32shot* ./ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/*up32shot* ./ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/*up64shot* ./ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/*up64shot* ./ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/*up128shot* ./ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/*up128shot* ./ -r


###32 64 128 downstream, pt ft maml multi
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/models/*down32shot* ./ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/models/*down32shot* ./ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/*down32shot* ./ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/*down32shot* ./ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/models/*down64shot* ./ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/models/*down64shot* ./ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/*down64shot* ./ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/*down64shot* ./ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/models/*down128shot* ./ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/models/*down128shot* ./ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/*down128shot* ./ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/*down128shot* ./ -r

###number of prompt token: 50,200  pt maml multi
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/models/*50prompt* ./ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/*50prompt* ./ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/*50prompt* ./ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/models/*200prompt* ./ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/*200prompt* ./ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/*200prompt* ./ -r

###number of meta-training tasks: maml cls2cls
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/*12uptasks* ./ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/*24uptasks* ./ -r

###noqa2qa maml,验证一下是不是完整
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/*noqa2qa* ./ -r

cd ..