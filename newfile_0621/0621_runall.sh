mkdir alltouse_0621
cd alltouse_0621

###cp alllog
cp /export/share/sjoty/continual-learning/MetaPromptTuning/*.log ./
cp /export/share/sjoty/continual-learning/MetaPromptTuning/*.sh ./

###cp all folders


###32 64 128 upstream, maml multi
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/*up64shot* ./ -r
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/*up128shot* ./ -r


###number of prompt token: 50,200  pt maml multi
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/*200prompt* ./ -r

###number of meta-training tasks: maml cls2cls
cp /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/*24uptasks* ./ -r

###base nopara2para
cp t5base_nopara2para/allfolders/T5-base-nopara2para /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit/models/ -r
cp t5base_nopara2para/allfolders/T5-base-ft-nopara2para /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFitFT/models/ -r
cp t5base_nopara2para/allfolders/T5-base-maml-nopara2para-3e-5-2-5000-5e-1 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/models/ -r
cp t5base_nopara2para/allfolders/T5-base-fomaml-nopara2para-3e-5-2-5000-5e-1 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_fomaml/models/ -r
cp t5base_nopara2para/allfolders/T5-base-multitask-nopara2para-5e-1-4-20 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_multi/models/ -r
cp t5base_nopara2para/allfolders/T5-base-reptile-nopara2para-3e-5-2-5000-5e-1-10 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_reptile/models/ -r

cd ..