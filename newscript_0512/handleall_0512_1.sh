########run new experiments
cp task_32 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/ -r
cp task_64 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/ -r
cp task_128 /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml/ -r

cd /export/share/sjoty/continual-learning/MetaPromptTuning/CrossFit_maml
cd task_32
python _build_gym.py --build --n_proc=10
cd ..

cd task_64
python _build_gym.py --build --n_proc=10
cd ..

cd task_128
python _build_gym.py --build --n_proc=10
cd ..