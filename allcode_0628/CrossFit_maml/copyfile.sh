allinnerlr=(2e-5 3e-5 5e-5)
#allinnerlr=(2e-5)
allgradient=(1 2 4)
allstep=(2500 5000 10000)
alloutlr=(2e-1 3e-1 5e-1)
#alloutlr=(2e-1 3e-1)

#CHECKPOINT="models/upstream-maml-noncls2cls/last-model.pt"

#size="base"
size="large"
#IDENTIFIER="T5-"$size-"maml"
#IDENTIFIER="T5-"$size-"fomaml"
for onelr in ${allinnerlr[@]}
do
        for oneg in ${allgradient[@]}
        do
                for onestep in ${allstep[@]}
                do
                        for outerlr in ${alloutlr[@]}
                        do
				#CHECKPOINT="models/upstream-maml-random-"$onelr"-"$oneg"-"$onestep"-"$outerlr
      				#echo "++++++++++++++++++++++++++++++"
				#mkdir "./models/T5-large-maml-"$onelr"-"$oneg"-"$onestep"-"$outerlr
				cp ./models/T5-large-maml-2e-5-1-2500-2e-1/*  "./models/T5-large-maml-"$onelr"-"$oneg"-"$onestep"-"$outerlr/ -r
			 done
                done
        done
done

