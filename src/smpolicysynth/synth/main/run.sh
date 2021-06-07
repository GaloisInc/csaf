for i in {1..4}
do
	for b in "car" "pen" "cartpole" "mc" "acrobot" "quad" "quadpo"
	do
		echo "$b $i"
		python3 synth.py "${b}_${i}_1_1" &> "out/$b$i.txt" 
	done
done
	 
