for i in {0..5}
do
	for b in "car" "pen" "cp" "mc" "acrobot" "quad" "quadpo"
	do
		echo "$b $i"
		python3 direct_opt.py "${b}_${i}_1_1" &> "out/diropt_${b}${i}.txt" 
	done
done
	 
