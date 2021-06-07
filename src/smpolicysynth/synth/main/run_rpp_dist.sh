for i in {0..1}
do
	echo "$i"
	python3 run_rpp_diff_dist.py "${i}" &> "out/rpp_dist6_$i.txt" 
done
	 
