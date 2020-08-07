
for file in `ls logs*`
do
	echo ' . . . . . . . '
	echo $file

	grep Day $file |grep -v 'Day 26'|awk -F" " '{print $9}'| awk -v field="$1" '{sum+=$field; n++;}END {if (n > 0) print sum/n;else {print 'error' > "/tmp/.stderr"; exit 1};}'

done


