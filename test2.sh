
for file in `ls logs*`
do
	#echo ' . . . . . . . '
	echo $file

	#grep Day $file |grep -v 'Day 26'|awk -F" " '{print $9}'| awk -v field="$1" '{sum+=$field; n++;}END {if (n > 0) print sum/n;else {print 'error' > "/tmp/.stderr"; exit 1};}'

	grep Day $file |grep -v 'Day 26'|awk -F" " '{print $9}'|awk '
	BEGIN{
}
{
	sum+=$0
}
NR==1 {
max=$1;min=$1
next
}
$1>max {
max=$1
}
$1<min {
min=$1
}
END{
printf "max min average number is:%s, %s, %s \n",max,min,sum/NR
}'


done


