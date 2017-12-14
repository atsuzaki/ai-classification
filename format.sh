location=$1

for f in "$@"; do
	cat $f | tr '\n' ' ' | tr '[:upper:]' '[:lower:]' | tr -d '[:punct:]' | tr -d '[:digit:]' >> result
	echo >> result
done
