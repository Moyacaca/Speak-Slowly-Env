DIR=$1
zip=$2
cd $DIR
filename="${zip%.*}";
unzip -o ${zip} -d ${filename}; 
# for zip in *.zip;
# 	do
# 		filename="${zip%.*}";
# 		#jar xvf
# 		unzip -o ${zip} -d ${filename}; 
# done
