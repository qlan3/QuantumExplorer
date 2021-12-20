cd logs
files=$(ls)
for filename in $files
do
    printf "zip $filename ...\n"
    zip -r $filename.zip ./$filename/
done
cd ..