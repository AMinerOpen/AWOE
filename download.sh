mkdir tmp
cd tmp
LANG=$1
FN=keywords_aminer_$LANG.zip
if [ ! -f "$FN" ]; then
    URL=https://lfs.aminer.cn/misc/awoe/$FN
    wget $URL
fi
unzip keywords_aminer_$LANG.zip
rm keywords_aminer_$LANG.zip
cd ..
