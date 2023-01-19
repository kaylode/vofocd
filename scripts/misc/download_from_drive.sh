FILE_ID=$1
OUTPATH=$2

GGDRIVE_PATH = "'https://docs.google.com/uc?export=download&id=${FILE_ID}'"

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate ${GGDRIVE_PATH} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILE_ID}" -O $OUTPATH && rm -rf /tmp/cookies.txt