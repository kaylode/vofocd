mkdir data
mkdir data/aim
cd data/aim

gdown 1oq1UDJfVRXOR8S9frbsEeg7Trmi_i5zL -O annotation_4cls.json
gdown 1lqt9csAJGveWTRHMs6LIqEMesIDrLFpR -O annotation_5cls.json
gdown 1xQAONKOKHwl1UV7JngBw4eJvj2fKxywx -O Train4classes.zip
gdown 1stv13UMd0N-4vPPvR5tFSosA2TofvKmQ -O Train5classes.zip

unzip Train4classes.zip
unzip Train5classes.zip