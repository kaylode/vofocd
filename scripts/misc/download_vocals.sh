mkdir data
mkdir data/aim
cd data/aim

/bin/bash download_from_drive.sh 1oq1UDJfVRXOR8S9frbsEeg7Trmi_i5zL annotation_4cls.json
/bin/bash download_from_drive.sh 1lqt9csAJGveWTRHMs6LIqEMesIDrLFpR annotation_5cls.json
/bin/bash download_from_drive.sh 1xQAONKOKHwl1UV7JngBw4eJvj2fKxywx Train4classes.zip
/bin/bash download_from_drive.sh 1stv13UMd0N-4vPPvR5tFSosA2TofvKmQ Train5classes.zip

unzip Train4classes.zip
unzip Train5classes.zip