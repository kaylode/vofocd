mkdir data
mkdir data/aim

/bin/bash scripts/misc/download_from_drive.sh 1oq1UDJfVRXOR8S9frbsEeg7Trmi_i5zL data/aim/annotation_4cls.json
/bin/bash scripts/misc/download_from_drive.sh 1lqt9csAJGveWTRHMs6LIqEMesIDrLFpR data/aim/annotation_5cls.json
/bin/bash scripts/misc/download_from_drive.sh 1xQAONKOKHwl1UV7JngBw4eJvj2fKxywx data/aim/Train4classes.zip
/bin/bash scripts/misc/download_from_drive.sh 1stv13UMd0N-4vPPvR5tFSosA2TofvKmQ data/aim/Train5classes.zip

unzip data/aim/Train4classes.zip -d data/aim/
unzip data/aim/Train5classes.zip -d data/aim/