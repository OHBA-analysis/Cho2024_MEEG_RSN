# This script is used to organize directories and files. Then,
# only the eyes closed segments are extracted from the preprocessed
# and source reconstructed data.

# Copy source reconstructed data
cp -r preproc/ preproc_ec/
cp -r src/ src_ec/

# Remove `logs` and `report` directories for memory
rm -r *ec/logs
rm -r *ec/report

# Remove other files for memory
find preproc_ec/ -type f ! -name "*_preproc_raw.fif" -exec rm {} +
find src_ec/ -type d \( -name "rhino" -o -name "bem" -o -name "parc" -o -name "beamform" \) -exec rm -r {} +
find src_ec/ -type f ! -name "sflip_parc-raw.fif" -exec rm {} +

# Extract task segments (eyes closed)
python scripts/select_events.py subject &> select_events.log

# Eliminate original data from new directories
rm -r *ec/*/*.fif

echo "Process completed."
