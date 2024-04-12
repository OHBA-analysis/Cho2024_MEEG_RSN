# This script is used to organize directories and files. Then,
# only the eyes closed segments are extracted from the source 
# reconstructed data (without structurals).

# Copy source reconstructed data
cp -r src_no_struct/ src_ec_no_struct/

# Remove `logs` and `report` directories for memory
rm -r src_ec_no_struct/logs
rm -r src_ec_no_struct/report

# Remove other files for memory
find src_ec_no_struct/ -type d \( -name "rhino" -o -name "bem" -o -name "parc" -o -name "beamform" \) -exec rm -r {} +
find src_ec_no_struct/ -type f ! -name "sflip_parc-raw.fif" -exec rm {} +

# Extract task segments (eyes closed)
python scripts/select_events.py standard &> select_events.log

# Eliminate original data from new directories
rm -r src_ec_no_struct/*/*.fif

echo "Process completed."
