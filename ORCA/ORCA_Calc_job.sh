

molecule="urea" #name of molecule
fid="TZVP" #fidelity of calculation

echo "Running ORCA for ${fid} fidelity for ${molecule}."
echo "For the ${start}-${end} geoms"

#create the input file
cp $fid.inp input.inp

#copy geometries from some directory
cp ~/geometryfilepath/*.zip .

mkdir geoms/ #directory for geometries
mkdir tmp/ #directory for storing temporary files during orca calculations
mkdir logs/ #directory to store the log files resulting from orca calculations

#unzip geometries to respective folder
unzip -q \*.zip -d geoms/

#pal procs of input file - this means run orca as 64 pal processes
echo "%PAL nproc 64 end" >> input.inp

start=0 #starting geometry number
end=100 #ending geoemtry number - this means orca calculations will run for first 100 geometries

#loop over geometries
for i in $(seq $start $end)
do
	#select correct name of the sample
	sample=$(awk -v taskID=$i '$1==taskID {print $2}' single_mol_config.txt)
	#select corresponding geometry
	geometry="${molecule}_${sample}.xyz"
	
	#copy input and xyz file
	cp input.inp tmp/
	cp geoms/$geometry tmp/$geometry
	
	#add xyzfile info to input file
	echo "* xyzfile 0 1 tmp/${geometry} " >> tmp/input.inp
	
	#run orca
	# it is necessary to use the full path to orca while using openMPI pal processes
	<full_path_to_orca>/orca tmp/input.inp>logs/"${geometry}".log
	
	#clean tmp for next run
	rm tmp/*
done

zip -j -q logs_${molecule}_${fid}_${start}_${end}.zip logs/*.log

cp logs_${molecule}_${fid}_${start}_${end}.zip ~/<path_to_store_logfiles>/.


