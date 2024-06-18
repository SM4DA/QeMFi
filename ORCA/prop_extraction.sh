molecule="urea"
fid="TZVP"

#folder to store all logfiles
mkdir logfiles/

#copy logfile zips to workdir
#general case
cp logfiles.zip logfiles/.

cd $workdir
mkdir -p "${workdir}"/tmp/

for z in logfiles/*.zip; do unzip -q $z -d tmp/;done
echo "Unzipped log files for $fid of $molecule. Extracting all properties."

for i in $(seq 1 100) #To match the example of orca calculation script. Modify as required.
do
	#select correct geom (sequential)
	sample=$(awk -v taskID=$i '$1==taskID {print $2}' single_mol_config.txt)
	logfile=tmp/${molecule}_${sample}.xyz.log
	
	#vertical excitation 
	#energies
	awk '/ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS/,/ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS/ { if ($0 ~ /fosc/) { getline; for (i=0; i<=10; i++) { getline; print $2 } exit } }' $logfile | paste -s -d'\t'>>${fid}_EV.dat
	#osc strength
	awk '/ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS/,/ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS/ { if ($0 ~ /fosc/) { getline; for (i=0; i<=10; i++) { getline; print $4 } exit } }' $logfile | paste -s -d'\t'>>${fid}_fosc.dat
	#transition dipole X
	awk '/ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS/,/ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS/ { if ($0 ~ /fosc/) { getline; for (i=0; i<=10; i++) { getline; print $6 } exit } }' $logfile | paste -s -d'\t'>>${fid}_TX.dat
	#Tr Y
	awk '/ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS/,/ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS/ { if ($0 ~ /fosc/) { getline; for (i=0; i<=10; i++) { getline; print $7 } exit } }' $logfile | paste -s -d'\t'>>${fid}_TY.dat
	#Tr Z
	awk '/ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS/,/ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS/ { if ($0 ~ /fosc/) { getline; for (i=0; i<=10; i++) { getline; print $8 } exit } }' $logfile | paste -s -d'\t'>>${fid}_TZ.dat
	
	#SCF
	grep "E(SCF)" $logfile | awk '{print $3}' >> ${fid}_SCF.dat
	
	#molecular dipole moments
	awk '/Electronic contribution:/ {print $3, $4, $5}' $logfile >> ${fid}_DPe.dat
	awk '/Nuclear contribution   :/ {print $4, $5, $6}' $logfile >> ${fid}_DPn.dat
	
	#rotational constants
	awk '/Rotational constants in cm-1:/ {print $5, $6, $7}' $logfile >> ${fid}_RotConst.dat
	awk '/x,y,z \[a\.u\.\] :/ {print $4, $5, $6}' $logfile >> ${fid}_DPRo.dat
	
done
#zip all properties into one file
zip -j -q props_${molecule}_${fid}.zip *.dat



