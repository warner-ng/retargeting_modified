declare -A urdfs
urdfs["h1"]="H1/urdf/h1.urdf"
urdfs["h1_2"]="H1_2/urdf/h1_2.urdf"
urdfs["g1"]="G1/urdf/g1.urdf"
urdfs["orca"]="Orca/urdf/orca.urdf"
urdfs["gr1t2"]="GR1T2/urdf/gr1t2.urdf"

python visualize/visualize.py -urdf "${urdfs[$1]}" \
    -mapping "output/""$1"_data/full_train/joint_id.txt