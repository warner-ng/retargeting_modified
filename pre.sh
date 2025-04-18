# 实现的效果如下：
# python scripts/make_srdf.py -name "h1" -path "robot/H1/urdf"
# python scripts/make_srdf.py -name "h1_2" -path "robot/H1_2/urdf"
# python scripts/make_srdf.py -name "g1" -path "robot/G1/urdf"
# python scripts/make_srdf.py -name "orca" -path "robot/Orca/urdf"
# python scripts/make_srdf.py -name "gr1t2" -path "robot/GR1T2/urdf"


if [ "$1" == "extract" ]; then
    cd data && ls *.tar.bz2 | xargs -n1 tar jxvf && cd ../
fi

python scripts/process_amass_raw.py
python scripts/pacer_dataset.py
python scripts/full_dataset.py

declare -a robots=(
  "h1"
  "h1_2"
  "g1"
  "orca"
  "gr1t2"
  "grmini1t2"
)

declare -A paths
paths["h1"]="H1/urdf"
paths["h1_2"]="H1_2/urdf"
paths["g1"]="G1/urdf"
paths["orca"]="Orca/urdf"
paths["gr1t2"]="GR1T2/urdf"
paths["grmini1t2"]="GRMini1T2/urdf"

for robot in "${robots[@]}"; do
  python scripts/make_srdf.py -name "$robot" -path "robot/""${paths[$robot]}"
done