# 上一个pre.sh准备好了srdf
# PushQue将新任务ID加入队列Que

# GenQue清理已完成任务
# ChkQue检查队列中是否有已完成任务，若有则调用GenQue更新队列
# 然后运行main.py
# 然后运行fix_height.py



if [ $1 -le 0 ]; then
    echo "Number of workers (argment 1) should be more than 0!!!"
    exit 1
fi

function PushQue {
    Que="$Que $1"
    Nrun=$(($Nrun+1))
}

function GenQue {
    OldQue=$Que; Que=""; Nrun=0bash
    for PID in $OldQue; do
        if [[ -d /proc/$PID ]]; then
            PushQue $PID
        fi
    done
}

function ChkQue {
    OldQue=$Que
    for PID in $OldQue; do
        if [[ ! -d /proc/$PID ]]; then
            GenQue; break
        fi
    done
}

declare -a folders=(
  "full_train"
  "full_valid"
  "full_test"
#   "run_train"
#   "walk_train"
#   "crawl_train"
)

declare -a robots=(
  "h1"
  "h1_2"
  "g1"
  "orca"
  "gr1t2"
)

declare -A urdfs
urdfs["h1"]="H1/urdf/h1.urdf"
urdfs["h1_2"]="H1_2/urdf/h1_2.urdf"
urdfs["g1"]="G1/urdf/g1.urdf"
urdfs["orca"]="Orca/urdf/orca.urdf"
urdfs["gr1t2"]="GR1T2/urdf/gr1t2.urdf"

declare -A srdfs
srdfs["h1"]="H1/urdf/h1.srdf"
srdfs["h1_2"]="H1_2/urdf/h1_2.srdf"
srdfs["g1"]="G1/urdf/g1.srdf"
srdfs["orca"]="Orca/urdf/orca.srdf"
srdfs["gr1t2"]="GR1T2/urdf/gr1t2.srdf"

Nrun=0
for folder in "${folders[@]}"; do
    for robot in "${robots[@]}"; do
        python src/main.py -path robot -pkg robot -smpl smpl \
            -urdf "${urdfs[$robot]}" -srdf "${srdfs[$robot]}" \
            -yaml config/$robot.yml -data smpl_data/amass_$folder.pkl \
            -output "output/""$robot""_data/$folder" -rate 30 -headless &
        PushQue $!
        while [[ $Nrun -ge $1 ]]; do
            ChkQue; sleep 0.01
        done
    done
done
wait

Nrun=0
for folder in "${folders[@]}"; do
    for robot in "${robots[@]}"; do
        data_folder="output/""$robot""_data/$folder/"
        python src/fix_height.py -urdf "${urdfs[$robot]}" \
            -mapping "$data_folder""joint_id.txt" -folder $data_folder &
        PushQue $!
        while [[ $Nrun -ge $1 ]]; do
            ChkQue; sleep 0.01
        done
    done
done
wait

cd output && mkdir RetargetDataset
for robot in "${robots[@]}"; do
    tar --use-compress-program="pigz -9 -k " -cf "RetargetDataset/""$robot""_data.tar.gz" "$robot""_data/"
done