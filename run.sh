SCENE=$1
sh colmap.sh data/$SCENE
python run.py --config configs/$SCENE.txt  --no_ndc --spherify --lindisp --expname=$SCENE --lossrate=0.001 --priorback=True
