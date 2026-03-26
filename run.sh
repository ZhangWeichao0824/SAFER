for dataset in  Corafull-CL Arxiv-CL 
do
python train.py --dataset $dataset --method safer --backbone GCN --gpu 0 --ILmode classIL --inter-task-edges False --minibatch False
done

for dataset in  Reddit-CL Products-CL 
do
python train.py --dataset $dataset --method safer --backbone GCN --gpu 0 --ILmode classIL --inter-task-edges False --minibatch False
done
