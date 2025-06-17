nohup python -u main.py -t 1 -jr 1 -nc 20 -nb 100 -data Cifar100_dir -lam 5 -m cnn -algo FedMB -di
d 0 > ../result/s_FedMB_Cifar100_dir0.01_lam5_hb.out 2>&1 &