import shlex, subprocess, sys

cmd = "python main.py --train --test --pseudolabels 100 --pseudolabel_trials 3 --pseudolabel_interval 600 --augment --lr 0.001 --kl_scale 0.45 --seq_length 50 --train_iters 5000"
cmd = shlex.split(cmd) + sys.argv[1:]

p = subprocess.Popen(cmd)
exit(p.wait())
