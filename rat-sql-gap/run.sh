# ps aux|grep python|grep -v grep|cut -c 9-15|xargs kill -15

python run.py train experiments/sparc-configs/gap-run.jsonnet