# keras-learnings

Install the requirements.txt if needed.

Activate the virtual env:
```
source env/bin/activate
```

Train like this:
```bash
python train_simple_nn.py --dataset animals --model output/simple_nn.model \
--label-bin output/simple_nn_lb.pickle --plot output/simple_nn_plot.png
```

Run like this:
```bash
python predict.py --image hoop2.jpg --model output/simple_nn.model \
--label-bin output/simple_nn_lb.pickle --width 32 --height 32 --flatten 1
```