# Project for RL: Pong Learning using DQN networks

## Instructions

Before we start we have to install the weights and biases package (we can disable the synchronization with the online service using command line tools). Run

```
pip install wandb
```


If you want to run the algorithm locally, you can use the following command but have to create a folder called 

```
python train.py --env Pong-v0 --local True
```

If you want to run the code with sync to the wandb settings run

```
python train.py --env Pong-v0 --logwandb True
```

Per default it is assumed that the model is run on colab (see instructions below) and no sync to wandb is run.

In order to run the code using the colab enviornment use the provided file 
`colab-runner.ipynb` and import it as a jupyter notebook as a new project. Then uplaod all other files in the folder `files` to the colab runtime via file upload on the left hand side of the enviornment. The models have to be saved on your google drive so they don't get lost when the runtime dies so you have to mount your drive in the first step and add you API key. You also need a folder `models` in your personal drive which you connect. Per default, the best model and the latest model is saved in the folder and always overwritten after the evaluation period is over.

## Results

These results have been obtained so far:


| Model | Best Episode | Mean return |
| ------| ------------ | ------ |
| Initial | 725 | 6.6 |
| ? | ? | ? |

For reference: Human perofmance is 9.8 and DQN reference implemtation achieves a score of 18.2

## Future work



The following ideas could be implemented to improve the code:

- Use the same action 4 times could make the training more stable 
- ... (add your ideas here)
