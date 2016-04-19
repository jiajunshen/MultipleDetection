import numpy as np
from CNNForMnist import load_data
import os

def selectSamples(examples, nSamples):
    nExamples = examples.shape[0]
    samples = []
    for i in range(nSamples):
        samples.append(examples[np.random.randint(0, nExamples)])
    return samples

def placeDistractions(config, examples):
    distractors = selectSamples(examples, config['num_dist'])
    dist_w = config['dist_w']
    megapatch_w = config['megapatch_w']
    patch = np.zeros((1, megapatch_w, megapatch_w))
    for d_patch in distractors:
        t_y = np.random.randint(megapatch_w - dist_w + 1)
        t_x = np.random.randint(megapatch_w - dist_w + 1)
        s_y = np.random.randint(d_patch.shape[1] - dist_w + 1)
        s_x = np.random.randint(d_patch.shape[2] - dist_w + 1)
        patch[0, t_y:t_y + dist_w, t_x:t_x + dist_w] += d_patch[0, s_y:s_y+dist_w, s_x:s_x+dist_w]
    patch[patch > 1] = 1
    return patch


def placeSpriteRandomly(obs, sprite, boarder):
    h = obs.shape[1]
    w = obs.shape[2]
    spriteH = sprite.shape[1]
    spriteW = sprite.shape[2]
    print(h, w, spriteH, spriteW)
    y = np.random.randint(boarder, h - spriteH - boarder + 1)
    x = np.random.randint(boarder, w - spriteW - boarder + 1)
    obs[:, y:y+spriteH, x:x+spriteW] = obs[:, y:y+spriteH, x:x+spriteW] + sprite
    obs[obs > 1] = 1
    obs[obs < 0] = 0

    return obs


def updateConfig(config, extraConfig):
    if extraConfig != None:
        for key, value in extraConfig:
            config[key] = value
    return config

def createData(extraConfig = None):
    config = {
        'x_train_path': "/X_train.npy",
        'y_train_path': "/Y_train.npy",
        'x_test_path': "/X_test.npy",
        'y_test_path': "/Y_test.npy",
        'megapatch_w': 28,
        'num_dist': 1,
        'dist_w': 10,
        'boarder': 0,
        'nDigits': 0,
        'nClasses': 10,
    }
    config = updateConfig(config, extraConfig)
    X_train, y_train, X_test, y_test = load_data(config['x_train_path'], config['y_train_path'], config['x_test_path'], config['y_test_path'])
    nExamples = X_train.shape[0]
    obs = np.zeros((nExamples, config['megapatch_w'], config['megapatch_w']))
    step = nExamples
    obs = placeDistractions(config, X_train)
    perm = np.arange(nExamples)
    for i in range(config['nDigits']):
        step = step + 1
        if step > nExamples:
            np.random.permutation(perm)
            step = 1

        sprite = X_train[perm[step]] 
        obs = placeSpriteRandomly(obs, sprite, config['boarder'])
        selectedDigit = y_train[perm[step]]

    return obs

