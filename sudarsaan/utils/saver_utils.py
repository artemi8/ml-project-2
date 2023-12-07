import numpy as np
import pickle



def save_normalizer(path, Xnormalizer, Ynormalizer, task='regression'):
    
    
    name_list = ['scalerTrainX', 'scalerTrainY']
    
    if task == 'classification':
        print('Saving for classification task')
        with open(f'{path}/{name_list[0]}.pkl','wb') as f:
            pickle.dump(Xnormalizer, f)
    
    
    else:
        normalilzer_list = [Xnormalizer, Ynormalizer]
        print('Saving for regression task\n')
        print('Caution, saving normalizer in order!.. Check arguments')
        for nlz, name in zip(normalilzer_list, name_list):
            with open(f'{path}/{name}.pkl','wb') as f:
                pickle.dump(nlz, f)
        
def load_normalizer(path, task='regression'):
    
    
    name_list = ['scalerTrainX', 'scalerTrainY']
    
    if task == 'classification':
        print('Loading for classification task')
        scalerTrainX = None
        with open(f'{path}/{name_list[0]}.pkl','rb') as f:
            scalerTrainX = pickle.load(f)
        return scalerTrainX
        
    else:
        print('Loading for regression task\n')
        print('Caution, loading normalizer in order!.. Check arguments')
        name_list = ['scalerTrainX', 'scalerTrainY']
        normalilzer_list = []

        for name in name_list:
            with open(f'{path}/{name}.pkl','rb') as f:
                normalilzer_list.append(pickle.load(f))

        return normalilzer_list[0], normalilzer_list[1]
        
def save_dataset(xTrain, yTrain, xTest, yTest, path):
    print('Caution, saving dataset in order!.. Check arguments')
    data_list = [xTrain, yTrain, xTest, yTest]
    data_name = ['xTrain', 'yTrain', 'xTest', 'yTest']
    
    for arr, name in zip(data_list, data_name):
        with open(f'{path}/{name}.npy', 'wb') as f:
            np.save(f,arr)

def load_dataset(path):
    
    print('Caution, loading dataset in order!.. Check arguments')
    
    data_name = ['xTrain', 'yTrain', 'xTest', 'yTest']      
    data_list = []
    for name in data_name:
        with open(f'{path}/{name}.npy', 'rb') as f:
            data_list.append(np.load(f))
            
    return data_list[0], data_list[1], data_list[2], data_list[3]
            