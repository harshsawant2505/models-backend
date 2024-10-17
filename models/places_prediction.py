import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_places, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_places)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_places = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_places)
        ratings[id_places - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

# def recommend_places(user_id, sae, places_df, training_set, nb_places):
#     input_data = training_set[user_id - 1].unsqueeze(0)
#     with torch.no_grad():
#         predicted_ratings = sae(input_data)
#     predicted_ratings = predicted_ratings.numpy().flatten()
#     user_ratings = training_set[user_id - 1].numpy()
#     unrated_places = np.where(user_ratings == 0)[0]
#     recommended_place_indices = unrated_places[np.argsort(predicted_ratings[unrated_places])[::-1]] 
#     recommended_place_names = places_df.iloc[recommended_place_indices, 1].values[:5] 
#     recommended_place_ratings = predicted_ratings[recommended_place_indices][:5]
#     return list(zip(recommended_place_names, recommended_place_ratings))


places = pd.read_csv('places.csv', sep=',', header=None, engine='python', encoding='latin-1')
places.columns = ['index_no', 'place_name', 'city_name', 'terrain', 'architecture', 'religion', 'activities', 'sports', 'mood', 'food', 'attractions', 'animals_found']

dataset = pd.read_csv('users.csv', delimiter=',')
dataset = np.array(dataset, dtype='int')

# Split the dataset into training and testing sets
training_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_places = int(max(max(training_set[:,1]), max(test_set[:,1])))

training_set = convert(training_set)
test_set = convert(test_set)

training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)
nb_epoch = 100
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = training_set[id_user].unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.requires_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_places/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data*mean_corrector)
            s += 1.
            optimizer.step()
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = training_set[id_user].unsqueeze(0)
    target = test_set[id_user].unsqueeze(0)
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.requires_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_places/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data*mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))

torch.save(sae.state_dict(), 'sae_model.pth')

