import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('models_data.csv')

ce_data = data[data['LossType'] == 'Cross Entropy']
fl_data = data[data['LossType'] == 'Focal Loss']

plt.figure(figsize=(10, 6))
for model in ce_data['Modelo'].unique():
    model_data = ce_data[ce_data['Modelo'] == model]
    plt.plot(model_data['Epoch'], model_data['Loss'], label=model)

plt.title('Pérdidas con Cross Entropy (CE)', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(title='Modelo')
plt.grid(True)
plt.savefig('loss_ce.png') 
plt.show()

plt.figure(figsize=(10, 6))
for model in fl_data['Modelo'].unique():
    model_data = fl_data[fl_data['Modelo'] == model]
    plt.plot(model_data['Epoch'], model_data['Loss'], label=model)

plt.title('Pérdidas con Focal Loss (FL)', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(title='Modelo')
plt.grid(True)
plt.savefig('loss_fl.png')
plt.show()
