import network
#import data

model = network.Network.model(numInputs=8, numOutputs=2)
model.build()
print(model.summary())