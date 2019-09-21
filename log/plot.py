import matplotlib.pyplot as plt

with open('epochs.csv') as f:
    lines = f.readlines()
loss_s = []
for loss in lines:
    loss = loss.rstrip()
    loss_s.append(float(loss))

plt.plot(loss_s)
plt.ylabel('avg loss')
plt.show()