import torch
import joblib
import matplotlib.pyplot as plt

#
# lossAdam = joblib.load('Adam.pkl')
# lossMoment = joblib.load('moment.pkl')
# lossSgd = joblib.load('loss01.pkl')
# lossRms = joblib.load('Rms.pkl')
# plt.plot(lossSgd,  label="SGD")
# plt.plot(lossMoment, label="Moment")
# plt.plot(lossRms, label="RMS")
# plt.plot(lossAdam, label="Adam")
# plt.ylabel("Loss (cross-entropy)")
# plt.xlabel("Epoch")
# plt.grid()
# plt.legend()
# plt.title("optimizer compare")
# plt.show()

loss01 = joblib.load('loss01.pkl')
loss001 = joblib.load('loss001.pkl')
loss1 = joblib.load('loss02.pkl')
loss005 = joblib.load('loss005.pkl')

plt.plot(loss005, label="lr=0.5")
plt.plot(loss1,  label="lr=0.02")
plt.plot(loss01, label="lr=0.01")
plt.plot(loss001, label="lr=0.005")
plt.ylabel("Loss (cross-entropy)")
plt.xlabel("Epoch")
plt.grid()
plt.legend()
plt.title("LR compare")
plt.show()