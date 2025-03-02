import matplotlib.pyplot as plt

with open("state_value.txt", "r") as f:
    state_value = f.readlines()
y = []
for value in state_value:
    value = float(value[0: -1])
    y.append(value)
x = list(range(len(y)))
fig = plt.figure(figsize = (12, 8))
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, y, marker = ".", linewidth = 3)
ax.set_xlabel(xlabel = "training turn", fontsize = 18)
ax.set_ylabel(ylabel = "state value", fontsize = 18)
ax.tick_params(axis = 'both', labelsize = 18)

with open("reward.txt", "r") as f:
    reward = f.readlines()
y = []
for r in reward:
    r = float(r[0: -1])
    y.append(r)
x = list(range(len(y)))
fig = plt.figure(figsize = (12, 8))
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, y, marker = ".", linewidth = 3)
ax.set_xlabel(xlabel = "training turn", fontsize = 18)
ax.set_ylabel(ylabel = "reward", fontsize = 18)
ax.tick_params(axis = 'both', labelsize = 18)

with open("offl_rto.txt", "r") as f:
    offl_rto = f.readlines()
y = []
for rto in offl_rto:
    rto = float(rto[0: -1])
    y.append(rto)
x = list(range(len(y)))
fig = plt.figure(figsize = (12, 8))
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, y, marker = ".", linewidth = 3)
ax.set_xlabel(xlabel = "training turn", fontsize = 18)
ax.set_ylabel(ylabel = "offl_rto", fontsize = 18)
ax.tick_params(axis = 'both', labelsize = 18)

with open("tspw_rto.txt", "r") as f:
    tspw_rto = f.readlines()
y = []
for rto in tspw_rto:
    rto = float(rto[0: -1])
    y.append(rto)
x = list(range(len(y)))
fig = plt.figure(figsize = (12, 8))
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, y, marker = ".", linewidth = 3)
ax.set_xlabel(xlabel = "training turn", fontsize = 18)
ax.set_ylabel(ylabel = "tspw_rto", fontsize = 18)
ax.tick_params(axis = 'both', labelsize = 18)

with open("advantage.txt", "r") as f:
    advantage = f.readlines()
y = []
for a in advantage:
    a = float(a[0: -1])
    y.append(a)
x = list(range(len(y)))
fig = plt.figure(figsize = (12, 8))
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, y, marker = ".", linewidth = 3)
ax.set_xlabel(xlabel = "training turn", fontsize = 18)
ax.set_ylabel(ylabel = "advantage", fontsize = 18)
ax.tick_params(axis = 'both', labelsize = 18)

with open("prob.txt", "r") as f:
    prob = f.readlines()
y = []
for p in prob:
    p = float(p[0: -1])
    y.append(p)
x = list(range(len(y)))
fig = plt.figure(figsize = (12, 8))
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, y, marker = ".", linewidth = 3)
ax.set_xlabel(xlabel = "training turn", fontsize = 18)
ax.set_ylabel(ylabel = "prob", fontsize = 18)
ax.tick_params(axis = 'both', labelsize = 18)

with open("value_loss.txt", "r") as f:
    value_loss = f.readlines()
y = []
for loss in value_loss:
    loss = float(loss[0: -1])
    y.append(loss)
x = list(range(len(y)))
fig = plt.figure(figsize = (12, 8))
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, y, marker = ".", linewidth = 3)
ax.set_xlabel(xlabel = "training num", fontsize = 18)
ax.set_ylabel(ylabel = "value loss", fontsize = 18)
ax.tick_params(axis = 'both', labelsize = 18)

with open("entropy.txt", "r") as f:
    entropy = f.readlines()
y = []
for e in entropy:
    e = float(e[0: -1])
    y.append(e)
x = list(range(len(y)))
fig = plt.figure(figsize = (12, 8))
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, y, marker = ".", linewidth = 3)
ax.set_xlabel(xlabel = "training num", fontsize = 18)
ax.set_ylabel(ylabel = "entropy", fontsize = 18)
ax.tick_params(axis = 'both', labelsize = 18)

with open("policy_loss.txt", "r") as f:
    policy_loss = f.readlines()
y = []
for loss in policy_loss:
    loss = float(loss[0: -1])
    y.append(loss)
x = list(range(len(y)))
fig = plt.figure(figsize = (12, 8))
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, y, marker = ".", linewidth = 3)
ax.set_xlabel(xlabel = "training num", fontsize = 18)
ax.set_ylabel(ylabel = "policy loss", fontsize = 18)
ax.tick_params(axis = 'both', labelsize = 18)