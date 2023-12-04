import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Display result files", 
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_dir", default="./result/numpy", type=str, dest="data_dir")
args = parser.parse_args()
data_dir = args.data_dir

lst_data = os.listdir(data_dir)
lst_label = [f for f in lst_data if f.startswith('label')]
lst_input = [f for f in lst_data if f.startswith('input')]
lst_output = [f for f in lst_data if f.startswith('output')]

lst_label.sort()
lst_input.sort()
lst_output.sort()

id_list = []
for i in range (0, len(lst_label)):
  if lst_label[i].startswith('label_0.795000004'):
    id_list.append(i)
  if lst_label[i].startswith('label_1.980000004'):
    id_list.append(i)
differ_list = []

for id_iter in id_list:

  label = np.load(os.path.join(data_dir, lst_label[id_iter]))
  input = np.load(os.path.join(data_dir, lst_input[id_iter]))
  output = np.load(os.path.join(data_dir, lst_output[id_iter]))
  difference = np.abs(output - label)

  fig = plt.figure()
  ax = fig.add_subplot(1, 3, 1, projection='3d') # projection='3d' for 3d plot
  ax.axes.xaxis.set_ticklabels([])
  ax.axes.yaxis.set_ticklabels([])
  #Create X and Y data
  x = np.arange(-30, 30, 1) # shape of our data is 40 x 60
  y = np.arange(0, 40, 1)
  X, Y = np.meshgrid(x, y)
  surf = ax.plot_surface(X, Y, label, rstride=1, cstride=1, antialiased=True, cmap="plasma")
  ax.set_title('Ground Truth')
  ax = fig.add_subplot(1, 3, 2, projection='3d') # projection='3d' for 3d plot
  ax.axes.xaxis.set_ticklabels([])
  ax.axes.yaxis.set_ticklabels([])
  ax.plot_surface(X, Y, output, rstride=1, cstride=1, antialiased=True, cmap="plasma")
  ax.set_title('Prediction')
  ax = fig.add_subplot(1, 3, 3, projection='3d') # projection='3d' for 3d plot
  ax.axes.xaxis.set_ticklabels([])
  ax.axes.yaxis.set_ticklabels([])
  ax.plot_surface(X, Y, difference, rstride=1, cstride=1, antialiased=True, cmap="plasma")
  ax.set_title('Difference')
  ax.set_zlim3d(np.min(difference), np.max(difference))
  plt.show()
  
  # Contour plot
  x = np.arange(-30, 30, 1) # shape of our data is 40 x 60
  y = np.arange(0, 40, 1)
  X, Y = np.meshgrid(x, y)
  plt.subplot(131)
  CS = plt.contour(X, Y, label, levels=10, colors='k', vmin=np.min(label), vmax=np.max(label))
  cntr = plt.contourf(X, Y, label, levels=10, vmin=np.min(label), vmax=np.max(label))
  plt.colorbar(cntr)
  ax = plt.gca()
  ax.axes.xaxis.set_ticklabels([])
  ax.axes.yaxis.set_ticklabels([])
  plt.title('Ground Truth')
  plt.subplot(132)
  CS = plt.contour(X, Y, output, levels=10, colors='k', vmin=np.min(output), vmax=np.max(output))
  cntr = plt.contourf(X, Y, output, levels=10, vmin=np.min(output), vmax=np.max(output))
  plt.colorbar(cntr)
  ax = plt.gca()
  ax.axes.xaxis.set_ticklabels([])
  ax.axes.yaxis.set_ticklabels([])
  plt.title('Prediction')
  plt.subplot(133)
  cntr = plt.contourf(X, Y, difference, levels=4, vmin=np.min(difference), vmax=np.max(difference))
  plt.colorbar(cntr)
  ax = plt.gca()
  ax.axes.xaxis.set_ticklabels([])
  ax.axes.yaxis.set_ticklabels([])
  plt.title('Difference')
  plt.show()
