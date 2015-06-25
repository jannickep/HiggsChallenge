# Author: Jannicke Pearkes
# Purpose: To make two parallel co-ordinate plots of the hyper-parameter space one with error and the other with time as a gradient

"""
Copyright (C) 2013 Matthew Woodruff
This script is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This script is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License
along with this script. If not, see <http://www.gnu.org/licenses/>.
===========================================================
parallel_coordinate.py
Reproduce the behavior of Jon's parallel_coordinate.m using
matplotlib and pandas
"""

import matplotlib              # plotting library
import matplotlib.mlab as mlab # matlab compatibility functions
from matplotlib.backends import backend_agg as agg # raster backend
import pandas       # data analysis library
import numpy        # numerical routines
import cPickle 

fig = matplotlib.figure.Figure() # create the figure
agg.FigureCanvasAgg(fig)         # attach the rasterizer
ax = fig.add_subplot(1, 1, 1)    # make axes to plot on

#array = cPickle.load(open("array.p","rb"))
array = cPickle.load(open("linear_search.p","rb"))
table = pandas.DataFrame(array)
#header = cPickle.load(open("header.p","rb"))
#header = header.split(",")
num_cols = 7# 11
header = ["finetuning learning rate", "pre-training epochs",
           "pre-training learning rate", "training epochs",
           "batch size", "neurons per layer",
           "number of layers"]
v = 2000 #number of lines to view (debug)
w = 20
mins = table.min()
maxs = table.max()
scaled = table.copy()
scaled = scaled.ix[0:v,0:w]
for column in table.columns:
    if column == num_cols:
    	mm = table[column].min()
        mx = table[column].max()
        #mm = 16.2
        #mx = 20.2
        print "max:"+str(mx)
        print "min:"+str(mm)
        #scaled[column] = table[column]
    	scaled[column] = (table[column] - mm) / (mx - mm)
    else: 
    	#scaled[column] = table[column]
        mmm = table[column].min()
        mmx = table[column].max()
        print "column: "+str(column)
        print "min: "+str(mmm)
        print "max: "+str(mmx)
        if (mmx-mmm)!= 0:
            scaled[column] = (table[column] - mmm) / (mmx - mmm)
        else:
        	scaled[column] = 0
print "normal"
print  table.ix[0:v,0:w]
print "scaled"
print scaled.ix[0:v,0:w]
#sorted_ = scaled.sort_index(axis = 0,by = 7 )
sorted1 = scaled.ix[0:v,0:w].sort(num_cols, axis = 0)#, ascending = False) #works with 6 but not 7
print "sorted"
print sorted1
sorted1 = sorted1.reset_index(drop=True)
print "sorted with new index"
print sorted1
cmap = matplotlib.cm.get_cmap("jet")

for solution in sorted1[:].iterrows():
#for solution in scaled[:].iterrows():
     ys = solution[1][0:num_cols]
     zs = solution[1][num_cols]
     #ys = ys.ix[0:10]
     #print ys
     #print zs
     #print "--"
     xs = range(len(ys))
     #print xs
# for col in range(0,10):
#     ys = scaled.ix[:,col]
#     xs = range(len(ys))
     ax.plot(xs, ys, c=cmap(zs), linewidth=2)


sm = matplotlib.cm.ScalarMappable(cmap=cmap)
sm.set_array([mm,mx])
#sm.set_array([16.2,20.2])
#sm.set_array([0,1])

cbar = fig.colorbar(sm)

cbar.ax.set_ylabel("Percent Validation Error (%)")

ax.set_title("Parallel Coordinate Plot of Hyper-Parameters")
ax.set_ylabel("Scaled Values")
ax.set_ylim(-.1,1.1)
ax.set_xticks(range(0,num_cols))

# Set the x axis ticks
'''
dimension = 10 
for dimension, (axx,xx) in enumerate(zip(axes, x[:-1])):
    axx.xaxis.set_major_locator(ticker.FixedLocator([xx]))
    ticks = len(axx.get_yticklabels())
    labels = list()
    step = min_max_range[dimension][2] / (ticks - 1)
    mn   = min_max_range[dimension][0]
    for i in xrange(ticks):
        v = mn + i*step
        labels.append('%4.2f' % v)
    axx.set_yticklabels(labels)
'''



#ax.set_xticklabels([" {0}".format(ii) for ii in range(1,10)])
ax.set_xticklabels(header[0:num_cols], rotation = 30)
fig.set_figheight(12)
fig.set_figwidth(20)
fig.savefig("parallel_coordinate.pdf")