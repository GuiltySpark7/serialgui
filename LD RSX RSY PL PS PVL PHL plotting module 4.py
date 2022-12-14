import numpy as np
import tkinter as tk
from tkinter import *
import ctypes
from bokeh.plotting import figure, output_file, show
from timeit import default_timer as timer


ctypes.windll.shcore.SetProcessDpiAwareness(1)


def replace_zeroes_with_last_nonzero(arr):
    index = np.arange(len(arr))
    index[arr == 0] = 0
    index = np.maximum.accumulate(index)
    return arr[index]


def split_array_by_change_in_x_direction(xs, ys):
    dxs = xs - np.roll(xs, 1)
    gradientIDs = replace_zeroes_with_last_nonzero(np.sign(dxs))
    nodes = gradientIDs - np.roll(gradientIDs, 1)
    nodes[0] = 0
    splitMarkers = np.array(np.where(nodes != 0)).tolist()[0]
    xys = np.rot90(np.array((ys, xs)), k=-1)
    return np.split(xys, (splitMarkers))


def rollWrapWithConst1D(arr, roll, constant):
    arr = np.roll(arr, roll)
    if roll > 0:
        arr[: roll] = constant
    elif roll < 0:
        arr[roll: len(arr)] = constant
    return arr


def selectXsYs(xs, ys, xmin, xmax, ymin, ymax):
    clipMask = np.logical_and(
               np.logical_and(xs >= xmin, xs <= xmax),
               np.logical_and(ys >= ymin, ys < ymax))

    clipMask = np.logical_or(
               np.logical_or(clipMask, rollWrapWithConst1D(clipMask, 1, False)),
               np.logical_or(clipMask, rollWrapWithConst1D(clipMask, -1, False)))
    return xs[clipMask], ys[clipMask]


cross = [(1,0), (2,0), (3,0), (4,0), (-1,0), (-2,0), (-3,0), (-4,0), (0,1), (0,2),
         (0,3), (0,4), (0,-1), (0,-2), (0,-3), (0,-4)]

circle = [(3,0), (3,1), (3,2), (2,2), (2,3), (1,3), (0,3), (-1,3), (-2,2), (-3,2),
          (-3,1), (-3,0), (-3,-1), (-2,-2), (-2,-3), (-3,-3), (-2,-3), (-1,-3),
          (0,-3),(1,-3),(2,-3),(2,-2),(3,-2),(3,-1)]


class example():
    def __init__(self, graphWidth=1800, graphHeight=900, colour=True):
        self.graphWidth = graphWidth
        self.graphHeight = graphHeight
        self.plotArray = np.full((graphHeight, graphWidth), 0)
        self.plotArrayOverlay = np.full((graphHeight, graphWidth), 0)
        self.lines = {}
        self.xaxis = {}
        self.yaxis = {}
        self.xaxislinelinks = {}
        self.yaxislinelinks = {}

    def updateData(self, xs, ys, lineID, follow=None, zoom=None):
        np.concatenate([self.lines[lineID]['xs'], xs])
        np.concatenate([self.lines[lineID]['ys'], ys])
        xmaxN = max(xs)
        xminN = min(xs)
        ymaxN = max(ys)
        if ymaxN > ymaxO:
            ymaxD = ymaxN - ymaxO
        else:
            ymaxD = 0
        ymaxD = ymaxN - ymaxO
        if follow is True:
            # Nudge, sum up the nudge values of all the new data points, nudge + for above maximum
            # nudge - for bellow minimum
            pass

        if zoom is True:
            np.concatenate([self.lines[lineID]['xs'], xs])
            pass

        xsLast = self.lines[lineID]['xs'][-1]
        xs = np.append(xsLast, np.clip(xs, xsLast, None))


    def loaddata(self, xs, ys, lineID=None, xaxisID=None, yaxisID=None,
                 xmax=None, xmin=None, ymax=None, ymin=None, plotLine=True,
                 plotScatter=False, scatterMask=None, lineYfill=False):
        # Auto generate a unique lineID if none given
        # if lineID given and already exists, previous line overwritten
        # if new lineID given then a new line is created
        if lineID in self.lines:
            print(f'existing line: {lineID}')
            if xaxisID is None:
                xaxisID = self.lines[lineID]['xaxis']
                print(f'refrencing old x axis: {xaxisID}')
            if yaxisID is None:
                yaxisID = self.lines[lineID]['yaxis']
                print(f'refrencing old y axis: {yaxisID}')
        else:
            if lineID is None:
                lineID = "line " + str(len(self.lines) + 1)
                i = 2
                if lineID in self.lines:
                    lineID = "line " + str(len(self.lines) + i)
                    i += 1
            self.lines[lineID] = {}
            print(f'new line ID: {lineID}')

        # X axis parameter loading
        # New X axis
        if xaxisID not in self.xaxis:
            # Auto gen xaxis name if None given
            if xaxisID is None:
                xaxisID = "x axis " + str(len(self.xaxis) + 1)
                i = 2
                if xaxisID in self.xaxis:
                    xaxisID = "x axis " + str(len(self.xaxis) + i)
                    i += 1
            print(f'new X axis: {xaxisID}')
            if xmax is None:
                xmax = max(xs)
                print(f'xmax auto gened : {xmax}')
            else:
                print(f'xmax defined : {xmax}')
            if xmin is None:
                xmin = min(xs)
                print(f'xmin auto gened : {xmin}')
            else:
                print(f'xmax defined : {xmin}')
            self.xaxis[xaxisID] = {'xmax': xmax, 'xmin': xmin, 'linesLinked': set(lineID)}
            self.lines[lineID]['xaxis'] = xaxisID
            self.xaxislinelinks[lineID] = xaxisID
            print(f'new xaxis created and {lineID} linked to {xaxisID}')
            print(self.xaxis[xaxisID])
            print(self.lines[lineID]['xaxis'])
            print(self.xaxislinelinks[lineID])
        # Old X axis
        else:
            print('Old X axis, load the parameters')
            xmax = self.xaxis[xaxisID]['xmax']
            xmin = self.xaxis[xaxisID]['xmin']
            print(f'xmax = {xmax}')
            print(f'xmin = {xmin}')

        # Y axis parameter loading
        # how much headroom the auto scale leaves
        headroomPercent = 20
        HR = 1 + (headroomPercent/100)
        # New Y axis
        if yaxisID not in self.yaxis:
            # Auto gen yaxis name if None given
            if yaxisID is None:
                yaxisID = "y axis " + str(len(self.yaxis) + 1)
                i = 2
                if yaxisID in self.yaxis:
                    yaxisID = "y axis " + str(len(self.yaxis) + i)
                    i += 1
            print(f'new Y axis: {yaxisID}')
            if ymax is None:
                ymax = max(ys)*HR
                print(f'ymax auto gened : {ymax}')
            else:
                print(f'ymax defined : {ymax}')
            if ymin is None:
                ymin = min(ys)*HR
                print(f'ymin auto gened : {ymin}')
            else:
                print(f'ymin defined : {ymin}')
            self.yaxis[yaxisID] = {'ymax': ymax, 'ymin': ymin, 'linesLinked': set(lineID)}
            self.lines[lineID]['yaxis'] = yaxisID
            self.yaxislinelinks[lineID] = yaxisID
            print(f'new yaxis entry created and {lineID} linked to {yaxisID}')
            print(self.yaxis[yaxisID])
            print(self.lines[lineID]['yaxis'])
            print(self.yaxis[yaxisID]['linesLinked'])
        # Old Y axis
        else:
            print('Old X axis, load the parameters')
            ymax = self.yaxis[yaxisID]['ymax']
            ymin = self.yaxis[yaxisID]['ymin']

        self.lines[lineID].update({
                           'xs': xs,
                           'ys': ys,
                           'plotLine': plotLine,
                           'lineYfill': lineYfill,
                           'scatterMask': scatterMask,
                           'plotScatter': plotScatter,
                           'roll': 0})

        xsT, ysT = self.xsysTransform(lineID)

        if plotLine is True:
            Xline, Yline = self.plotLine(xsT, ysT, lineYfill=lineYfill)
            self.lines[lineID].update({
                               'Xline': Xline,
                               'Yline': Yline})
        if plotScatter is True:
            Xmarkers, Ymarkers = self.plotScatter(xsT, ysT, scatterMask=scatterMask)
            self.lines[lineID].update({
                               'Xmarkers': Xmarkers,
                               'Ymarkers': Ymarkers})


    def drawLineHorozontal(self, yVal, lineID):
        yaxisID = self.lines[lineID]['yaxis']
        ymin = self.yaxis[yaxisID]['ymin']
        ymax = self.yaxis[yaxisID]['ymax']
        graphH = self.graphHeight
        graphW = self.graphWidth
        ySF = (graphH - 1)/(ymax-ymin)

        X = np.arange(graphW)
        for val in yVal:
            Y = np.abs(((np.full(graphW, val) - ymin) * ySF) - (graphH - 1)).astype(np.int16)
            self.plotArray[Y, X] = 255

    def drawLineVertical(self, xVal, xaxisID=None, lineID=None):
        if lineID is not None:
            xaxisID = self.lines[lineID]['xaxis']
        xmin = self.xaxis[xaxisID]['xmin']
        xmax = self.xaxis[xaxisID]['xmax']
        graphW = self.graphWidth
        graphH = self.graphHeight
        xSF = (graphW - 1)/(xmax-xmin)

        Y = np.arange(graphH)
        for val in xVal:
            X = ((np.full(graphH, val) - xmin) * xSF).astype(np.int16)
            self.plotArray[Y, X] = 255

    def xrescale(self, xmin=None, xmax=None, xaxisID=None, lineID=None):
        # Identify the axis to be scaled
        if lineID is not None:
            xaxisID = self.lines[lineID]['xaxis']
        # get old xmax, xmin if not defined
        if xmin is None:
            xmin = self.xaxis[xaxisID]['xmin']
        if xmax is None:
            xmax = self.xaxis[xaxisID]['xmax']
        xminO = self.xaxis[xaxisID]['xmin']
        xmaxO = self.xaxis[xaxisID]['xmax']

        self.xaxis[xaxisID].update({'xmin': xmin,
                                    'xmax': xmax})
        print(f'rescaling x axis {xaxisID}')
        print(f'xmin: {xminO} --> {xmin}')
        print(f'xmax: {xmaxO} --> {xmax}')

        # cycle through all of the lines attatched to the relavent x-axis,
        # removing their old pixels then rescaling and plotting
        for lineID in self.xaxis[xaxisID]['linesLinked']:
            print(f'Line {lineID} attatched to {xaxisID}')

            xsT, ysT = self.xsysTransform(lineID)
            self.lines[lineID].update({'xsT': xsT,
                                           'ysT': ysT})

            print('New xsT, ysT, xSF, ySF generated')
            print('xsT:', xsT)
            print('ysT:', ysT)

            # replot the line if it was previously plotted
            if self.lines[lineID]['plotLine'] is True:
                Xline = self.lines[lineID]['Xline']
                Yline = self.lines[lineID]['Yline']
                self.plotArray[Yline, Xline] = 0
                print(f'{lineID} Old line cleared')

                Xline, Yline = self.plotLine(xsT, ysT, lineYfill=self.lines[lineID]['lineYfill'])
                self.lines[lineID].update({'Xline': Xline,
                                           'Yline': Yline})
                print('new line generated and plotted')

            # replot scatter if it was previously plotted
            if self.lines[lineID]['plotScatter'] is True:
                Xmarkers = self.lines[lineID]['Xmarkers']
                Ymarkers = self.lines[lineID]['Ymarkers']
                self.plotArray[Ymarkers, Xmarkers] = 0
                print(f'{lineID} Old Scatter cleared.')

                Xmarkers, Ymarkers = self.plotScatter(xsT, ysT, scatterMask=self.lines[lineID]['scatterMask'])
                self.lines[lineID].update({'Xmarkers': Xmarkers,
                                           'Ymarkers': Ymarkers})
                print('new markers generated and plotted')

    def yrescale(self, ymin=None, ymax=None, yaxisID=None, lineID=None):
        # Identify the axis to be scaled
        if lineID is not None:
            yaxisID = self.lines[lineID]['yaxis']
        # get old ymax, ymin if not defined
        if ymin is None:
            ymin = self.yaxis[yaxisID]['ymin']
        if ymax is None:
            ymax = self.yaxis[yaxisID]['ymax']
        yminO = self.yaxis[yaxisID]['ymin']
        ymaxO = self.yaxis[yaxisID]['ymax']

        self.yaxis[yaxisID].update({'ymin': ymin,
                                    'ymax': ymax})
        print(f'rescaling x axis {yaxisID}')
        print(f'ymin: {yminO} --> {ymin}')
        print(f'ymax: {ymaxO} --> {ymax}')

        # cycle through all of the lines attatched to the relavent x-axis,
        # removing their old pixels then rescaling and plotting
        for lineID in self.yaxis[yaxisID]['linesLinked']:
            print(f'Line {lineID} attatched to {yaxisID}')

            xsT, ysT = self.xsysTransform(lineID)
            self.lines[lineID].update({'xsT': xsT,
                                       'ysT': ysT})

            print('New xsT, ysT, xSF, ySF generated')
            print('xsT:', xsT)
            print('ysT:', ysT)

            # replot the line if it was previously plotted
            if self.lines[lineID]['plotLine'] is True:
                Xline = self.lines[lineID]['Xline']
                Yline = self.lines[lineID]['Yline']
                self.plotArray[Yline, Xline] = 0
                print(f'{lineID} Old line cleared')

                Xline, Yline = self.plotLine(xsT, ysT, lineYfill=self.lines[lineID]['lineYfill'])
                self.lines[lineID].update({'Xline': Xline,
                                           'Yline': Yline})
                print('new line generated and plotted')

            # replot scatter if it was previously plotted
            if self.lines[lineID]['plotScatter'] is True:
                Xmarkers = self.lines[lineID]['Xmarkers']
                Ymarkers = self.lines[lineID]['Ymarkers']
                self.plotArray[Ymarkers, Xmarkers] = 0
                print(f'{lineID} Old Scatter cleared.')

                Xmarkers, Ymarkers = self.plotScatter(xsT, ysT, scatterMask=self.lines[lineID]['scatterMask'])
                self.lines[lineID].update({'Xmarkers': Xmarkers,
                                           'Ymarkers': Ymarkers})
                print('new markers generated and plotted')


    def plotLine(self, xsT, ysT, lineYfill=False):
        Xline = []
        Yline = []
        xysSplit = split_array_by_change_in_x_direction(xsT.astype(np.int16), ysT.astype(np.int16))
        print(xysSplit)
        i = 0
        for arr in xysSplit:
            if i != 0:
                arr = np.vstack((xysSplit[i-1][-1], arr))
            arr = arr[np.argsort(arr[:,0], axis=0)]
            Xline.append(np.arange(np.ptp(arr[:, 0])) + min(arr[:, 0]))
            Yline.append(np.interp(Xline[i], arr[:, 0], arr[:, 1]).astype(np.int16))
            print('round X', i)
            print(arr)
            print('Xline = ', Xline[i])
            print('Yline = ', Yline[i])
            i = i + 1

        if lineYfill is True:
            j = 0
            xysSplit = split_array_by_change_in_x_direction(ysT.astype(np.int16), xsT.astype(np.int16))
            print(xysSplit)
            for arr in xysSplit:
                if j > 0:
                    arr = np.vstack((xysSplit[j-1][-1], arr))
                arr = arr[np.argsort(arr[:,0], axis=0)]
                Yline.append(np.arange(np.ptp(arr[:, 0])) + min(arr[:, 0]))
                Xline.append(np.interp(Yline[j+i], arr[:, 0], arr[:, 1]).astype(np.int16))
                print('round X', j)
                print(arr)
                print('Xline = ', Xline[i+j])
                print('Yline = ', Yline[i+j])
                j = j + 1
        Xline = np.concatenate(Xline)
        Yline = np.concatenate(Yline)
        clipMask = np.logical_and(np.logical_and(Xline>=0, Xline<self.graphWidth), np.logical_and(Yline>=0, Yline<self.graphHeight))
        Xline = Xline[clipMask]
        Yline = Yline[clipMask]
        # Xline = np.arange(graphW)
        # Yline = np.clip(np.interp(Xline, xsT, ysT).astype(np.int16), 0, graphH-1)
        print('final Xline = ', Xline)
        print('final Yline = ', Yline)
        self.plotArray[Yline, Xline] = 255
        return (Xline, Yline)

    def plotScatter(self, xsT, ysT, scatterMask=None):
        if scatterMask is None:
            scatterMask = cross
        newYsList = []
        newXsList = []
        for x, y in scatterMask:
            newYsList.append(ysT + y)
            newXsList.append(xsT + x)
        Ymarkers = np.clip(np.append(ysT, newYsList), 0, self.graphHeight-1)
        Xmarkers = np.clip(np.append(xsT, newXsList), 0, self.graphWidth-1)
        self.plotArray[Ymarkers, Xmarkers] = 255
        return (Xmarkers, Ymarkers)

    def xsysTransform(self, lineID):
        xaxisID = self.lines[lineID]['xaxis']
        yaxisID = self.lines[lineID]['yaxis']
        xmax = self.xaxis[xaxisID]['xmax']
        xmin = self.xaxis[xaxisID]['xmin']
        ymax = self.yaxis[yaxisID]['ymax']
        ymin = self.yaxis[yaxisID]['ymin']
        graphH = self.graphHeight
        graphW = self.graphWidth
        xs = self.lines[lineID]['xs']
        ys = self.lines[lineID]['ys']

        xs, ys = selectXsYs(xs, ys, xmin, xmax, ymin, ymax)

        ySF = (graphH - 1)/(ymax-ymin)
        xSF = (graphW - 1)/(xmax-xmin)

        ysT = np.rint(np.abs(((ys - ymin) * ySF) - (graphH - 1))).astype(np.int16)
        xsT = np.rint(((xs - xmin) * xSF)).astype(np.int16)

        self.lines[lineID].update({'xScaleFactor': xSF,
                                   'yScaleFactor': ySF,
                                   'xsT': xsT,
                                   'ysT': ysT})
        return xsT, ysT

    def updateFrameBoundsFollow(self, xs, ys, lineID):
        # convert to be class function and use self refrence
        # xs
        if xs is not None:
            xmaxO = self.xaxis[self.lines[lineID]['xaxis']]['xmax']
            xminO = self.xaxis[self.lines[lineID]['xaxis']]['xmin']
            xmaxN = max(xs)
            xminN = min(xs)
            if xmaxN > xmaxO or xminN < xminO:
                # check if there is a frame shift needed
                rangexsO = xmaxO - xminO
                rangexsN = xmaxN - xminN
                if rangexsN <= rangexsO:
                    # if the new xs are have a small range then the new frame
                    # can be placed simply by looking at the direction of old
                    # data and what would be the smallest shift
                    if abs(xmaxO - xmaxN) > abs(xmaxO - (xminN + rangexsO)):
                        xmax = xminN + rangexsO - (rangexsO * 1.1)
                        xmin = xminN - (rangexsO * 1.1)
                    else:
                        xmax = xmaxN + (rangexsO * 1.1)
                        #xmin = xmaxN - rangexsO  (rangexsO * 1.1)+
                    print('New value range < xaxis range, simple calc')
                else:
                    # otherwise the new frame must be placed by looking at how
                    # the new xs got to where they are
                    xs = np.flip(xs)
                    arrmax = np.maximum.accumulate(xs)
                    arrmin = np.minimum.accumulate(xs)
                    arrrange = arrmax - arrmin
                    firstOutOFRangeIndex = len(arrrange[arrrange <= rangexsO])
                    minimum = arrmin[firstOutOFRangeIndex-1]
                    maximum = arrmax[firstOutOFRangeIndex-1]
                    if xs[firstOutOFRangeIndex] > minimum:
                        xmin = minimum
                        xmax = minimum + rangexsO
                    else:
                        xmin = maximum - rangexsO
                        xmax = maximum
                    print('New value range > xaxis range, complicated calc')
                xshift = xmax - xmaxO


                print(f'x shifted: {xshift}')
            else:
                xmin = xminO
                xmax = xmaxO
                xshift = 0
                print(f'no x frame shift, xmin = {xmin}, xmax = {xmax}')


    def updateImage(self):
        '''Convert attribute numpy plotArray to a PPM format image which tkinter Canvas takes'''
        height, width = (self.plotArray).shape
        data = f'P5 {width} {height} 255 '.encode() + self.plotArray.astype(np.uint8).tobytes()
        return tk.PhotoImage(width=width, height=height, data=data, format='PPM', master=root)

# [23,567,123,578,243,744,234,754,234,678,234,679,184,284,872,56,568,432,777,176]
# [0,1,5,6,8,78,56,34,20,15,3,34,23,23,45,3,45,67,8,5]
# [2,2,2,4,5,6,7,7,7,8,9,12,14,13,12,6,3,5,7,0]
xs = np.arange(0,np.pi*3,0.001)
ys = np.tan(xs)
len(ys)
xs = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
ys = np.array([0,1,2,3,4,5,6,7,8,9,8,7,6,5,4,3,2,1,0])
# dys = ys - np.roll(ys,1)
# d2ys = dys - np.roll(dys,1)

graph = example()
graph.loaddata(xs, ys, lineID='y', plotScatter=True, lineYfill=True, plotLine=False, ymax=20, ymin=-20)
graph.xrescale(xmin=np.pi/2-1, xmax=np.pi/2+1, lineID='y')
graph.yrescale(ymin=-100, ymax=50, lineID='y')

graph.updateFrameBoundsFollow([1,18,170], [5,6,7], 'y')
#graph.drawLineHorozontal([1,2], 'y')
#graph.drawLineVertical([np.pi/2], lineID='y')
# graph.lines['auto']['xs'][-1]
root = Tk()

graphWidth = 1800
graphHeight = 900


img = graph.updateImage()

plotArea = Canvas(root, width=graphWidth, height=graphHeight, background="grey75")
plotArea.create_image(0, 0, anchor="nw", image=img)
plotArea.grid(row=0, column=0)

#.tk.call('tk', 'scaling', 2.0)
root.mainloop()

rangex = graph.xaxis[graph.lines['y']['xaxis']]['xmax'] - graph.xaxis[graph.lines['y']['xaxis']]['xmin']
rangey = graph.yaxis[graph.lines['y']['yaxis']]['ymax'] - graph.yaxis[graph.lines['y']['yaxis']]['ymin']
rangey

i = 0
start = timer()
while i <= 100:
    graph = example()
    graph.loaddata(xs, ys, lineID='y', plotScatter=True, lineYfill=False, xmin=0, xmax=18, ymax=10, ymin=-10)
    graph.xrescale(xmin=np.pi/2-0.3, xmax=np.pi/2+0.3, lineID='y')
    i += 1
end = timer()

print(end - start)
