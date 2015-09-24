#!/usr/bin/env python3.4

#######################################################################
## Nikolaus Mayer, 2013-2015                                         ##
##                                                                   ##
## Image grid viewer                                                 ##
#######################################################################

from __future__ import print_function
from PIL import Image
from PIL.ImageQt import ImageQt
from functools import partial
import numpy as np
import os
import sys
import re
import platform
from PyQt5 import QtCore, QtGui, QtWidgets
import gridview_c_interface

__version__ = '0.2.0'


preload = True

filename_template = '%07d-%s'
filename_template_batch = '%07d(%03d)-%s'

dir = '.'
batch = True

flowScale = 1.0
flowStyle = 0



## Configuration
configuration = {}
def readConfig(filename):
  '''Read a configuration file'''
  if not filename or not isinstance(filename, str) or \
     not os.path.isfile(filename):
    raise ValueError()
  with open(filename) as f:
    lines = f.readlines()
  lines = [line.split('#')[0].strip() for line in lines]
  global configuration
  configuration['X'], configuration['Y'] = \
      [int(v) for v in lines[0].split(' ')]
  cells = {}
  for line in lines[1:]:
    xstr,ystr,suffix = line.split(' ')
    cells['%s %s'%(xstr,ystr)] = suffix
  grid = []
  for x in range(configuration['X']):
    col = []
    for y in range(configuration['Y']):
      col.append({'suffix': cells['%s %s'%(x,y)]})
    grid.append(col)
  configuration['grid'] = grid



def GenerateFilenames(suffix):
  '''List all existing files complying to a certain pattern'''
  if batch:
    files = []
    i = j = 0
    while True:
      file = os.path.join(dir, filename_template_batch%(i,j,suffix))
      if os.path.isfile(file):
        files.append(file)
        j += 1
      elif j > 0:
        i += 1
        j = 0
      else:
        return files
  else:
    files = []
    i = 0
    while True:
      file = os.path.join(dir, filename_template%(i,suffix))
      if os.path.isfile(file):
        files.append(file)
        i += 1
      else:
        return files


def readPFM(file):
  '''Read .pfm files'''
  with open(file, 'rb') as file:
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == 'PF':
      color = True
    elif header == 'Pf':
      color = False
    else:
      raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
      width, height = map(int, dim_match.groups())
    else:
      raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
      endian = '<'
      scale = -scale
    else:
      endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def readFlow(name):
  '''Read optical flow images from .flo or .pfm files'''
  if name.endswith('.pfm') or name.endswith('.PFM'):
    return readPFM(name)[0][:,:,0:2]
  with open(name, 'rb') as f:
    header = f.read(4)
    if header != b'PIEH':
      raise Exception('Flow file (%s) header does not contain PIEH'\
                      %(name))
    width = np.fromfile(f, np.int32, 1)
    height = np.fromfile(f, np.int32, 1)
    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))
    return flow.astype(np.float32)


class Cell(object):
  '''Image grid cell for normal images'''
  def __init__(self, imagelabel, caption, suffix):
    self.label     = imagelabel
    self.caption   = caption
    self.suffix    = suffix
    self.filenames = []
    self.images    = []

  def clear(self):
    '''Discard all data'''
    self.filenames = []
    self.images    = []

  def addImage(self, filename):
    '''Add image (or image source)'''
    self.filenames.append(filename)
    if preload:
      img = QtGui.QImage(filename)
      if img is None:
        raise ValueError("Failed to read '%s'!" % f)
      self.images.append(QtGui.QPixmap.fromImage(img))

  def namePart(self, fullname):
    '''Extract filename from filepath'''
    return fullname.split('/')[-1]

  def index(self, idx, *args):
    '''Change displayed image'''
    if not self.filenames:
      return
    self.caption.setText(self.namePart(self.filenames[idx]))
    if preload:
      self.label.setPixmap(self.images[idx]\
                           .scaled(self.label.size(), 
                                   QtCore.Qt.KeepAspectRatio))
    else:
      f = self.filenames[idx]
      self.label.setPixmap(QtGui.QPixmap.fromImage(QtGui.QImage(f))\
                           .scaled(self.label.size(),
                                   QtCore.Qt.KeepAspectRatio))


class FlowCell(Cell):
  '''Image grid cell for optical flow images'''
  def __init__(self, imagelabel, caption, suffix):
    super().__init__(imagelabel, caption, suffix)
    self.raw_data = []

  def clear(self):
    '''Discard all data'''
    super().clear()
    self.raw_data = []

  def addImage(self, filename):
    '''Add image (or image source)'''
    self.filenames.append(filename)
    if preload:
      raw_flow_data = readFlow(filename)
      self.raw_data.append(raw_flow_data)

  def index(self, idx, scale, *args):
    '''Change displayed image'''
    if not self.filenames:
      return
    self.caption.setText(self.namePart(self.filenames[idx]))
    if preload:
      if flowStyle == 0:
        flow_image = gridview_c_interface.ColorFlow(self.raw_data[idx],
                                                    flowScale)
      else:
        flow_image = gridview_c_interface.ColorFlow2(self.raw_data[idx],
                                                     flowScale)
      self.label.setPixmap(QtGui.QPixmap.fromImage(
                           ImageQt(Image.fromarray(flow_image)))\
                           .scaled(self.label.size(),
                                   QtCore.Qt.KeepAspectRatio))
    else:
      raw_flow_data = readFlow(self.filenames[idx])
      if flowStyle == 0:
        flow_image = gridview_c_interface.ColorFlow(raw_flow_data, 
                                                    flowScale)
      else:
        flow_image = gridview_c_interface.ColorFlow2(raw_flow_data, 
                                                     flowScale)
      self.label.setPixmap(QtGui.QPixmap.fromImage(\
                           ImageQt(Image.fromarray(flow_image))\
                           .scaled(self.label.size(),
                                   QtCore.Qt.KeepAspectRatio)))

  def scale(self, scale):
    '''Change optical flow scalingG'''



class MainWindow(QtWidgets.QMainWindow):
  def __init__(self, parent=None):
    super(MainWindow, self).__init__(parent)

    self.currentIndex = 0
    self.grid = []
    self.needFlowTools = False

    ## Grid of images
    gridlayout = QtWidgets.QGridLayout(self)
    global configuration
    for y in range(configuration['Y']):
      for x in range(configuration['X']):
        cell_container = QtWidgets.QWidget()
        cell_layout = QtWidgets.QVBoxLayout()
        image_label = QtWidgets.QLabel()
        image_label.setMinimumSize(80,60)
        image_label.setAlignment(QtCore.Qt.AlignCenter)
        image_label.setAutoFillBackground(True)
        image_label.setPalette(QtGui.QPalette(QtCore.Qt.white))
        image_caption = QtWidgets.QLabel()
        image_caption.setFixedHeight(20)
        image_caption.setAlignment(QtCore.Qt.AlignCenter)
        cell_layout.addWidget(image_label)
        cell_layout.addWidget(image_caption)
        cell_container.setLayout(cell_layout)
        gridlayout.addWidget(cell_container, y, x)
        suffix = configuration['grid'][x][y]['suffix']
        if suffix.endswith('ppm'):
          self.grid.append(Cell(image_label, image_caption, suffix))
        else:
          self.grid.append(FlowCell(image_label, image_caption, suffix))
          self.needFlowTools = True
    self.gridcontainer = QtWidgets.QWidget()
    self.gridcontainer.setLayout(gridlayout)

    mainlayout = QtWidgets.QVBoxLayout()
    mainlayout.addWidget(self.gridcontainer)

    ## Slider and buttons
    toolsContainer = QtWidgets.QWidget()
    toolsLayout = QtWidgets.QHBoxLayout()
    toolsContainer.setLayout(toolsLayout)
    toolsContainer.setFixedHeight(80)

    sliderLayout = QtWidgets.QVBoxLayout()
    sliderContainer = QtWidgets.QWidget()
    sliderContainer.setLayout(sliderLayout)
    sliderContainer.setFixedHeight(80)
    toolsLayout.addWidget(sliderContainer)
    # Frame-control slider as QDockWidget
    self.frameSlider = QtWidgets.QSlider(orientation=QtCore.Qt.Horizontal)
    self.frameSlider.setRange(0,0)
    self.frameSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
    self.frameSliderLabel = QtWidgets.QLabel("Frame: 0")
    self.frameSliderLabel.setMinimumSize(80,10)
    framesliderlayout = QtWidgets.QHBoxLayout()
    framesliderlayout.addWidget(self.frameSlider)
    framesliderlayout.addWidget(self.frameSliderLabel)
    frameslidercontainer = QtWidgets.QWidget()
    frameslidercontainer.setLayout(framesliderlayout)
    frameslidercontainer.setFixedHeight(40)
    sliderLayout.addWidget(frameslidercontainer)
    
    # Slider for optical flow scaling if necessary
    if self.needFlowTools:
      self.flowScaleSlider = QtWidgets.QSlider(orientation=QtCore.Qt.Horizontal)
      self.flowScaleSlider.setRange(1,3200)
      self.flowScaleSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
      self.flowScaleSlider.setValue(100)
      self.flowScaleSliderLabel = QtWidgets.QLabel("OF scale: 1.0")
      self.flowScaleSliderLabel.setMinimumSize(80,10)
      flowScaleSliderLayout = QtWidgets.QHBoxLayout()
      flowScaleSliderLayout.addWidget(self.flowScaleSlider)
      flowScaleSliderLayout.addWidget(self.flowScaleSliderLabel)
      flowScaleSliderContainer = QtWidgets.QWidget()
      flowScaleSliderContainer.setLayout(flowScaleSliderLayout)
      flowScaleSliderContainer.setFixedHeight(40)
      sliderLayout.addWidget(flowScaleSliderContainer)

    # Flow visualization choice
    if self.needFlowTools:
      flowStyleContainer = QtWidgets.QWidget()
      flowStyleLayout = QtWidgets.QVBoxLayout()
      flowStyleWhiteButton = QtWidgets.QRadioButton('&Sintel style')
      flowStyleWhiteButton.setChecked(True)
      flowStyleWhiteButton.clicked.connect(partial(self.setFlowStyle, 0))
      flowStyleBlackButton = QtWidgets.QRadioButton('&Middlebury style')
      flowStyleBlackButton.clicked.connect(partial(self.setFlowStyle, 1))
      flowStyleButtonGroup = QtWidgets.QButtonGroup()
      flowStyleButtonGroup.addButton(flowStyleWhiteButton)
      flowStyleButtonGroup.addButton(flowStyleBlackButton)
      flowStyleLayout.addWidget(flowStyleWhiteButton)
      flowStyleLayout.addWidget(flowStyleBlackButton)
      flowStyleContainer.setLayout(flowStyleLayout)
      toolsLayout.addWidget(flowStyleContainer)

    mainlayout.addWidget(toolsContainer)
    self.mainlayoutcontainer = QtWidgets.QWidget()
    self.mainlayoutcontainer.setLayout(mainlayout)
    self.setCentralWidget(self.mainlayoutcontainer)

    # Status bar with resize grip, for status messages
    status = self.statusBar()
    status.setSizeGripEnabled(True)
    status.showMessage("Ready", 5000)

    ## Actions
    # Open one image set
    fileOpenAction  = self.createAction("&Open folder",
                                        self.fileOpen,
                                        None,
                                        "Open a folder containing images")
    # Exit program
    fileQuitAction = self.createAction("&Quit",
                                       self.close,
                                       "Ctrl+Q",
                                       "Close the application")
    # Display "About" dialog
    helpAboutAction= self.createAction("&About",
                                       self.helpAbout,
                                       None,
                                       "More information about this program")

    # Menus and toolbars
    self.fileMenu = self.menuBar().addMenu("&File")
    self.addActions(self.fileMenu, (fileOpenAction,
                                    None, 
                                    fileQuitAction,))
    self.helpMenu = self.menuBar().addMenu("&Help")
    self.addActions(self.helpMenu, (helpAboutAction,))
    fileToolBar = self.addToolBar("File")
    fileToolBar.setObjectName("FileToolBar")
    self.addActions(fileToolBar, (fileOpenAction,))

    self.setWindowTitle(QtWidgets.QApplication.applicationName())
    
    self.frameSlider.valueChanged.connect(self.frameSliderChange)
    if self.needFlowTools:
      self.flowScaleSlider.valueChanged.connect(self.flowScaleSliderChange)


  def setFlowStyle(self, newStyle):
    '''Assign to global flowStyle'''
    global flowStyle
    flowStyle = newStyle
    self.changeFrame(self.currentIndex)

  def resizeEvent(self, resizeEvent):
    if self.grid[0].filenames:
      self.changeFrame(self.currentIndex)


  def frameSliderChange(self, newValue):
    """React to the user manipulating the frame control slider"""
    if newValue == self.currentIndex:
      return
    self.changeFrame(newValue)


  def flowScaleSliderChange(self, newValue):
    """React to the user manipulating the flow scale control slider"""
    global flowScale
    flowScale = 0.01*newValue
    for cell in self.grid:
      if isinstance(cell, FlowCell):
        cell.index(self.currentIndex, flowScale)
    self.flowScaleSliderLabel.setText("OF scale: %.2f" % (flowScale))
    if self.needFlowTools:
      self.flowScaleSlider.setValue(newValue)


  def helpAbout(self):
    """Display "About" dialog"""
    QtWidgets.QMessageBox.about(self, "About Gridview",
        """<b>Gridview</b>  v%s
        <p>Copyright &copy; 2013-2015 Nikolaus Mayer
        <p>Python %s - Qt %s - PyQt %s on %s""" % \
          ( __version__, 
            platform.python_version(),
            QtCore.QT_VERSION_STR,
            QtCore.PYQT_VERSION_STR,
            platform.system() ))

  def addActions(self, target, actions):
    """Add a list of actions (and separators) to a menu/toolbar target"""
    for action in actions:
      if action is None:
        target.addSeparator()
      else:
        target.addAction(action)


  def createAction(self, text, 
                   slot=None, 
                   shortcut=None, 
                   tip=None, 
                   checkable=False, 
                   signal="triggered()"):
    """Shortcut for creating a QAction object and setting attributes"""
    action = QtWidgets.QAction(text, self)
    if shortcut is not None:
      action.setShortcut(shortcut)
    if tip is not None:
      action.setToolTip(tip)
      action.setStatusTip(tip)
    if slot is not None:
      action.triggered.connect(slot)
    if checkable:
      action.setCheckable(True)
    return action


  def fileOpen(self):
    """Load a set of images, discovered from one specimen selected by the user"""
    header = "Choose folder"
    folder = QtWidgets.QFileDialog.getExistingDirectory(
              self,
              "%s - %s" % (QtWidgets.QApplication.applicationName(), header),
              '.')

    global dir
    dir = folder
    
    testfile = filename_template_batch%(0,0,self.grid[0].suffix)
    global batch
    batch = os.path.isfile(os.path.join(dir, testfile))

    # If the user selected a file, autodiscover a fitting set
    if folder:
      self.loadFiles()
      return True
    return False


  def resetFrameSlider(self):
    """Reset the frame control slider to 0 and update range and label"""
    if not self.grid[0].filenames:
      return
    self.frameSlider.setRange(0, len(self.grid[0].filenames)-1)
    self.frameSlider.setValue(0)
    self.frameSliderLabel.setText("Frame: 0")


  def loadFiles(self):
    """Load file set, given one specimen (fname)"""
    if not dir:
      return
    for cell in self.grid:
      cell.clear()
      found_filenames = GenerateFilenames(cell.suffix)
      if not found_filenames:
        raise Exception()
      for i,f in enumerate(found_filenames):
        cell.addImage(f)
        if preload:
          self.updateStatus("Reading images... %d/%d." % (i+1, len(found_filenames)))
    else:
      self.resetFrameSlider()
      self.changeFrame(0)


  def changeFrame(self, index):
    """Change the displayed frame"""
    self.currentIndex = index
    for cell in self.grid:
      cell.index(index, flowScale)

    titlestr = "%s - %s" % \
               (dir, QtWidgets.QApplication.applicationName())
    self.setWindowTitle(titlestr)

    self.frameSliderLabel.setText("Frame: %d" % (index))
    self.frameSlider.setValue(index)


  def wheelEvent(self, event):
    """Scroll through image set via mouse wheel"""
    cid = self.currentIndex
    if event.angleDelta().y() > 0:
      cid += 1
    else:
      cid -= 1
    cid = max(cid, 0)
    cid = min(cid, len(self.grid[0].filenames)-1)
    self.changeFrame(cid)


  def updateStatus(self, message=None):
    """Write a new message to the application's status bar"""
    if message is None:
      return
    self.statusBar().showMessage(message, 5000)



def main():

  if len(sys.argv) >= 2:
    readConfig(sys.argv[1])
  else:
    print('Using default.cfg layout')
    readConfig('default.cfg')

  if '--no-preload' in sys.argv:
    global preload
    preload = False

  app = QtWidgets.QApplication(sys.argv)
  app.setApplicationName("Gridview")
  form = MainWindow()
  form.show()
  app.exec_()

if __name__ == '__main__':
  main()

