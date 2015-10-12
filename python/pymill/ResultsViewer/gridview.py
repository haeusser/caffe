#!/usr/bin/env python3.4

#######################################################################
## Nikolaus Mayer, 2013-2015                                         ##
##                                                                   ##
## Image grid viewer                                                 ##
#######################################################################

from PIL import Image
from PIL.ImageQt import ImageQt
from functools import partial
import logging
import numpy as np
import os
import sys
import re
import platform
from PyQt5 import QtCore, QtGui, QtWidgets
import gridview_c_interface

__version__ = '0.2.0'


## Preload data?
preload = False

## Randomize data?
permute = True

## Script file location
here = os.path.dirname(__file__)

## Structure of the names of the expected data files
filename_template = '%07d%s'
filename_template_batch = '%07d(%03d)%s'

dir = None
batch = True

flowScale = 1.0
flowStyle = 1

floatMin = 0.0
floatMax = 1.0

## Will be set to updateStatus(message) function handle
updateStatus = None



## Configuration
configuration = None
def parseConfig(lines):
  '''Parse configuration data'''
  global configuration
  configuration = {}
  configuration['X'], configuration['Y'] = \
      [int(v) for v in lines[0].split(' ')[:2]]
  cells = {}
  for line in lines[1:]:
    ## Options
    if line.startswith('option'):
      parts = line.split(' ')[1:]
      configuration[parts[0]] = parts[1:]
    ## Grid cells
    else:
      xstr,ystr,suffix = line.split(' ')
      cells['%s %s'%(xstr,ystr)] = suffix
  grid = []
  for x in range(configuration['X']):
    col = []
    for y in range(configuration['Y']):
      index_str = '%s %s'%(x,y)
      col.append({'suffix': cells[index_str]})
    grid.append(col)
  configuration['grid'] = grid
  logging.debug(configuration)

def readConfig(filename):
  '''Read a configuration file'''
  if not filename or not isinstance(filename, str) or \
     not os.path.isfile(filename):
    raise ValueError()
  with open(filename) as f:
    lines = f.readlines()
  ## Discard comments and blank lines
  lines = [line.split('#')[0].strip() for line in lines 
        if line.split('#')[0].strip()]
  parseConfig(lines)



def GenerateFilenames(suffices):
  '''List all existing files complying to a certain pattern'''
  '''if suffix == 'none':
    return ['dummy']
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
  '''
  found = [[] for s in suffices]
  if batch:
    i = j = 0
    while True:
      valid_index = False
      for (idx,suffix) in enumerate(suffices):
        f = os.path.join(dir, filename_template_batch(i,j,suffix))
        if os.path.isfile(f):
          found[idx].append(f)
          valid_index = True
        elif idx < len(suffices)-1 or valid_index:
          found[idx].append(None)
      if valid_index:
        j += 1
      elif j > 0:
        i += 1
        j = 0
      else:
        return found
  else:
    i = 0
    while True:
      valid_index = False
      for (idx,suffix) in enumerate(suffices):
        f = os.path.join(dir, filename_template%(i,suffix))
        if os.path.isfile(f):
          found[idx].append(f)
          valid_index = True
        elif idx < len(suffices) or valid_index:
          found[idx].append(None)
      if valid_index:
        i += 1
      else:
        return found 
          



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


def readFloat(name):
  '''Read float file (binary blob with leading dimension info)'''
  with open(name, 'rb') as f:
    if f.readline() != b'float\n':
      raise Exception('float file %s did not contain <float> keyword' % name)

    dim = int(f.readline())
    dims = []
    count = 1
    for i in range(0, dim):
      d = int(f.readline())
      dims.append(d)
      count *= d

    ## Hacky hack. multichannel data -> use first channel
    if dim == 3:
      dim = 2
      dims = dims[:2]
      dims = dims[::-1]

    data = np.fromfile(f, np.float32, count).reshape(dims)
    if dim == 2:
      #data = np.transpose(data, (1, 0))
      data = np.transpose(data, (0, 1))
    else:
      raise Exception('dimensions (%s) not supported'%(','.join(dims)))
    """elif dim == 3:
      if dims[2] == 1:
        data = data[:,:,0]
        logging.debug(data.shape)
        #data = np.transpose(data, (2, 1, 0))
        #data = np.transpose(data, (1, 0, 2))
        #logging.debug(data.shape)
        #data = np.transpose(data, (1, 0))
        #data = np.transpose(data, (0, 1))
      else:
        raise Exception('3d data not (yet) supported')
        #data = np.transpose(data, (2, 1, 0))
        #data = np.transpose(data, (1, 0, 2))"""

    return data


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
      if filename is None:
        self.images.append(None)
      else:
        img = QtGui.QImage(filename)
        if img is None:
          raise ValueError("Failed to read '%s'!" % f)
        self.images.append(QtGui.QPixmap.fromImage(img))

  def namePart(self, fullname):
    '''Extract filename from filepath'''
    if fullname is None:
      return '(missing)'
    else:
      return fullname.split('/')[-1]

  def index(self, idx, *args):
    '''Change displayed image'''
    if not self.filenames:
      return
    self.caption.setText(self.namePart(self.filenames[idx]))
    if self.filenames[idx] is None:
      self.label.clear()
      return
    else:
      if preload:
        self.label.setPixmap(self.images[idx]\
                             .scaled(self.label.size(), 
                                     QtCore.Qt.KeepAspectRatio))
      else:
        f = self.filenames[idx]
        self.label.setPixmap(QtGui.QPixmap.fromImage(QtGui.QImage(f))\
                             .scaled(self.label.size(),
                                     QtCore.Qt.KeepAspectRatio))


class EmptyCell(Cell):
  '''Empty grid cell'''
  def __init__(self, imagelabel, caption, suffix):
    self.suffix = suffix
    pass

  def clear(self):
    pass

  def addImage(self, filename):
    pass

  def index(self, idx, *args):
    pass



class FloatCell(Cell):
  '''Image grid cell for floating point image'''
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
      if filename is None:
        self.raw_data.append(None)
      else:
        raw_float_data = readFloat(filename)
        self.raw_data.append(raw_float_data)

  def index(self, idx, *args):
    '''Change displayed image'''
    if not self.filenames:
      return
    self.caption.setText(self.namePart(self.filenames[idx]))
    if self.filenames[idx] is None:
      self.label.clear()
      return
    else:
      if preload:
        raw_float_data = self.raw_data[idx]
      else:
        raw_float_data = readFloat(self.filenames[idx])

    if (floatMax-floatMin)<0.01:
      updateStatus('Float scale minimum too close to maximum!')
      return

    scale = 255./(floatMax-floatMin)
    offset = -floatMin
    self.label.setPixmap(QtGui.QPixmap.fromImage(
                         ImageQt(Image.fromarray((raw_float_data+offset)*scale)\
                                 .convert('L')))\
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
      if filename is None:
        self.raw_data.append(None)
      else:
        raw_flow_data = readFlow(filename)
        self.raw_data.append(raw_flow_data)

  def index(self, idx, *args):
    '''Change displayed image'''
    if not self.filenames:
      return
    self.caption.setText(self.namePart(self.filenames[idx]))
    if self.filenames[idx] is None:
      self.label.clear()
      return
    else:
      if preload:
        raw_flow_data = self.raw_data[idx],
      else:
        raw_flow_data = readFlow(self.filenames[idx])
      flow_image = gridview_c_interface.Flow(flowStyle, 
                                             raw_flow_data,
                                             flowScale)
      self.label.setPixmap(QtGui.QPixmap.fromImage(\
                           ImageQt(Image.fromarray(flow_image))\
                           .scaled(self.label.size(),
                                   QtCore.Qt.KeepAspectRatio)))



class MainWindow(QtWidgets.QMainWindow):
  def __init__(self, parent=None):
    super(MainWindow, self).__init__(parent)

    global updateStatus
    updateStatus = self.updateStatus

    self.currentIndex = 0
    self.grid = []
    self.needFloatTools = False
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
        elif suffix.endswith('float') or suffix.endswith('float3'):
          self.grid.append(FloatCell(image_label, image_caption, suffix))
          self.needFloatTools = True
        elif suffix.endswith('flo'):
          self.grid.append(FlowCell(image_label, image_caption, suffix))
          self.needFlowTools = True
        elif suffix == 'none':
          self.grid.append(EmptyCell(image_label, image_caption, suffix))
        else:
          raise Exception('Invalid suffix: >%s<'%(suffix))
    self.gridcontainer = QtWidgets.QWidget()
    self.gridcontainer.setLayout(gridlayout)

    mainlayout = QtWidgets.QVBoxLayout()
    mainlayout.addWidget(self.gridcontainer)

    ## Choose input mode
    testfile = filename_template_batch%(0,0,self.grid[0].suffix)
    global batch
    batch = os.path.isfile(os.path.join(dir, testfile))

    ## Slider and buttons
    toolsContainer = QtWidgets.QWidget()
    toolsContainer.setContentsMargins(0,0,0,0)
    toolsLayout = QtWidgets.QHBoxLayout()
    toolsLayout.setContentsMargins(0,0,0,0)
    toolsContainer.setLayout(toolsLayout)
    h = 40
    if self.needFloatTools:
      h += 60
    if self.needFlowTools:
      h += 40
    toolsContainer.setFixedHeight(h)

    sliderLayout = QtWidgets.QVBoxLayout()
    sliderLayout.setContentsMargins(0,0,0,0)
    sliderContainer = QtWidgets.QWidget()
    sliderContainer.setContentsMargins(0,0,0,0)
    sliderContainer.setLayout(sliderLayout)
    sliderContainer.setFixedHeight(100)
    toolsLayout.addWidget(sliderContainer)
    # Frame-control slider as QDockWidget
    self.frameSlider = QtWidgets.QSlider(orientation=QtCore.Qt.Horizontal)
    self.frameSlider.setRange(0,0)
    self.frameSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
    self.frameSliderLabel = QtWidgets.QLabel("Frame: 0")
    self.frameSliderLabel.setFixedSize(100,20)
    framesliderlayout = QtWidgets.QHBoxLayout()
    framesliderlayout.setContentsMargins(0,0,0,0)
    framesliderlayout.addWidget(self.frameSlider)
    framesliderlayout.addWidget(self.frameSliderLabel)
    frameslidercontainer = QtWidgets.QWidget()
    frameslidercontainer.setContentsMargins(0,0,0,0)
    frameslidercontainer.setLayout(framesliderlayout)
    frameslidercontainer.setFixedHeight(40)
    sliderLayout.addWidget(frameslidercontainer)

    # Sliders for float scaling if necessary
    if self.needFloatTools:
      ## Min, max
      if 'floatlimits' in configuration:
        limits = {'min': 100*float(configuration['floatlimits'][0]),
                  'max': 100*float(configuration['floatlimits'][1])}
      else:
        limits = {'min': -25500,
                  'max': 25500}
      ## Preset values
      if 'floatminmax' in configuration:
        presets = {'min': 100*float(configuration['floatminmax'][0]),
                   'max': 100*float(configuration['floatminmax'][1])}
      else:
        presets = {'min': 0, 
                   'max': 100}
      global floatMin, floatMax
      floatMin, floatMax = presets['min']/100., presets['max']/100.

      self.floatMinSlider = QtWidgets.QSlider(orientation=QtCore.Qt.Horizontal)
      self.floatMinSlider.setRange(limits['min'], limits['max'])
      self.floatMinSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
      self.floatMinSlider.setValue(presets['min'])
      self.floatMinSliderLabel = QtWidgets.QLabel("Float min: %.2f"%(presets['min']/100.))
      self.floatMinSliderLabel.setFixedSize(100,20)
      floatMinSliderLayout = QtWidgets.QHBoxLayout()
      floatMinSliderLayout.setContentsMargins(0,0,0,0)
      floatMinSliderLayout.addWidget(self.floatMinSlider)
      floatMinSliderLayout.addWidget(self.floatMinSliderLabel)
      floatMinSliderContainer = QtWidgets.QWidget()
      floatMinSliderContainer.setContentsMargins(0,0,0,0)
      floatMinSliderContainer.setLayout(floatMinSliderLayout)
      floatMinSliderContainer.setFixedHeight(40)
      ## Max
      self.floatMaxSlider = QtWidgets.QSlider(orientation=QtCore.Qt.Horizontal)
      self.floatMaxSlider.setRange(limits['min'], limits['max'])
      self.floatMaxSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
      self.floatMaxSlider.setValue(presets['max'])
      self.floatMaxSliderLabel = QtWidgets.QLabel("Float max: %.2f"%(presets['max']/100.))
      self.floatMaxSliderLabel.setFixedSize(100,20)
      floatMaxSliderLayout = QtWidgets.QHBoxLayout()
      floatMaxSliderLayout.setContentsMargins(0,0,0,0)
      floatMaxSliderLayout.addWidget(self.floatMaxSlider)
      floatMaxSliderLayout.addWidget(self.floatMaxSliderLabel)
      floatMaxSliderContainer = QtWidgets.QWidget()
      floatMaxSliderContainer.setContentsMargins(0,0,0,0)
      floatMaxSliderContainer.setLayout(floatMaxSliderLayout)
      floatMaxSliderContainer.setFixedHeight(40)
      ## Container
      floatSlidersLayout = QtWidgets.QHBoxLayout()
      floatSlidersLayout.setContentsMargins(0,0,0,0)
      floatSlidersLayout.addWidget(floatMinSliderContainer)
      floatSlidersLayout.addWidget(floatMaxSliderContainer)
      floatSlidersContainer = QtWidgets.QWidget()
      floatSlidersContainer.setContentsMargins(0,0,0,0)
      floatSlidersContainer.setLayout(floatSlidersLayout)
      floatSlidersContainer.setFixedHeight(40)
      sliderLayout.addWidget(floatSlidersContainer)
    
    # Slider for optical flow scaling if necessary
    if self.needFlowTools:
      global flowScale
      flowScale = 100*0.0005
      self.flowScaleSlider = QtWidgets.QSlider(orientation=QtCore.Qt.Horizontal)
      self.flowScaleSlider.setRange(1,3200)
      self.flowScaleSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
      self.flowScaleSlider.setValue(100)
      self.flowScaleSliderLabel = QtWidgets.QLabel("OF scale: 0.05")
      self.flowScaleSliderLabel.setFixedSize(100,20)
      flowScaleSliderLayout = QtWidgets.QHBoxLayout()
      flowScaleSliderLayout.setContentsMargins(0,0,0,0)
      flowScaleSliderLayout.addWidget(self.flowScaleSlider)
      flowScaleSliderLayout.addWidget(self.flowScaleSliderLabel)
      flowScaleSliderContainer = QtWidgets.QWidget()
      flowScaleSliderContainer.setContentsMargins(0,0,0,0)
      flowScaleSliderContainer.setLayout(flowScaleSliderLayout)
      flowScaleSliderContainer.setFixedHeight(40)
      sliderLayout.addWidget(flowScaleSliderContainer)
      # Flow visualization choice
      flowStyleContainer = QtWidgets.QWidget()
      flowStyleContainer.setContentsMargins(0,0,0,0)
      flowStyleLayout = QtWidgets.QVBoxLayout()
      flowStyleLayout.setContentsMargins(0,0,0,0)
      flowStyleSintelButton = QtWidgets.QRadioButton('&Sintel')
      flowStyleSintelButton.clicked.connect(partial(self.setFlowStyle, 0))
      flowStyleMiddleburyButton = QtWidgets.QRadioButton('&Middlebury')
      flowStyleMiddleburyButton.setChecked(True)
      flowStyleMiddleburyButton.clicked.connect(partial(self.setFlowStyle, 1))
      flowStyleButtonGroup = QtWidgets.QButtonGroup()
      flowStyleButtonGroup.addButton(flowStyleSintelButton)
      flowStyleButtonGroup.addButton(flowStyleMiddleburyButton)
      flowStylesLabel = QtWidgets.QLabel()
      flowStylesLabel.setText('Flow style')
      flowStyleLayout.addWidget(flowStylesLabel)
      flowStyleLayout.addWidget(flowStyleSintelButton)
      flowStyleLayout.addWidget(flowStyleMiddleburyButton)
      flowStyleContainer.setLayout(flowStyleLayout)
      toolsLayout.addWidget(flowStyleContainer)

    mainlayout.addWidget(toolsContainer)
    self.mainlayoutcontainer = QtWidgets.QWidget()
    self.mainlayoutcontainer.setLayout(mainlayout)
    self.setCentralWidget(self.mainlayoutcontainer)

    # Status bar with resize grip, for status messages
    status = self.statusBar()
    status.setSizeGripEnabled(True)
    self.updateStatus("Ready")

    ## Actions
    # Open one image set
    #fileOpenAction  = self.createAction("&Open folder",
    #                                    self.fileOpen,
    #                                    None,
    #                                    "Open a folder containing images")

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
    #self.addActions(self.fileMenu, (fileOpenAction,
    #                                None, 
    #                                fileQuitAction,))
    self.addActions(self.fileMenu, (fileQuitAction,))
    self.helpMenu = self.menuBar().addMenu("&Help")
    self.addActions(self.helpMenu, (helpAboutAction,))
    #fileToolBar = self.addToolBar("File")
    #fileToolBar.setObjectName("FileToolBar")
    #self.addActions(fileToolBar, (fileOpenAction,))

    self.setWindowTitle(QtWidgets.QApplication.applicationName())
    
    self.frameSlider.valueChanged.connect(self.frameSliderChange)
    if self.needFloatTools:
      self.floatMinSlider.valueChanged.connect(self.floatMinSliderChange)
      self.floatMaxSlider.valueChanged.connect(self.floatMaxSliderChange)
    if self.needFlowTools:
      self.flowScaleSlider.valueChanged.connect(self.flowScaleSliderChange)

    ## if a data folder was provided as a command line argument, use it
    if dir is not None:
      self.fileOpen(dir)


  def setFlowStyle(self, newStyle):
    '''Assign to global flowStyle'''
    global flowStyle
    flowStyle = newStyle
    self.changeFrame(self.currentIndex)
    self.flowScaleSliderChange(self.flowScaleSlider.value())
    self.updateStatus('Flow display style set to %s' \
                      %(['Sintel', 'Middlebury'][flowStyle]))


  def resizeEvent(self, resizeEvent):
    if self.grid[0].filenames:
      self.changeFrame(self.currentIndex)


  def frameSliderChange(self, newValue):
    """React to the user manipulating the frame control slider"""
    if newValue == self.currentIndex:
      return
    self.changeFrame(newValue)
    self.updateStatus('Frame %d'%(newValue))


  def floatMinSliderChange(self, newValue):
    """React to the user manipulating the float min control slider"""
    global floatMin
    floatMin = 0.01*newValue
    for cell in self.grid:
      if isinstance(cell, FloatCell):
        cell.index(self.currentIndex)
    self.floatMinSliderLabel.setText("Float min: %.2f" % (floatMin))
    if self.needFloatTools:
      self.floatMinSlider.setValue(newValue)
    self.updateStatus('Float minimum set to %f'%(floatMin))

    
  def floatMaxSliderChange(self, newValue):
    """React to the user manipulating the float max control slider"""
    global floatMax
    floatMax = 0.01*newValue
    for cell in self.grid:
      if isinstance(cell, FloatCell):
        cell.index(self.currentIndex)
    self.floatMaxSliderLabel.setText("Float max: %.2f" % (floatMax))
    if self.needFloatTools:
      self.floatMaxSlider.setValue(newValue)
    self.updateStatus('Float maximum set to %f'%(floatMax))


  def flowScaleSliderChange(self, newValue):
    """React to the user manipulating the flow scale control slider"""
    global flowScale
    if flowStyle == 0:    ## Sintel
      flowScale = 0.01*newValue
    elif flowStyle == 1:  ## Middlebury
      flowScale = 0.0005*newValue

    for cell in self.grid:
      if isinstance(cell, FlowCell):
        cell.index(self.currentIndex, flowScale)
    self.flowScaleSliderLabel.setText("OF scale: %.3f" % (flowScale))
    if self.needFlowTools:
      self.flowScaleSlider.setValue(newValue)
    self.updateStatus('Flow scale set to %f'%(flowScale))


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


  def fileOpen(self, folder=None):
    """Load a set of images, discovered from one specimen selected by the user"""
    ### If no folder is given, ask the user to choose one
    #if folder is None:
    #  header = "Choose folder"
    #  folder = QtWidgets.QFileDialog.getExistingDirectory(
    #            self,
    #            "%s - %s" % (QtWidgets.QApplication.applicationName(), 
    #                         header),
    #            '.')
    #  global dir
    #  dir = folder

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

    ## Find files
    suffices = [cell.suffix for cell in self.grid]
    found_filenames = GenerateFilenames(suffices)

    ## Generate permutation sequence (trivial ordered list if permute
    #  is disabled)
    global permute
    if permute:
      permute = list(range(len(found_filenames[0])))
      import random
      random.seed(0)
      random.shuffle(permute)
    else:
      permute = list(range(len(found_filenames[0])))

    print(permute)

    for (cell, files) in zip(self.grid, found_filenames):
      if isinstance(cell, EmptyCell):
        continue
      cell.clear()
      if not files:
        raise Exception('No filenames found. Did you provide a configuration?')
      for i in range(len(files)):
        cell.addImage(files[permute[i]])
        if preload:
          self.updateStatus("Reading images... %d/%d." % (i+1, len(files)))
    else:
      self.resetFrameSlider()
      self.changeFrame(0)


  def changeFrame(self, index):
    """Change the displayed frame"""
    self.currentIndex = index
    for cell in self.grid:
      cell.index(index)

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
  if len(sys.argv) > 1 and '--debug' in sys.argv[1:]:
    logging.basicConfig(level=logging.DEBUG)
  else:
    logging.basicConfig(level=logging.INFO)

  ## Try to use all command line arguments
  if len(sys.argv) > 1:
    for arg in sys.argv[1:]:
      if os.path.isdir(arg):
        global dir
        dir = arg
        logging.debug('Reading from folder >%s<'%(arg))
      elif arg.endswith('.cfg') and os.path.isfile(arg):
        readConfig(arg)
        logging.debug('Using config file >%s<'%(arg))
      elif arg == '--preload':
        global preload
        preload = True
        logging.debug('Will preload data')
      elif arg == '--no-permute':
        global permute
        permute = False
        logging.debug('Will not permute data')

  ## If no explicit configuration file was given, check if the data
  #  folder contains a 'viewer.cfg' and use it, else use a safe
  #  default.
  if configuration is None:
    if dir is not None:
      cfg = os.path.join(dir, 'viewer.cfg')
      if os.path.isfile(cfg):
        readConfig(cfg)
        logging.debug('Using config file >viewer.cfg< in data folder')
  if configuration is None:
    ## Default configuration
    parseConfig(['1 1', '0 0 img0.ppm'])
    logging.debug('Using default configuration (probably not a good idea)')

  app = QtWidgets.QApplication(sys.argv)
  app.setApplicationName("Gridview")
  form = MainWindow()
  form.show()
  app.exec_()

if __name__ == '__main__':
  main()

