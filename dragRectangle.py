import matplotlib
class DraggableRectangle:

    def __init__(self, rect, rectangle_number, text):

        self.rect = rect
        self.press = None
        self.x_rect = rect.xy[0]
        self.y_rect = rect.xy[1]
        self.rect.figure.canvas.draw()
        self.rectangle_number=rectangle_number
        self.text = text

    def connect(self):
        'connect to all the events we need'
        #print("DraggableRectangle connect()")
        self.rect.figure.text(self.x_rect, self.y_rect, str(self.rectangle_number))
        self.cidpress = self.rect.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.rect.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.rect.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        #print("DraggableRectangle on_press()")
        if event.inaxes != self.rect.axes: return
        contains, attrd = self.rect.contains(event)
        if not contains: return
        x0, y0 = self.rect.xy
        self.press = x0, y0, event.xdata, event.ydata

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        # print("DraggableRectangle on_motion()")
        if self.press is None: return
        if event.inaxes != self.rect.axes: return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.rect.set_x(x0 + dx)
        self.rect.set_y(y0 + dy)
        self.rect.figure.canvas.draw()
        #self.ax.text(x0 + dx,y0 + dy, str(self.rectangle_number))
        self.text.set_position(xy=(x0 + dx,y0 + dy))

    def on_release(self, event):
        self.x_rect, self.y_rect = self.rect.xy
        self.press = None
        self.rect.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)
