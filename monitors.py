try:
    from monitors_extension import Presenter
except ImportError:
    import logging as log

    class Presenter:
        def __init__(self, keys, yPos=20, graphSize=(150, 60), historySize=20):
            self.yPos = yPos
            self.graphSize = graphSize
            self.graphPadding = 0
            if keys:
                log.warning("monitors_extension wasn't found")

        def handleKey(self, key):
            pass

        def drawGraphs(self, frame):
            pass

        def reportMeans(self):
            return ""
