class prediction:
    def __init__(self, classifier):
        self.model = classifier

    def Predict(self, inp_img):
        pred = self.model.predict(inp_img).argmax()
        return pred