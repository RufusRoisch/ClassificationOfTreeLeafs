import PythonFiles.CNN as CNN
import PythonFiles.PretrainedCNN as preCNN
import PythonFiles.AugmentationCNN as augCNN

if __name__ == '__main__':
    print("Which model to you want to try out?")
    print("1: CNN, 2: CNN_hsv, 3: CNN_bw, 4: PretrainedCNN, 5: CNN_aug")

    whichModel = input()

    # runs improved base model
    if whichModel.__eq__("1"):
        CNN.run_model(1)

    # runs improved base model with hsv data
    elif whichModel.__eq__("2"):
        CNN.run_model(2)

    # runs improved base model with bw data
    elif whichModel.__eq__("3"):
        CNN.run_model(3)

    # runs vgc-19 pretrained model
    elif whichModel.__eq__("4"):
        preCNN.run_model()

    # runs base model with augmented data
    elif whichModel.__eq__("5"):
        augCNN.run_model()

    # prints error message if input was wrong
    else:
        print("ERROR: There is no model with the number -{}-".format(whichModel))
