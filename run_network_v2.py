# tag::test_setup[]
from dlgo.nn import load_mnist
from dlgo.nn import network
from dlgo.nn.layers_v2 import DenseLayer, ActivationLayer
import argparse

def main(epochs):
	print(f"Training for {epochs} epochs")
	training_data, test_data = load_mnist.load_data()  # <1>

	net = network.SequentialNetwork()  # <2>

	net.add(DenseLayer(784, 392))  # <3>
	net.add(ActivationLayer(392))
	net.add(DenseLayer(392, 196))
	net.add(ActivationLayer(196))
	net.add(DenseLayer(196, 10))
	net.add(ActivationLayer(10))  # <4>

	# <1> First, load training and test data.
	# <2> Next, initialize a sequential neural network.
	# <3> You can then add dense and activation layers one by one.
	# <4> The final layer has size 10, the number of classes to predict.
	# end::test_setup[]

	# tag::test_run[]
	net.train(training_data, epochs=epochs, mini_batch_size=10,
			  learning_rate=3.0, test_data=test_data)  # <1>

	# <1> You can now easily train the model by specifying train and test data, the number of epochs, the mini-batch size and the learning rate.
	# end::test_run[]

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Train mininst.")
	parser.add_argument('epochs', type=int, help='Number of epochs', nargs='?', default=1)
	args = parser.parse_args()

	main(args.epochs)
	
