package main;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import Jama.Matrix;
import utils.Instance;

/**
 * This is an implementation of a neural network using the
 * stochastic gradient descent algorithm for backpropogation
 * with a cross-entropy cost function, L2 regularization, and altered starting weights
 * 
 * @author Clayton Minicus
 * @author Tyler TerBush
 *
 */
public class ImprovedSGDNeuralNetwork {

	// array of network layer sizes
	private int[] layers;
	// number of layers in the network
	private int numberOfLayers;
	// number of iterations
	private int iterations;
	// size of each mini-batch
	private int batchSize;
	// learning rate
	private double eta;
	// regularization parameter
	private double lambda;

	// per-layer network biases
	private double[][] biases;
	// weights connecting each layer of the network
	private Matrix[] weights;

	/**
	 * A class used to hold a set of biases and weights
	 */
	private class Tuple {
		double[][] biases;
		Matrix[] weights;

		Tuple(double[][] biases, Matrix[] weights) {
			this.biases = biases;
			this.weights = weights;
		}
	}

	/**
	 * Creates an instance of a SGDNeuralNetwork
	 * @param layers An array of layer sizes
	 * @param iterations The number of iterations to train over
	 * @param batchSize The size of each mini-batch
	 * @param eta The learning rate
	 */
	public ImprovedSGDNeuralNetwork(int[] layers, int iterations, int batchSize, double eta, double lambda) {
		this.layers = layers;
		this.numberOfLayers = layers.length;
		this.iterations = iterations;
		this.batchSize = batchSize;
		this.eta = eta;
		this.lambda = lambda;

		biases = initializeBiases(true);
		weights = initializeWeights(true);
	}

	/**
	 * Initializes an array of biases from the second layer and up.
	 * The first layer is always the input layer and so does not have any biases.
	 * @param gaussian A boolean to determine whether or not to initialize the biases with
	 * a random gaussian distribution with mean 0 and variance 1
	 * @return The initialized biases
	 */
	private double[][] initializeBiases(boolean gaussian) {
		// create biases
		double[][] biases = new double[numberOfLayers - 1][];

		// iterate through layer sizes
		for (int layer = 1; layer < numberOfLayers; layer++) {
			int layerSize = layers[layer];
			double[] bias = new double[layerSize];
			Random r = new Random();
			// initialize each layer to random gaussian distribution
			if (gaussian) {
				for (int i = 0; i < layerSize; i++) {
					bias[i] = r.nextGaussian();
				}
			}
			// assign bias layer and increment counter
			biases[layer - 1] = bias;
		}
		return biases;
	}

	/**
	 * Initializes an array of weights.
	 * The weights determine the relationship between two layers in the network
	 * @param gaussian A boolean to determine whether or not to initialize the weights with
	 * a random gaussian distribution with mean 0 and variance 1
	 * @return The initialized weights
	 */
	private Matrix[] initializeWeights(boolean gaussian) {

		Matrix[] weights = new Matrix[numberOfLayers - 1];

		for (int a = 0; a < numberOfLayers - 1; a++) {
			int layerSize = layers[a];
			int nextLayerSize = layers[a + 1];

			Matrix matrix = new Matrix(nextLayerSize, layerSize);
			Random r = new Random();

			if (gaussian) {
				for (int i = 0; i < matrix.getRowDimension(); i++) {
					for (int j = 0; j < matrix.getColumnDimension(); j++) {
						// FIRST CHANGE: reducing variance of weight distribution
						matrix.set(i, j, r.nextGaussian() / Math.sqrt((double) layerSize));
					}
				}
			}

			weights[a] = matrix;
		}
		return weights;
	}

	/**
	 * Trains the SGDNeuralNetwork using the given training data and evaluates the network
	 * using the testing data set every iteration
	 * @param train The training set to use
	 * @param test The testing set to use after each iteration
	 */
	public void train(Instance[] train, Instance[] test) {
		// get instances and length
		Instance[] instances = train;
		int numberOfInstances = instances.length;

		// get formatter
		DecimalFormat formatter = new DecimalFormat("#.##");
		System.out.println("Training...");

		// repeat for this.iterations
		for (int iterations = 0; iterations < this.iterations; iterations++) {

			// print progress
			String percentage = formatter.format((double) iterations / (double) this.iterations * 100.0);
			System.out.print(percentage + "%\r");

			// shuffle the array of instances for each iteration
			shuffle(instances);

			// segment into mini batches
			ArrayList<Instance[]> miniBatches = new ArrayList<>();
			for (int batch = 0; batch < numberOfInstances; batch += batchSize) {
				miniBatches.add(Arrays.copyOfRange(instances, batch, batch + batchSize));
			}

			// update for each mini batch
			for (Instance[] batch : miniBatches) {
				update(batch, numberOfInstances);
			}

			// evaluate tests
			evaluate(test);
		}
		System.out.println("Finished");
	}

	/**
	 * Determines output activations for given input activations
	 * @param initialActivations The initial activations for the network
	 * @return An array of doubles containing the final output activations
	 */
	private double[] feedForward(double[] initialActivations) {
		double[] a = initialActivations;
		for (int i = 0; i < numberOfLayers - 1; i++) {
			Matrix weight = this.weights[i];
			double[] bias = this.biases[i];
			// a = sigma(wx + b)
			a = sigmoid(addVectors(matrixVectorDotProduct(weight, a), bias));
		}
		return a;
	}

	/**
	 * Updates the neural network for a given batch of instances
	 * @param batch The mini-batch to update the network with
	 */
	private void update(Instance[] batch, int numberOfInstances) {
		double[][] learningBiases = initializeBiases(false);
		Matrix[] learningWeights = initializeWeights(false);

		for (Instance instance : batch) {
			// calculate back propagation
			Tuple delta = backPropogate(instance);
			// get delta biases and delta weights
			double[][] deltaBiases = delta.biases;
			Matrix[] deltaWeights = delta.weights;

			// update biases and weights
			for (int b = 0; b < learningBiases.length; b++) {
				learningBiases[b] = addVectors(learningBiases[b], deltaBiases[b]);
			}

			for (int w = 0; w < learningWeights.length; w++) {
				learningWeights[w] = learningWeights[w].plus(deltaWeights[w]);
			}
		}

		double batchLearningRate = eta / (double) batch.length;
		double weightRegularization = 1.0 - eta * (lambda / (double) numberOfInstances);

		// b = this.b - learnRate * deltaB
		for (int b = 0; b < learningBiases.length; b++) {
			this.biases[b] = subtractVectors(this.biases[b], multiplyVectorScalar(learningBiases[b], batchLearningRate));
		}

		// w = this.w - learnRate * deltaW
		for (int w = 0; w < learningWeights.length; w++) {
			this.weights[w] = this.weights[w].times(weightRegularization).minus(learningWeights[w].times(batchLearningRate));
		}
	}

	/**
	 * The back propogation algorithm which executes a forward pass, calculates the cost,
	 * and executes a backwards pass to update the weigths and biases
	 * @param instance The instance to calculate the updates for
	 * @return A Tuple containing the appropriate change in biases and weights
	 */
	private Tuple backPropogate(Instance instance) {

		// blank biases and weights
		double[][] learningBiases = initializeBiases(false);
		Matrix[] learningWeights = initializeWeights(false);

		// first activation is the input
		double[] activation = instance.getFeatureVector();
		// array of activations
		double[][] activations = new double[numberOfLayers][];
		activations[0] = activation;

		// keeps track of pre-sigmoid values
		double[][] zs = new double[numberOfLayers - 1][];

		// ----------- feed forward --------------
		for (int i = 0; i < numberOfLayers - 1; i++) {
			Matrix weight = this.weights[i];
			double[] bias = this.biases[i];

			// get pre-sigmoid values
			double[] z = addVectors(matrixVectorDotProduct(weight, activation), bias);
			// add it to the array
			zs[i] = z;
			// activation function
			activation = sigmoid(z);
			// store the new activations
			activations[i + 1] = activation;
		}

		// ---------- backwards pass --------------
		// SECOND CHANGE - no mutliplying by sigmoid_prime here
		double[] delta = costDerivative(activations[activations.length - 1], instance.getLabel());
		learningBiases[learningBiases.length - 1] = delta;
		learningWeights[learningWeights.length - 1] = vectorToMatrixMultiplication(delta, activations[activations.length - 2]);

		for (int i = 2; i < numberOfLayers; i++) {
			double[] z = zs[zs.length - i];
			double[] sigmoidPrime = sigmoidPrime(z);
			delta = multiplyVectors(matrixVectorDotProduct(this.weights[this.weights.length - i + 1].transpose(), delta), sigmoidPrime);
			learningBiases[learningBiases.length - i] = delta;
			learningWeights[learningWeights.length - i] = vectorToMatrixMultiplication(delta, activations[activations.length - i - 1]);
		}

		return new Tuple(learningBiases, learningWeights);
	}

	/**
	 * Evaluates the network for the given instances
	 * @param instances The instances to use to evaluate the network
	 */
	public void evaluate(Instance[] instances) {
		int correct = 0;
		int total = instances.length;
		for (Instance instance : instances) {
			double[] output = feedForward(instance.getFeatureVector());
			int prediction = convertVectorToLabel(output);
			if (prediction == instance.getLabel()) {
				correct++;
			}
		}
		System.out.println("Correct: " + (double) correct / (double) total * 100.0 + "%");
	}

	/**
	 * Calculates the activation for a given input
	 * @param z The input for which to calculate the activation
	 * @return The activation value
	 */
	private double sigmoid(double z) {
		return 1.0 / (1.0 + Math.exp(-z));
	}

	/**
	 * Calculates the activation for an array of inputs
	 * @param z The array for which to calculate the activations
	 * @return An array of calculated activations
	 */
	private double[] sigmoid(double[] z) {
		double[] ret = new double[z.length];
		for (int i = 0; i < z.length; i++) {
			ret[i] = sigmoid(z[i]);
		}
		return ret;
	}

	/**
	 * Calculates the derivative of the activation function
	 * @param z The array for which to calculate the derivatives
	 * @return An array of calculated derivatives
	 */
	private double[] sigmoidPrime(double[] z) {
		return multiplyVectors(sigmoid(z), subtractVectorFromScalar(sigmoid(z), 1.0));
	}

	/**
	 * Calculates the cost derivatives for the given activations compared to the correct answer
	 * using a quadratic cost function
	 * @param activations The output activations to calculate the cost for
	 * @param label The correct label for the activations
	 * @return The array of costs for each output activation
	 */
	private double[] costDerivative(double[] activations, int label) {
		double[] labelVector = convertLabelToVector(label);
		return subtractVectors(activations, labelVector);
	}

	/**
	 * Converts a label to a vector representation
	 * @param label The label to convert
	 * @return An array representation of the label
	 */
	private double[] convertLabelToVector(int label) {
		double[] ret = new double[10];
		ret[label] = 1.0;
		return ret;
	}

	/**
	 * Converts a vector into the corresponding label
	 * @param vector The vector to convert
	 * @return The label representation
	 */
	private int convertVectorToLabel(double[] vector) {
		int ret = -1;
		double largestValue = -1.0;
		for (int i = 0; i < vector.length; i++) {
			if (vector[i] > largestValue) {
				largestValue = vector[i];
				ret = i;
			}
		}
		return ret;
	}

	/**
	 * Calculates the dot product of a matrix and a vector
	 * @param matrix The matrix to calculate the dot product with
	 * @param vector The vector to calculate the dot product with
	 * @return The vector representing the dot product
	 */
	private double[] matrixVectorDotProduct(Matrix matrix, double[] vector) {
		assert matrix.getColumnDimension() == vector.length;

		double[] ret = new double[matrix.getRowDimension()];
		for (int i = 0; i < matrix.getRowDimension(); i++) {
			double sum = 0.0;
			for (int j = 0; j < vector.length; j++) {
				sum += matrix.get(i, j) * vector[j];
			}
			ret[i] = sum;
		}
		return ret;
	}

	/**
	 * Calculates an element-wise addition of two vectors
	 * @param vector1 The first vector
	 * @param vector2 The second vector
	 * @return The vector representing the addition
	 */
	private double[] addVectors(double[] vector1, double[] vector2) {
		assert vector1.length == vector2.length;

		double[] ret = new double[vector1.length];
		for (int i = 0; i < vector1.length; i++) {
			ret[i] = vector1[i] + vector2[i];
		}
		return ret;
	}

	/**
	 * Calculates an element-wise subtraction of two vectors
	 * @param vector1 The first vector
	 * @param vector2 The second vector
	 * @return The vector representing the subtraction
	 */
	private double[] subtractVectors(double[] vector1, double[] vector2) {
		assert vector1.length == vector2.length;

		double[] ret = new double[vector1.length];
		for (int i = 0; i < ret.length; i++) {
			ret[i] = vector1[i] - vector2[i];
		}
		return ret;
	}

	/**
	 * Calculates an element-wise multiplication of two vectors
	 * @param vector1 The first vector
	 * @param vector2 The second vector
	 * @return The vector representing the multiplication
	 */
	private double[] multiplyVectors(double[] vector1, double[] vector2) {
		assert vector1.length == vector2.length;

		double[] ret = new double[vector1.length];
		for (int i = 0; i < ret.length; i++) {
			ret[i] = vector1[i] * vector2[i];
		}
		return ret;
	}

	/**
	 * Calculates an element-wise multiplication of a vector and a scalar
	 * @param vector The vector
	 * @param scalar The scalar value
	 * @return The vector representing the scalar multiplication
	 */
	private double[] multiplyVectorScalar(double[] vector, double scalar) {
		double[] ret = new double[vector.length];
		for (int i = 0; i < ret.length; i++) {
			ret[i] = vector[i] * scalar;
		}
		return ret;
	}

	/**
	 * Calculates an element-wise subtraction of a vector from a scalar
	 * @param vector The vector
	 * @param scalar The scalar value
	 * @return The vector representing the scalar subtraction
	 */
	private double[] subtractVectorFromScalar(double[] vector, double scalar) {
		double[] ret = new double[vector.length];
		for (int i = 0; i < ret.length; i++) {
			ret[i] = scalar - vector[i];
		}
		return ret;
	}

	/**
	 * Calculates a Matrix using two input vectors
	 * @param vector1 The first vector
	 * @param vector2 The second vector
	 * @return The Matrix representing the multiplication of two vectors
	 */
	private Matrix vectorToMatrixMultiplication(double[] vector1, double[] vector2) {
		Matrix ret = new Matrix(vector1.length, vector2.length);
		for (int i = 0; i < vector1.length; i++) {
			for (int j = 0; j < vector2.length; j++) {
				ret.set(i, j, vector1[i] * vector2[j]);
			}
		}
		return ret;
	}

	/**
	 * Performs a Fisher-Yates Shuffle of the input array
	 * @param instances The array of instances to shuffle in place
	 */
	private static void shuffle(Instance[] instances) {
		Random random = ThreadLocalRandom.current();
		for (int i = instances.length - 1; i > 0; i--) {
			int index = random.nextInt(i + 1);
			// Simple swap
			Instance a = instances[index];
			instances[index] = instances[i];
			instances[i] = a;
		}
	}
}
