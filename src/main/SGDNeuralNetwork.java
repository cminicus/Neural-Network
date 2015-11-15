package main;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import Jama.Matrix;
import utils.Instance;

public class SGDNeuralNetwork {

	private int[] layers;
	private int numberOfLayers;
	private int iterations;
	private int batchSize;
	private double eta;

	private double[][] biases;
	private Matrix[] weights;
	
	private class Tuple {
		double[][] biases;
		Matrix[] weights;
		
		Tuple(double[][] biases, Matrix[] weights) {
			this.biases = biases;
			this.weights = weights;
		}
	}

	public SGDNeuralNetwork(int[] layers, int iterations, int batchSize, double eta) {
		this.layers = layers;
		this.numberOfLayers = layers.length;
		this.iterations = iterations;
		this.batchSize = batchSize;
		this.eta = eta;

		biases = initializeBiases(true);
		weights = initializeWeights(true);
	}

	// initialize biases with random gaussian distributions or 0s
	private double[][] initializeBiases(boolean gaussian) {
		// create biases
		double[][] biases = new double[numberOfLayers - 1][];

		// iterate through layer sizes
		for (int layer = 1; layer < layers.length; layer++) {
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

	// initialize weights with random gaussian distributions
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
						matrix.set(i, j, r.nextGaussian());
					}
				}
			}

			weights[a] = matrix;
		}
		return weights;
	}

//	public void train(Instance[] i) {
	public void train(Instance[] i, Instance[] test) {
		// get instances and length
		Instance[] instances = i;
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
				update(batch);
			}
			evaluate(test);
		}
		System.out.println("Finished");
	}

	private double[] feedForward(double[] initialActivations) {
		double[] a = initialActivations;
		for (int i = 0; i < numberOfLayers - 1; i++) {
			Matrix weight = weights[i];
			double[] bias = biases[i];
			// a = wx + b
			a = sigmoid(addVectors(matrixVectorDotProduct(weight, a), bias));
		}
		return a;
	}
	
	private void update(Instance[] batch) {
		double[][] biases = initializeBiases(false);
		Matrix[] weights = initializeWeights(false);
		
		for (Instance instance : batch) {
			// calculate back propagation
			Tuple delta = backPropogate(instance);
			// get delta biases and delta weights
			double[][] deltaBiases = delta.biases;
			Matrix[] deltaWeights = delta.weights;
			
			// update biases and weights
			for (int b = 0; b < biases.length; b++) {
				biases[b] = addVectors(biases[b], deltaBiases[b]);
			}
			
			for (int w = 0; w < weights.length; w++) {
				weights[w] = weights[w].plus(deltaWeights[w]);
			}
		}
		
		double batchLearningRate = eta / (double) batch.length;
		
		// b = this.b - learnRate * deltaB
		for (int b = 0; b < biases.length; b++) {
			biases[b] = subtractVectors(this.biases[b], multiplyVectorScalar(biases[b], batchLearningRate));
		}
		this.biases = biases;
		
		// w = this.w - learnRate * deltaW
		for (int w = 0; w < weights.length; w++) {
			weights[w] = this.weights[w].minus(weights[w].times(batchLearningRate));
		}
		this.weights = weights;
		
	}
	
	private Tuple backPropogate(Instance instance) {
		
		// blank biases and weights
		double[][] biases = initializeBiases(false);
		Matrix[] weights = initializeWeights(false);
		
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
		double[] delta = costDerivative(activations[activations.length - 1], instance.getLabel());
		delta = multiplyVectors(delta, sigmoidPrime(zs[zs.length - 1]));
		biases[biases.length - 1] = delta;
		weights[weights.length - 1] = vectorToMatrixMultiplication(delta, activations[activations.length - 2]);
		
		for (int i = 2; i < numberOfLayers; i++) {
			double[] z = zs[zs.length - i];
			double[] sigmoidPrime = sigmoidPrime(z);
			double[] dotProduct = matrixVectorDotProduct(weights[weights.length - i + 1].transpose(), delta);
			delta = multiplyVectors(dotProduct, sigmoidPrime);
			biases[biases.length - i] = delta;
			weights[weights.length - i] = vectorToMatrixMultiplication(delta, activations[activations.length - i - 1]);
		}
		
		return new Tuple(biases, weights);
	}
	
	public void evaluate(Instance[] instances) {
		int correct = 0;
		int total = instances.length;
		for (Instance instance : instances) {
			double[] output = feedForward(instance.getFeatureVector());
			int predicition = convertVectorToLabel(output);
			if (predicition == instance.getLabel()) {
				correct++;
			}
		}
		System.out.println("Correct: " + (double) correct / (double) total * 100.0);
	}
	
	private double sigmoid(double z) {
		return 1.0 / (1.0 + Math.exp(-z));
	}

	private double[] sigmoid(double[] z) {
		double[] ret = new double[z.length];
		for (int i = 0; i < z.length; i++) {
			ret[i] = sigmoid(z[i]);
		}
		return ret;
	}

	private double[] sigmoidPrime(double[] z) {
		return multiplyVectors(sigmoid(z), subtractVectorFromScalar(sigmoid(z), 1.0));
	}
	
	private double[] costDerivative(double[] activations, int label) {
		double[] labelVector = convertLabelToVector(label);
		return subtractVectors(activations, labelVector);
	}
	
	// convert label into vector of 0.0 except for label-th index
	private double[] convertLabelToVector(int label) {
		double[] ret = new double[10];
		ret[label] = 1.0;
		return ret;
	}
	
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
	
	private double[] matrixVectorDotProduct(Matrix matrix, double[] vector) {
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

	private double[] addVectors(double[] vector1, double[] vector2) {
		// assert lengths are equal?
		double[] ret = new double[vector1.length];
		for (int i = 0; i < vector1.length; i++) {
			ret[i] = vector1[i] + vector2[i];
		}
		return ret;
	}

	private double[] subtractVectors(double[] vector1, double[] vector2) {
		double[] ret = new double[vector1.length];
		for (int i = 0; i < ret.length; i++) {
			ret[i] = vector1[i] - vector2[i];
		}
		return ret;
	}
	
	private double[] multiplyVectors(double[] vector1, double[] vector2) {
		double[] ret = new double[vector1.length];
		for (int i = 0; i < ret.length; i++) {
			ret[i] = vector1[i] * vector2[i];
		}
		return ret;
	}
	
	private double[] multiplyVectorScalar(double[] vector, double scalar) {
		double[] ret = new double[vector.length];
		for (int i = 0; i < ret.length; i++) {
			ret[i] = vector[i] * scalar;
		}
		return ret;
	}
	
	private double[] subtractVectorFromScalar(double[] vector, double scalar) {
		double[] ret = new double[vector.length];
		for (int i = 0; i < ret.length; i++) {
			ret[i] = scalar - vector[i];
		}
		return ret;
	}
	
	private Matrix vectorToMatrixMultiplication(double[] vector1, double[] vector2) {
		Matrix ret = new Matrix(vector1.length, vector2.length);
		for (int i = 0; i < vector1.length; i++) {
			for (int j = 0; j < vector2.length; j++) {
				ret.set(i, j, vector1[i] * vector2[j]);
			}
		}
		return ret;
	}
	
	// Fisher-Yates Shuffle
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
