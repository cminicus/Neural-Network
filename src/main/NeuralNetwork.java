package main;

import java.io.IOException;
import java.util.Arrays;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import utils.Instance;
import utils.MNISTReader;

/**
 * A class that contains the main method to load and train the neural network
 * @author Clayton Minicus
 * @author Tyler TerBush
 *
 */
public class NeuralNetwork {

	// The set of training instances
	private static Instance[] trainInstances;
	// The set of validation isntances
	private static Instance[] validationInstances;
	// The set of testing instances
	private static Instance[] testInstances;

	// The command line options
	private static Options options;

	// The name for the training images file
	private static String trainImagesFileName;
	// The name for the training labels file
	private static String trainLabelsFileName;
	// The name for the testing images file
	private static String testImagesFileName;
	// The name for the testing labels file
	private static String testLabelsFileName;

	/**
	 * Creates the command line options, parses the options, reads the files, and trains the network
	 * @param args The command line arguments
	 * @throws IOException
	 * @throws ParseException
	 */
	public static void main(String[] args) throws IOException, ParseException {

		createOptions();
		parseOptions(args);
		readFiles();
		trainNeuralNetwork();
	}

	/**
	 * Trains the neural network
	 */
	private static void trainNeuralNetwork() {
		int[] layers = {784, 30, 10};
		SGDNeuralNetwork network = new SGDNeuralNetwork(layers, 30, 10, 3.0);
		network.train(trainInstances, testInstances);
	}

	/**
	 * Creates the command line options
	 */
	private static void createOptions() {
		options = new Options();

		options.addOption("train_images", true, "train image files");
		options.addOption("train_labels", true, "train label files");

		options.addOption("test_images", true, "test image files");
		options.addOption("test_labels", true, "test label files");
	}

	private static void parseOptions(String[] args) throws ParseException {
		CommandLineParser parser = new DefaultParser();
		CommandLine commandLine = parser.parse(options, args);

		if (commandLine.hasOption("train_images")) {
			trainImagesFileName = commandLine.getOptionValue("train_images");
		} else {
			error("Missing -train_images argument");
		}

		if (commandLine.hasOption("train_labels")) {
			trainLabelsFileName = commandLine.getOptionValue("train_labels");
		} else {
			error("Missing -train_labels argument");
		}

		if (commandLine.hasOption("test_images")) {
			testImagesFileName = commandLine.getOptionValue("test_images");
		} else {
			error("Missing -test_images argument");
		}

		if (commandLine.hasOption("test_labels")) {
			testLabelsFileName = commandLine.getOptionValue("test_labels");
		} else {
			error("Missing -test_labels argument");
		}
	}

	private static void readFiles() throws IOException {
		Instance[] allTrainingInstances = MNISTReader.read(trainImagesFileName, trainLabelsFileName);

		// use first 50,000 instances as a training set
		trainInstances = Arrays.copyOfRange(allTrainingInstances, 0, 50000);

		// use last 10,000 instances as a validation set
		validationInstances = Arrays.copyOfRange(allTrainingInstances, 50000, 60000);

		testInstances = MNISTReader.read(testImagesFileName, testLabelsFileName);
	}

	/**
	 * Prints to standard error and exits the program
	 * @param string The string to print
	 */
	private static void error(Object string) {
		System.out.println("ERROR: " + string);
		System.exit(0);
	}
}
