package main;

import java.io.IOException;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import utils.Instance;
import utils.MNISTReader;

public class NeuralNetwork {
	
	private static Instance[] trainInstances;
	private static Instance[] testInstances;
	
	private static Options options;
	
	private static String trainImagesFileName;
	private static String trainLabelsFileName;
	private static String testImagesFileName;
	private static String testLabelsFileName;

	public static void main(String[] args) throws IOException, ParseException {
		
		createOptions();
		parseOptions(args);
		readFiles();
		trainNeuralNetwork();
	}
	
	private static void trainNeuralNetwork() {
		int[] layers = {784, 30, 10};
		SGDNeuralNetwork network = new SGDNeuralNetwork(layers, 30, 10, 3.0);
		network.train(trainInstances, testInstances);
//		network.evaluate(testInstances);
	}
	
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
		trainInstances = MNISTReader.read(trainImagesFileName, trainLabelsFileName);
		testInstances = MNISTReader.read(testImagesFileName, testLabelsFileName);
	}

	private static void error(Object string) {
		System.out.println("ERROR: " + string);
		System.exit(0);
	}
}
