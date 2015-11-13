package utils;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.text.DecimalFormat;

public class MNISTReader {

	public static Instance[] read(String imageFileName, String labelFileName) throws IOException {

		// get data streams
		DataInputStream labels = new DataInputStream(new FileInputStream(labelFileName));
		DataInputStream images = new DataInputStream(new FileInputStream(imageFileName));

		// make sure magic numbers are correct
		int magicNumber = labels.readInt();
		if (magicNumber != 2049) {
			System.err.println("Label file has wrong magic number: " + magicNumber + " (should be 2049)");
			System.exit(0);
		}
		magicNumber = images.readInt();
		if (magicNumber != 2051) {
			System.err.println("Image file has wrong magic number: " + magicNumber + " (should be 2051)");
			System.exit(0);
		}

		// get proper numbers
		int numLabels = labels.readInt();
		int numImages = images.readInt();
		int numRows = images.readInt();
		int numCols = images.readInt();

		// make sure numbers align
		if (numLabels != numImages) {
			System.err.println("Image file and label file do not contain the same number of entries.");
			System.err.println("  Label file contains: " + numLabels);
			System.err.println("  Image file contains: " + numImages);
			System.exit(0);
		}

		// keep track of labels/images
		int numLabelsRead = 0;
		int numImagesRead = 0;
		
		// formatter for percentage updates
		DecimalFormat formatter = new DecimalFormat("#.##");
		System.out.println("Reading...");

		// create array of instances
		Instance[] instances = new Instance[numLabels];
		while (labels.available() > 0 && numLabelsRead < numLabels) {

			// get label
			int label = labels.readByte();
			numLabelsRead++;
			
			// get vector
			FeatureVector vector = new FeatureVector(numCols * numRows);
			for (int index = 0; index < numCols * numRows; index++) {
				vector.put(index, images.readUnsignedByte());
			}
			
			// create and add instance
			Instance instance = new Instance(label, vector);
			instances[numImagesRead] = instance;

			numImagesRead++;
			
			// print progress
			String percentage = formatter.format((double) numImagesRead / (double) numImages * 100.0);
			System.out.print(percentage + "%\r");
		}
		
		// close streams
		labels.close();
		images.close();

		System.out.println("Finished");
		
		// return instances
		return instances;
	}
}
