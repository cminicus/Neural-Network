package utils;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.PrintWriter;

public class MNISTReader {

	public static void main(String[] args) throws IOException {
	    DataInputStream labels = new DataInputStream(new FileInputStream(args[0]));
	    DataInputStream images = new DataInputStream(new FileInputStream(args[1]));
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
	    int numLabels = labels.readInt();
	    int numImages = images.readInt();
	    int numRows = images.readInt();
	    int numCols = images.readInt();
	    if (numLabels != numImages) {
	      System.err.println("Image file and label file do not contain the same number of entries.");
	      System.err.println("  Label file contains: " + numLabels);
	      System.err.println("  Image file contains: " + numImages);
	      System.exit(0);
	    }

	    long start = System.currentTimeMillis();
	    int numLabelsRead = 0;
	    int numImagesRead = 0;
	    
	    PrintWriter writer = new PrintWriter("output/mnist.test", "UTF-8");
	    
	    while (labels.available() > 0 && numLabelsRead < numLabels) {
	      byte label = labels.readByte();
	      numLabelsRead++;
//	      int[][] image = new int[numCols][numRows];
//	      for (int colIdx = 0; colIdx < numCols; colIdx++) {
//	        for (int rowIdx = 0; rowIdx < numRows; rowIdx++) {
//	          image[colIdx][rowIdx] = images.readUnsignedByte();
//	        }
//	      }
	      
	      //writer.write(label + " ");
	      int[] image = new int[numCols * numRows];
	      for (int index = 0; index < numCols * numRows; index++) {
	        image[index] = images.readUnsignedByte();
	        //writer.write((index + 1) + ":" + image[index] + " ");
	      }
	      numImagesRead++;
	      //writer.println();
	      

	      // At this point, 'label' and 'image' agree and you can do whatever you like with them.
	      if (numLabelsRead % 100 == 0) {
	    	  writer.write(label + " ");
	    	  for (int index = 0; index < numCols * numRows; index++) {
	  	        writer.write((index + 1) + ":" + image[index] + " ");
	  	      }
	    	  writer.println();
	      }
	      
	      
	      if (numLabelsRead % 10 == 0) {
	        System.out.print(".");
	      }
	      if ((numLabelsRead % 800) == 0) {
	        System.out.print(" " + numLabelsRead + " / " + numLabels);
	        long end = System.currentTimeMillis();
	        long elapsed = end - start;
	        long minutes = elapsed / (1000 * 60);
	        long seconds = (elapsed / 1000) - (minutes * 60);
	        System.out.println("  " + minutes + " m " + seconds + " s ");
	      }
	    }
	    writer.close();
	    System.out.println();
	    long end = System.currentTimeMillis();
	    long elapsed = end - start;
	    long minutes = elapsed / (1000 * 60);
	    long seconds = (elapsed / 1000) - (minutes * 60);
	    System.out
	        .println("Read " + numLabelsRead + " samples in " + minutes + " m " + seconds + " s ");
	  }

}
