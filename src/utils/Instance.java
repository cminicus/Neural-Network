package utils;

public class Instance {
	
	private int label;
	private double[] featureVector;
	
	public Instance(int label, double[] featureVector) {
		this.label = label;
		this.featureVector = featureVector;
	}
	
	public int getLabel() {
		return this.label;
	}
	
	public double[] getFeatureVector() {
		return this.featureVector;
	}
}
