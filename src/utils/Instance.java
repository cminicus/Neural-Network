package utils;

public class Instance {
	
	private int label;
	private FeatureVector featureVector;
	
	public Instance(int label, FeatureVector featureVector) {
		this.label = label;
		this.featureVector = featureVector;
	}
	
	public int getLabel() {
		return this.label;
	}
	
	public FeatureVector getFeatureVector() {
		return this.featureVector;
	}
}
