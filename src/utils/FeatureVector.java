package utils;

public class FeatureVector {

	private int[] features;
	
	public FeatureVector(int size) {
		features = new int[size];
	}
	
	public void put(int key, int feature) {
		features[key] = feature;
	}
	
	public int getFeature(int key) {
		return features[key];
	}
}
