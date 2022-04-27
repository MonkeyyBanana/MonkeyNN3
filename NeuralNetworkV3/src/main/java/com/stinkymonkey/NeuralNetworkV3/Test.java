package com.stinkymonkey.NeuralNetworkV3;

public class Test {
	// Back Propagation
	private static float LEARNING_RATE = 0.1f;
	
	public static void main(String[] args) {
		float[] input = {0.5f, 0.10f, 2.2f, 4.2f, 1.2f};
		float[] out = {0.25f, 0.05f, 1.1f, 2.1f, 0.6f};
	
		// *C = (out - goal)^2 ACTIVATOR FUNC SQUARE
		run(input, out);
	}
	
	private static float Deriv(float x) {
		return x * (1 - x);
	}
	
	private static void run(float[] input, float[] goal) {
		
	}
	
	private static float[] calculateError(float[] goal, float[] out, float[] input) {
		float[] edit = new float[goal.length];
		
		for (int i = 0; i < goal.length; i++) {
			float error = goal[i] - out[i];
			float adjust = error * out[i] * (1 - out[i]);
			edit[i] = input[i] * adjust;
		}
		
		return edit;
	}
}
