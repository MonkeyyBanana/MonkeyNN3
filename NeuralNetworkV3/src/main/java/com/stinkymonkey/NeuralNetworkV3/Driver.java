package com.stinkymonkey.NeuralNetworkV3;

public class Driver {
	public static void main(String[] args) {
		NeuralNetwork nn = new NeuralNetwork(new int[] {3, 25, 25, 1});
		float[][] samp = {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}};
		float[][] goal = {{0}, {1}, {1}, {0}, {1}, {0}, {0}, {1}};
		
		nn.layers[1].MAX_BIAS = 0.05f;
		nn.layers[2].MAX_BIAS = 0.05f;
		
		NetTrainer trainer = new NetTrainer(nn, samp, goal);
		trainer.trainBackprop(50000);
		
		for (int j = 0; j < goal.length; j++) {
			System.out.println(nn.feedForward(samp[j])[0]);
		}
	}
}
