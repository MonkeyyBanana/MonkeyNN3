package com.stinkymonkey.NeuralNetworkV3;

public class NetTrainer {
	NeuralNetwork nn;
	float[][] TRAINING_DATA;
	float[][] TRAINING_GOAL;
	
	public NetTrainer(NeuralNetwork nn, float[][] trainingData, float[][] trainingGoal) {
		this.nn = nn;
		TRAINING_DATA = trainingData;
		TRAINING_GOAL = trainingGoal;
	}
	
	public void trainBackprop(int iterations) {
		for (int i = 0; i < iterations; i++) {
			for (int j = 0; j < TRAINING_GOAL.length; j++) {
				nn.feedForward(TRAINING_DATA[j]);
				nn.backProp(TRAINING_GOAL[j]);
			}
		}
	}
	
	public void trainMutation(int iterations, float fitness) {
		
	}
}
