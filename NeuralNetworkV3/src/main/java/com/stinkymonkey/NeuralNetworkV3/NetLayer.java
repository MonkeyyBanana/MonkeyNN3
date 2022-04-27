package com.stinkymonkey.NeuralNetworkV3;

import java.util.Random;

public class NetLayer {
	private Random random = new Random(1);
	
	public static float MIN_BIAS = 0.0f;
	public static float MAX_BIAS = 0.0f;
	
	public static float MIN_CHANCE = 0.0f;
	public static float MAX_CHANCE = 0.0f;
	
	public float LEARNING_RATE = 0.0333f;
	
	int numOfInputs;
	int numOfOutputs;
	
	float[] outputs;
	float[] inputs;
	float[][] weights;
	float[][] weightsDelta;
	float[] bias;
	float[] biasDelta;
	float[] gamma;
	float[] error;
	
	public NetLayer(int numOfInputs, int numOfOutputs) {
		this.numOfInputs = numOfInputs;
		this.numOfOutputs = numOfOutputs;
		
		outputs = new float[numOfOutputs];
		inputs = new float[numOfInputs];
		weights = new float[numOfOutputs][numOfInputs];
		weightsDelta = new float[numOfOutputs][numOfInputs];
		bias = new float[numOfOutputs];
		biasDelta = new float[numOfOutputs];
		gamma = new float[numOfOutputs];
		error = new float[numOfOutputs];
		
		initWeights();
	}
	
	public void initWeights() {
		for (int i = 0; i < numOfOutputs; i++) {
			for (int j = 0; j < numOfInputs; j++) {
				weights[i][j] = random.nextFloat() - 0.5f;
			}
			bias[i] = MIN_BIAS + (MAX_BIAS - MIN_BIAS) * random.nextFloat();
		}
	}
	
	private float Deriv(float x) {
		return x * (1 - x);
	}
	
	public float[] feedForward(float[] inputs) {
		this.inputs = inputs;
		
		for (int i = 0; i < numOfOutputs; i++) {
			outputs[i] = 0;
			
			for (int j = 0; j < numOfInputs; j++) {
				outputs[i] += inputs[j] * weights[i][j];
			}
			
			outputs[i] += bias[i];
			outputs[i] = NetActivator.Tanh.activator(outputs[i]);
		}
		
		return outputs;
	}
	
	public void backpropOutput(float[] expected) {
		for (int i = 0; i < numOfOutputs; i++)
			error[i] = outputs[i] - expected[i];
		
		for (int i = 0; i < numOfOutputs; i++) 
			gamma[i] = error[i] * Deriv(outputs[i]);
		
		for (int i = 0; i < numOfOutputs; i++) {
			for (int j = 0; j < numOfInputs; j++)
				weightsDelta[i][j] = gamma[i] * inputs[j];
			
			biasDelta[i] = gamma[i] * bias[i];
		}
	}
	
	public void backpropHidden(float[] gammaForward, float[][] weightsForward) {
		for (int i = 0; i < numOfOutputs; i++) {
			gamma[i] = 0;
			
			for (int j = 0; j < gammaForward.length; j++)
				gamma[i] += gammaForward[j] * weightsForward[j][i];
			
			gamma[i] *= Deriv(outputs[i]);
		}
		
		for (int i = 0; i < numOfOutputs; i++) {
			for (int j = 0; j < numOfInputs; j++)
				weightsDelta[i][j] = gamma[i] * inputs[j];
			
			biasDelta[i] = gamma[i] * bias[i];
		}
	}
	
	public void UpdateWeights() {
		for (int i = 0; i < numOfOutputs; i++) {
			for (int j = 0; j < numOfInputs; j++) 
				weights[i][j] -= weightsDelta[i][j] * LEARNING_RATE;
			
			bias[i] -= biasDelta[i] * LEARNING_RATE;
		}
	}
	
	public void mutate() {
		for (int i = 0; i < numOfOutputs; i++) {
			for (int j = 0; j < numOfInputs; j++) {
				//weights[i][j] = 
			}
		}
	}
}
