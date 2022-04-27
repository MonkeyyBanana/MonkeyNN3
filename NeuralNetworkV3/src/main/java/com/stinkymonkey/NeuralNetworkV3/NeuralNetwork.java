package com.stinkymonkey.NeuralNetworkV3;

public class NeuralNetwork {
	int[] layer;
	public NetLayer[] layers;
	
	public NeuralNetwork(int[] layer) {
		this.layer = new int[layer.length];
		for (int i = 0; i < layer.length; i++)
			this.layer[i] = layer[i];
		
		layers = new NetLayer[layer.length - 1];
		
		for (int i = 0; i < layers.length; i++) {
			layers[i] = new NetLayer(layer[i], layer[i + 1]);
		}
	}
	
	public NeuralNetwork(NeuralNetwork nn) {
		this.layer = new int[nn.layers.length];
		
		for (int i = 0; i < nn.layers.length; i++) {
			this.layers[i] = nn.layers[i];
			this.layers[i].initWeights();
		}
	}
	
	public void setLayerMMBias(int layer, float min, float max) {
		layers[layer].MIN_BIAS = min;
		layers[layer].MAX_BIAS = max;
	}
	
	public float[] feedForward(float[] inputs) {
		layers[0].feedForward(inputs);
		
		for (int i = 1; i < layers.length; i++) {
			layers[i].feedForward(layers[i - 1].outputs);
		}
		
		return layers[layers.length - 1].outputs;
	}
	
	public void backProp(float[] expected) {
		for (int i = layers.length - 1; i >= 0; i--) {
			if (i == layers.length - 1) 
				layers[i].backpropOutput(expected);
			else
				layers[i].backpropHidden(layers[i + 1].gamma, layers[i + 1].weights);
		}
		
		for (int i = 0; i < layers.length; i++)
			layers[i].UpdateWeights();
	}
}
