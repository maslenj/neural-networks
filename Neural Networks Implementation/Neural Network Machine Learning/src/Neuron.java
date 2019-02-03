/**
 * @author Jimmy Maslen
 */

import java.util.ArrayList;

public class Neuron {
    Double[] weights;
    double mostRecentOutput;
    double currentErrorSignal;

    Neuron(int numInputs){
        weights = new Double[numInputs+1];
        for(int i = 0; i < weights.length; i++){
            weights[i] = 0.05;
        }
    }

    public double getOutput(double[] inputs){
        double outputValue = 0.0;

        outputValue += weights[0];

        for(int i = 1; i < inputs.length+1; i++){
            outputValue += inputs[i-1]*weights[i];
        }

        double output = sigmoidFunction(outputValue);

        mostRecentOutput = output;
        return output;
    }

    public double sigmoidFunction(double value){
        return 1/(1+Math.exp(-value));
    }

}
