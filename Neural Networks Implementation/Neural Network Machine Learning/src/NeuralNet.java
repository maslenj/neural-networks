import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

/**
 * @author Jimmy Maslen
 */

public class NeuralNet {

    double learningRate;

    Neuron hiddenNeurons[];
    OutputNeuron outputNeurons[];

    public NeuralNet(int numInputs, int numHiddenNeurons, HashSet<Integer> categories, double learningRate){
        this.learningRate = learningRate;

        hiddenNeurons = new Neuron[numHiddenNeurons];
        for(int i = 0; i < numHiddenNeurons; i++){
            Neuron newHiddenNeuron = new Neuron(numInputs);
            hiddenNeurons[i] = newHiddenNeuron;
        }

        int currentIteration = 0;
        outputNeurons = new OutputNeuron[categories.size()];
        for(Integer category: categories){
            OutputNeuron newOutputNeuron = new OutputNeuron(numHiddenNeurons, category);
            outputNeurons[currentIteration] = newOutputNeuron;
            currentIteration ++;
        }

    }

    /**
     * takes one example and returns the category that the neural net thinks it belongs to
     */
    public int classifyOneExample(Example example){
        double[] hiddenNeuronOutputs = new double[hiddenNeurons.length];
        for(int i = 0; i < hiddenNeuronOutputs.length; i++){
            hiddenNeuronOutputs[i] = hiddenNeurons[i].getOutput(example.attributes);
        }

        ArrayList<Double> outputNeuronOutputs = new ArrayList<>();
        for(OutputNeuron outputNeuron: outputNeurons){
            outputNeuronOutputs.add(outputNeuron.getOutput(hiddenNeuronOutputs));
        }

        double greatestActivation = 0;
        int indexOfBestOutput = 0;
        for(int i = 0; i < outputNeuronOutputs.size(); i++){
            if(outputNeuronOutputs.get(i) > greatestActivation){
                indexOfBestOutput = i;
                greatestActivation = outputNeuronOutputs.get(i);
            }
        }

        return outputNeurons[indexOfBestOutput].category;
    }

    /**
     * takes a list of pre-categorized testing examples, and returns the fraction (or percent) that are correctly classified by this network
     */
    public double calculateAccuracy(List<Example> examples){
        int numCorrect = 0;
        for(Example example: examples){
            if(classifyOneExample(example) == example.category){
                numCorrect += 1;
            }
        }
        return ((double)numCorrect/examples.size());
    }

    /**
     * takes one pre-categorized example, classifies it, and then learns by modifying the network's weights
     */
    public void learnOneExample(Example example){
        classifyOneExample(example);

        //Calculate Error Signals
        for(OutputNeuron outputNeuron: outputNeurons){
            double correctOutput;
            if(example.category == outputNeuron.category){
                correctOutput = 1;
            } else {
                correctOutput = 0;
            }

            double actualOutput = outputNeuron.mostRecentOutput;

            outputNeuron.currentErrorSignal = (correctOutput-actualOutput)*(actualOutput)*(1-actualOutput);

        }

        int numNeuron = 0;
        for(Neuron hiddenNeuron: hiddenNeurons){
            double outputError = 0;
            for(OutputNeuron outputNeuron: outputNeurons){
                outputError += outputNeuron.weights[numNeuron+1]*outputNeuron.currentErrorSignal;
            }
            double hiddenOutput = hiddenNeuron.mostRecentOutput;
            hiddenNeuron.currentErrorSignal = outputError * (hiddenOutput * (1-hiddenOutput));
            numNeuron ++;
        }

        //Update weights
        for(OutputNeuron outputNeuron: outputNeurons){
            for(int numWeight = 0; numWeight < outputNeuron.weights.length; numWeight++){
                double outputErrorSignal = outputNeuron.currentErrorSignal;
                double hiddenOutput = 1;

                if(numWeight > 0){
                    hiddenOutput = hiddenNeurons[numWeight-1].mostRecentOutput;
                }
                outputNeuron.weights[numWeight] += outputErrorSignal * hiddenOutput * learningRate;
            }
        }
        for(Neuron hiddenNeuron: hiddenNeurons){
            for(int numWeight = 0; numWeight < hiddenNeuron.weights.length; numWeight++){
                double hiddenErrorSignal = hiddenNeuron.currentErrorSignal;
                double input = 1;
                if(numWeight > 0){
                    input = example.attributes[numWeight-1];
                }
                hiddenNeuron.weights[numWeight] += hiddenErrorSignal * input * learningRate;
            }
        }
    }

    /**
     *  takes a list of pre-categorized training examples (and maybe a list of pre-categorized validation examples), and a desired accuracy and/or time limit.
     *  It repeatedly learns from the training examples until a termination condition is reached.
     */
    public void learnManyExamples(List<Example> trainingExamples, List<Example> validationExamples, double desiredAccuracy, int printFrequency){
        int numEpoch = 0;
        int i = 0;
        boolean done = false;
        double accuracy = calculateAccuracy(validationExamples)*100;
        System.out.println("Number of Epochs: " + 0 + ", Training Accuracy: " + accuracy + "%");

        while(!done){ //
            if(i == trainingExamples.size()){
                i = 0;
                accuracy = calculateAccuracy(validationExamples)*100;
                if(accuracy >= desiredAccuracy*100){
                    done = true;
                    int numberEpoch = numEpoch + 1;
                    System.out.println("Number of Epochs: " + numberEpoch + ", Training Accuracy: " + accuracy + "%");
                } else if((numEpoch+1)%printFrequency == 0){
                    int numberEpoch = numEpoch + 1;
                    System.out.println("Number of Epochs: " + numberEpoch + ", Training Accuracy: " + accuracy + "%");
                }
                numEpoch++;
            } else {
                learnOneExample(trainingExamples.get(i));
                i++;
            }
        }
    }


}