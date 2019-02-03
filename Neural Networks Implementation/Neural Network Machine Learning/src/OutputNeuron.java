import java.util.ArrayList;

/**
 * @author Jimmy Maslen
 */

public class OutputNeuron extends Neuron {
    int category;

    public OutputNeuron(int numInputs, int category){
        super(numInputs);
        this.category = category;
    }
}
