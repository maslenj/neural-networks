/**
 * @author Jimmy Maslen
 */

import java.io.*;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

public class NetworkOperator {
    static String trainingDataSetName = "digits-train.txt";
    static String testingDataSetName = "digits-test.txt";
    static boolean MNISTData = false;

    static double validationPercent = 1;         //Set validationPercent to 1 for AND and XOR
    static int numHiddenNeurons = 128;
    static double learningRate = 0.03;
    static double desiredAccuracy = 0.97;
    static int printFrequency = 1;

    public static void main(String[] args){

        List<Example> trainingExamples;
        List<Example> testingExamples;

        if(MNISTData){
            trainingExamples = readDataFromImage("train-labels-idx1-ubyte", "train-images-idx3-ubyte");
            testingExamples = readDataFromImage("t10k-labels-idx1-ubyte", "t10k-images-idx3-ubyte");
        } else {
            trainingExamples = readDataFromFile(trainingDataSetName);
            testingExamples = readDataFromFile(testingDataSetName);
        }
        List<Example> validationExamples = new ArrayList<>();

        if(validationPercent == 1){
            validationExamples = trainingExamples;
        } else {
            int validationExamplesSize = (int) (trainingExamples.size()*validationPercent);
            for(int i = 0; i < validationExamplesSize; i++){
                int randomInt = (int) (Math.random()*trainingExamples.size());
                validationExamples.add(trainingExamples.get(randomInt));
                trainingExamples.remove(randomInt);
            }
        }

        NeuralNet myNeuralNet = buildNeuralNet(trainingExamples, numHiddenNeurons, learningRate);

        //Training
        myNeuralNet.learnManyExamples(trainingExamples, validationExamples, desiredAccuracy, printFrequency);
        System.out.println();

        //Testing
        System.out.println("Testing Accuracy: " + myNeuralNet.calculateAccuracy(testingExamples)*100 + "%");
    }

    public static ArrayList<Example> readDataFromFile(String data){
        ArrayList<Example> examples = new ArrayList<>();
        SimpleFile file = new SimpleFile(data);

        for(String line: file){

            String[] wordsInLine = line.split(",");

            ArrayList<Double> attributesList = new ArrayList<>();
            for(int i = 0; i < wordsInLine.length - 1; i++){
                attributesList.add(Double.parseDouble(wordsInLine[i]));
            }

            int category = Integer.parseInt(wordsInLine[wordsInLine.length-1]);

            double[] attributesArray =  new double[attributesList.size()];
            for(int i = 0; i < attributesList.size(); i++){
                attributesArray[i] = attributesList.get(i);
            }

            Example newExample = new Example(category, attributesArray);

            examples.add(newExample);
        }

        return examples;
    }

    static List<Example> readDataFromImage(String labelFileName, String imageFileName) {
        DataInputStream labelStream = openFile(labelFileName, 2049);
        DataInputStream imageStream = openFile(imageFileName, 2051);

        List<Example> examples = new ArrayList<>();

        try {
            int numLabels = labelStream.readInt();
            int numImages = imageStream.readInt();
            assert(numImages == numLabels) : "lengths of label file and image file do not match";

            int rows = imageStream.readInt();
            int cols = imageStream.readInt();
            assert(rows == cols) : "images in file are not square";
            assert(rows == 28) : "images in file are wrong size";

            for (int i = 0; i < numImages; i++) {
                int categoryLabel = Byte.toUnsignedInt(labelStream.readByte());
                double[] inputs = new double[rows * cols];
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < cols; c++) {
                        int pixel = 255 - Byte.toUnsignedInt(imageStream.readByte());
                        inputs[r * rows + c] = pixel / 255.0;
                    }
                }
                examples.add(new Example(categoryLabel, inputs));
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return examples;
    }

    static DataInputStream openFile(String fileName, int expectedMagicNumber) {
        DataInputStream stream = null;
        try {
            stream = new DataInputStream(new BufferedInputStream(new FileInputStream(fileName)));
            int magic = stream.readInt();
            if (magic != expectedMagicNumber) {
                throw new RuntimeException("file " + fileName + " contains invalid magic number");
            }
        } catch (FileNotFoundException e) {
            throw new RuntimeException("file " + fileName + " was not found");
        } catch (IOException e) {
            throw new RuntimeException("file " + fileName + " had exception: " + e);
        }
        return stream;
    }

    public static NeuralNet buildNeuralNet(List<Example> examples, int numHiddenNeurons, double learningRate){
        HashSet<Integer> categories = new HashSet<>();
        for(Example example: examples){
            categories.add(example.category);
        }

        NeuralNet thisNet = new NeuralNet(examples.get(0).attributes.length, numHiddenNeurons, categories, learningRate);
        return thisNet;
    }
}
