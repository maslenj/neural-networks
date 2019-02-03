/**
 * @author Jimmy Maslen
 */

import java.util.Arrays;

public class Example {
    public int category;
    public double[] attributes;

    public Example(int category, double[] attributes){
        this.category = category;
        this.attributes = attributes;
    }

    public String toString() {
        String thisString = "category: " + category + ", attributes: " + Arrays.toString(attributes);
        return thisString;
    }
}
