import java.awt.image.BufferedImage;
import javax.swing.JFrame;

import com.github.sarxos.webcam.Webcam;
import com.github.sarxos.webcam.WebcamPanel;
import com.github.sarxos.webcam.WebcamResolution;

public class SimpleWebcam {

    private Webcam webcam;

    public static void main(String[] args)
    {
        new SimpleWebcam();
    }

    SimpleWebcam()
    {
        webcam = Webcam.getDefault();
        webcam.setViewSize(WebcamResolution.VGA.getSize());

        WebcamPanel panel = new WebcamPanel(webcam);

        JFrame frame = new JFrame("Webcam");
        frame.getContentPane().add(panel);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setResizable(false);
        frame.pack();
        frame.setVisible(true);
    }

    BufferedImage getImage()
    {
        return webcam.getImage();
    }

}